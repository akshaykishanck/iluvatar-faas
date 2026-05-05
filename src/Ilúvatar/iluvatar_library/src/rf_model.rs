//! RF Model wrapper for iluvatar_rf_estimator.onnx
//!
//! Exposes `RfModel` — a loadable, reusable inference handle that any
//! dispatching policy can call to get a predicted GPU e2e latency.
//!
//! Feature layout (18 features, must match training column order):
//!   1  target_queue_len
//!   2  others_len_queue
//!   3  iat_fqdn
//!   4  num_running_funcs_filled
//!   5  gpu_warm_results_sec
//!   6  gpu_cold_results_sec
//!   7  is_cold_start

use anyhow::Result;
use ort::session::Session;
use ort::value::Tensor;
use std::sync::Arc;
use parking_lot::Mutex;
use tracing::{info, warn};

pub const NUM_FEATURES: usize = 7;

// ─────────────────────────────────────────────────────────────────────────────
// RfModel
// ─────────────────────────────────────────────────────────────────────────────

/// Thread-safe RF inference handle. Load once, share via `Arc<RfModel>`.
pub struct RfModel {
    session: Mutex<Session>,
}

impl RfModel {
    /// Load the ONNX model from `model_path`.
    pub fn new(model_path: &str) -> Result<Arc<Self>> {
        let session = Session::builder()?.commit_from_file(model_path)?;
        info!(model_path = model_path, "RF model loaded");
        Ok(Arc::new(Self { session: Mutex::new(session) }))
    }

    /// Run inference and return predicted e2e GPU latency in **seconds**.
    ///
    /// Returns `None` if inference fails so callers can fall back gracefully.
    ///
    /// # Arguments
    /// All values must match the feature column order used during training.
    pub fn predict(
        &self,
        target_queue_len: f32,
        others_len_queue: f32,
        iat_fqdn: f32,
        num_running_funcs: f32,
        gpu_warm_results_sec: f32,
        gpu_cold_results_sec: f32,
        is_cold_start: f32
    ) -> Option<f64> {
        let features: Vec<f32> = vec![
            target_queue_len,
            others_len_queue,
            iat_fqdn,
            num_running_funcs,
            gpu_warm_results_sec,
            gpu_cold_results_sec,
            is_cold_start,
        ];

        assert_eq!(features.len(), NUM_FEATURES);

        let input_tensor = Tensor::<f32>::from_array(([1usize, NUM_FEATURES], features))
            .map_err(|e| { warn!(error=%e, "RF: failed to build input tensor"); e })
            .ok()?;

        let mut session = self.session.lock();
        let outputs = session.run(ort::inputs!["float_input" => input_tensor])
            .map_err(|e| { warn!(error=%e, "RF: inference failed"); e })
            .ok()?;

        let output_array = outputs["variable"]
            .try_extract_tensor::<f32>()
            .map_err(|e| { warn!(error=%e, "RF: failed to extract output tensor"); e })
            .ok()?;

        let raw_pred = output_array.1[0];
        let latency_secs = (raw_pred as f64).exp_m1();
        info!(raw_pred, latency_secs, "RF prediction");
        Some(latency_secs)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Standalone smoke-test  (cargo test --nocapture)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const MODEL_PATH: &str =
        concat!(env!("CARGO_MANIFEST_DIR"), "/../iluvatar_worker_library/src/resources/iluvatar_rf_estimator_7_features.onnx");

    fn run_and_print(model: &Arc<RfModel>, label: &str,
        tq: f32, oq: f32, iat_f: f32, running: f32, warm: f32, cold: f32, cold_start: f32)
    {
        println!("──────────────────────────────────────────");
        println!("{}", label);
        match model.predict(tq, oq, iat_f, running, warm, cold, cold_start) {
            Some(secs) => {
                println!("  Predicted e2etime : {:.4} s  ({:.1} ms)", secs, secs * 1000.0);
            }
            None => println!("Inference failed"),
        }
    }

    #[test]
    fn smoke_test_all_scenarios() {
        let model = RfModel::new(MODEL_PATH).expect("Failed to load model");

        run_and_print(&model, "SCENARIO 1 — Warm hit, short queue",
            1.0, 3.0, 0.4, 4.0, 2.1, 3.8, 0.0);

        run_and_print(&model, "SCENARIO 2 — Cold start, empty queue",
            0.0, 0.0, 10.0, 0.0, 2.1, 3.8, 1.0);

        run_and_print(&model, "SCENARIO 3 — heavy contention",
            12.0, 35.0, 0.1, 20.0, 2.1, 3.8, 0.0);
        
        println!("All inference tests completed.");
    }
}

