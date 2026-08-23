# Random Forest (RF) Latency Predictor in Ilúvatar FaaS

This document serves as an end-to-end guide on how the machine learning-based Random Forest (RF) latency estimation model is exported, embedded, integrated, and utilized within Ilúvatar's runtime system.

---

## Executive Summary & Architecture Overview

Ilúvatar is a high-performance Serverless (FaaS) framework designed for heterogeneous compute (CPU and GPU). When dispatching incoming serverless function invocations, scheduling policies must decide whether to execute a function on a GPU or CPU, and whether to admit or evict function environments from cache.

To make optimal dispatching decisions (specifically in the **Landlord** caching and dispatch policy), Ilúvatar estimates the **Opportunity Cost** of GPU execution versus CPU execution:
$$\text{Opportunity Cost} = \text{CPU E2E Latency} - \text{GPU E2E Latency}$$

Rather than relying purely on static heuristics, Ilúvatar embeds a machine learning model in the form of a **Random Forest (RF) ONNX model** directly into the worker runtime. The model predicts the **total GPU end-to-end (E2E) latency** given real-time queue contention, function inter-arrival times, running functions, benchmark execution times, and container warm/cold state.

```
+-----------------------------------------------------------------------------------+
| 1. Python Offline Pipeline                                                        |
|   Trained Scikit-Learn RF Model ---> skl2onnx Export ---> iluvatar_rf_estimator.onnx  |
+-----------------------------------------------------------------------------------+
                                   |
                                   v (include_bytes!)
+-----------------------------------------------------------------------------------+
| 2. Rust Worker Runtime (iluvatar_library)                                         |
|   +---------------------------------------------------------------------------+   |
|   | RfModel Wrapper (rf_model.rs)                                             |   |
|   |   - Uses 'ort' (ONNX Runtime C++ bindings)                                |   |
|   |   - Transforms log-latency output back to seconds via .exp_m1()           |   |
|   +---------------------------------------------------------------------------+   |
|                                  |                                                |
|                                  v (Embedded into)                                |
|   +---------------------------------------------------------------------------+   |
|   | WorkerCharMap / CharMapRW (char_map.rs)                                   |   |
|   |   - Centralized system metrics & statistics repository                    |   |
|   |   - Exposes trait method get_rf_prediction(...)                           |   |
|   +---------------------------------------------------------------------------+   |
+-----------------------------------------------------------------------------------+
                                   |
                                   v (Invoked by)
+-----------------------------------------------------------------------------------+
| 3. Landlord Dispatcher (landlord.rs in iluvatar_worker_library)                   |
|   - Collects live runtime features (MQFQ queue sizes, inflight counts, IAT, etc.)|
|   - Queries CharMap for RF prediction -> gpu_est_total                            |
|   - Calculates Opportunity Cost & manages cache admission / eviction           |
+-----------------------------------------------------------------------------------+
```

---

## 1. Model Training & ONNX Export (Python Pipeline)

The RF model is initially trained offline in Python using `scikit-learn` on collected Ilúvatar trace logs.

### 1.1 Target Variable Transformation
To handle skewed latency distributions spanning orders of magnitude, the target variable during model training is the **natural log of latency plus one**:
$$y_{\text{train}} = \ln(\text{latency}_{\text{seconds}} + 1)$$

### 1.2 Export Script
Once trained, the `scikit-learn` model is serialized into standard ONNX format using `skl2onnx`:

```python
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# 1. Load trained scikit-learn model
rf_model = joblib.load("tuned_faas_rf_proportionate_data_7_features_final.joblib")
final_features = list(rf_model.feature_names_in_) # 7 features

# 2. Define input tensor schema (float_input with shape [batch_size, 7])
initial_type = [('float_input', FloatTensorType([None, len(final_features)]))]

# 3. Convert sklearn model to ONNX
onnx_model = convert_sklearn(rf_model, initial_types=initial_type)

# 4. Save ONNX binary file into Ilúvatar resource path
onnx_path = "src/Ilúvatar/iluvatar_worker_library/src/resources/iluvatar_rf_estimator_7_features.onnx"
with open(onnx_path, "wb") as f:
    f.write(onnx_model.SerializeToString())
```

### 1.3 Target ONNX Location in Repository
The exported binary file must be placed at:
`src/Ilúvatar/iluvatar_worker_library/src/resources/iluvatar_rf_estimator_7_features.onnx`

---

## 2. Rust ONNX Inference Engine (`RfModel` in `rf_model.rs`)

Location: `src/Ilúvatar/iluvatar_library/src/rf_model.rs`

The `RfModel` struct provides a thread-safe wrapper around the `ort` crate (ONNX Runtime bindings for Rust).

### 2.1 Struct Definition & Thread Safety
```rust
pub struct RfModel {
    session: Mutex<Session>,
}
```
* **Thread Safety**: An `ort::session::Session` is wrapped in a `parking_lot::Mutex` and managed via `Arc<RfModel>`, allowing concurrent calls from worker dispatcher threads.

### 2.2 Model Loading Methods
`RfModel` supports loading either from a file path or directly from memory bytes:
* `RfModel::new(model_path: &str) -> Result<Arc<Self>>`: Loads from disk at runtime.
* `RfModel::from_bytes(model_bytes: &[u8]) -> Result<Arc<Self>>`: Loads in-memory model bytes (enabling `include_bytes!`).

### 2.3 Prediction & Target Unit Inverse Transformation
The core inference logic is in `predict(...)`:

```rust
pub fn predict(
    &self,
    target_queue_len: f32,
    others_len_queue: f32,
    iat_fqdn: f32,
    num_running_funcs: f32,
    gpu_warm_results_sec: f32,
    gpu_cold_results_sec: f32,
    is_cold_start: f32,
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

    let input_tensor = Tensor::<f32>::from_array(([1usize, NUM_FEATURES], features)).ok()?;

    let mut session = self.session.lock();
    let outputs = session.run(ort::inputs!["float_input" => input_tensor]).ok()?;

    let output_array = outputs["variable"].try_extract_tensor::<f32>().ok()?;
    let raw_pred = output_array.1[0];

    // Convert raw prediction log(latency + 1) back to seconds using exp_m1()
    let latency_secs = (raw_pred as f64).exp_m1();
    Some(latency_secs)
}
```

> [!IMPORTANT]
> **Log Inverse Transformation**: Because the model was trained on $\ln(\text{latency} + 1)$, the predicted value `raw_pred` represents $\ln(\hat{y} + 1)$. To recover predicted latency in seconds:
> $$\hat{y} = e^{\text{raw\_pred}} - 1$$
> In Rust, this is executed using `.exp_m1()`, which provides high floating-point precision for values near zero.

---

## 3. Feature Schema Reference Table

The model expects exactly **7 float features** in strict column order:

| Index | Feature Name | Rust Data Type | Source in Runtime | Description / Unit |
| :---: | :--- | :---: | :--- | :--- |
| **0** | `target_queue_len` | `f32` | `MQFQ` queue length for target FQDN | Number of pending requests queued specifically for this function. |
| **1** | `others_len_queue` | `f32` | Sum of `MQFQ` queue lengths for all other FQDNs | Measure of total background contention across other functions on the worker. |
| **2** | `iat_fqdn` | `f32` | `CharMap::get_latest(fqdn, Chars::IAT)` | Inter-Arrival Time (IAT) of requests for this FQDN in seconds. |
| **3** | `num_running_funcs` | `f32` | `gpu_inflight_count()` from `gpu_queue` | Real-time count of invocations currently actively executing on the GPU (`in_flight`). |
| **4** | `gpu_warm_results_sec` | `f32` | `CharMap::get_avg(fqdn, Chars::GpuWarmTime)` | Historical benchmark average warm GPU execution time in seconds. |
| **5** | `gpu_cold_results_sec` | `f32` | `CharMap::get_avg(fqdn, Chars::GpuColdTime)` | Historical benchmark average cold GPU execution time in seconds. |
| **6** | `is_cold_start` | `f32` | Container state check (`1.0` if `Cold`, `0.0` otherwise) | Binary flag indicating whether container allocation requires a cold start. |

---

## 4. Integration into Characterization Map (`CharMap` in `char_map.rs`)

Location: `src/Ilúvatar/iluvatar_library/src/char_map.rs`

### 4.1 Why Embed inside `CharMap`?
`WorkerCharMap` is Ilúvatar's shared runtime statistics repository. Passing `WorkerCharMap` across worker dispatching modules is already standard practice. By embedding `RfModel` inside `CharMap`:
1. The ONNX model binary is compiled directly into the Rust executable via `include_bytes!`. No disk path dependencies or file loading failures at runtime.
2. Any dispatching policy with access to `WorkerCharMap` can query ML predictions via a single unified trait call.

### 4.2 How it is Initialized
In `CharMapRW::boxed()`:
```rust
rf_model: RfModel::from_bytes(
    include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../iluvatar_worker_library/src/resources/iluvatar_rf_estimator_7_features.onnx"
    ))
).ok(),
```

### 4.3 Trait Interface Extension
The `CharMap` trait exposes a default method:
```rust
fn get_rf_prediction(
    &self,
    target_queue_len: f32,
    others_len_queue: f32,
    iat_fqdn: f32,
    num_running_funcs: f32,
    gpu_warm_results_sec: f32,
    gpu_cold_results_sec: f32,
    is_cold_start: f32,
) -> Option<f64> {
    self.rf_model.as_ref()?.predict(
        target_queue_len, others_len_queue, iat_fqdn, num_running_funcs,
        gpu_warm_results_sec, gpu_cold_results_sec, is_cold_start
    )
}
```

---

## 5. Usage in Landlord Policy (`landlord.rs`)

Location: `src/Ilúvatar/iluvatar_worker_library/src/services/invocation/dispatching/landlord.rs`

The Landlord cache-aware dispatch policy uses `opp_cost(...)` to calculate credits and rent for function caching and device selection.

### 5.1 How Landlord Features Are Extracted
Inside `Landlord::opp_cost`:

```rust
// 1. Queue features from live Multi-Queue Fair Queueing (MQFQ)
let (target_queue_len, others_len_queue) = match self.gpu_queue.expose_mqfq() {
    None => (0.0_f32, 0.0_f32),
    Some(mqfq) => {
        let tq = mqfq.get(&reg.fqdn).map_or(0, |fq| fq.queue.len()) as f32;
        let oq: f32 = mqfq.iter()
            .filter(|entry| entry.key() != &reg.fqdn)
            .map(|entry| entry.value().queue.len() as f32)
            .sum();
        (tq, oq)
    }
};

// 2. Inter-arrival time & active running functions
let iat_fqdn = self.cmap.get_latest(&reg.fqdn, Chars::IAT) as f32;
let num_running = self.gpu_inflight_count() as f32;

// 3. Execution time benchmarks
let gpu_warm = self.cmap.get_avg(&reg.fqdn, Chars::GpuWarmTime) as f32;
let gpu_cold = self.cmap.get_avg(&reg.fqdn, Chars::GpuColdTime) as f32;

// 4. Physical container state check
let physical_state = self.cont_manager.container_available(&reg.fqdn, Compute::GPU);
let is_cold_start: f32 = if matches!(physical_state, ContainerState::Cold) { 1.0 } else { 0.0 };

// 5. Query RF Model prediction with fallback
let rf_prediction = self.cmap.get_rf_prediction(
    target_queue_len, others_len_queue,
    iat_fqdn, num_running,
    gpu_warm, gpu_cold, is_cold_start,
);

let gpu_est_total = match rf_prediction {
    None => fallback_gpu_est_total, // Fall back to simple heuristic if model missing/fails
    Some(pred) => pred,
};
```

---

## 6. How to Inspect Logs & Debug

Ilúvatar uses the `tracing` framework. Latency predictions and Landlord decisions are logged under `INFO` level.

### 6.1 Key Log Events
1. **Model Loading Event** (at worker initialization):
   ```text
   INFO iluvatar_library::rf_model: RF model loaded from memory bytes
   ```
2. **RF Model Latency Prediction Log** (in `landlord.rs`):
   ```text
   INFO iluvatar_worker_library::services::invocation::dispatching::landlord: RF Model Latency Prediction Log tid=... fqdn="helloworld" target_queue_len=1.0 others_len_queue=3.0 iat_fqdn=0.4 num_running=4.0 gpu_warm=2.1 gpu_cold=3.8 is_cold_start=0.0 avg_mem=512.0 sum_memory_running=2048.0 rf_prediction=Some(0.2451) fallback_prediction=0.3150
   ```
3. **Landlord Credit Log**:
   ```text
   INFO iluvatar_worker_library::services::invocation::dispatching::landlord: Landlord Credit tid=... fqdn="helloworld" is_warm_gpu=true mqfq_est=0.21 gpu_est=0.21 gpu_adj_est=0.21 gpu_est_err=0.01 cpu_est=0.85 cpu_exec=0.85 gpu_est_total=0.2451 cpu_est_total=0.85
   ```

### 6.2 Useful Commands to View Logs
When running a worker binary or test suite:
```bash
# View RF model predictions in real time
RUST_LOG=info cargo run --bin iluvatar_worker 2>&1 | grep "RF Model Latency Prediction Log"

# Filter for prediction values vs fallbacks
grep "rf_prediction" worker_output.log
```

---

## 7. Standalone Testing

You can run the built-in smoke test in `rf_model.rs` without launching the full worker cluster.

> [!NOTE]
> **ONNX Runtime Dynamic Library Dependency**:
> The `ort` crate requires `libonnxruntime.so` to be available on your system's dynamic linker search path (e.g., `LD_LIBRARY_PATH`) or pointed to by `ORT_DYLIB_PATH`. If `ort` fails with `failed to load from libonnxruntime.so: dlopen failed`, set `ORT_DYLIB_PATH` to your ONNX Runtime shared library location:

```bash
# On this system, set ORT_DYLIB_PATH to the ONNX Runtime library:
export ORT_DYLIB_PATH=/path/to/onnxruntime/lib/libonnxruntime.so

# Run standalone smoke test
cargo test -p iluvatar_library rf_model::tests::smoke_test_all_scenarios -- --nocapture
```

### Example Test Output:
```text
──────────────────────────────────────────
SCENARIO 1 — Warm hit, short queue
  Predicted e2etime : 4.3732 s  (4373.2 ms)
──────────────────────────────────────────
SCENARIO 2 — Cold start, empty queue
  Predicted e2etime : 1.8383 s  (1838.3 ms)
──────────────────────────────────────────
SCENARIO 3 — heavy contention
  Predicted e2etime : 15.7703 s  (15770.3 ms)
All inference tests completed.
test rf_model::tests::smoke_test_all_scenarios ... ok
```

---

## 8. Limitations & Scope for Future Improvement

Following are the key architectural limitations to be aware of and opportunities for research/development:


### Current Limitations

1. **Integrated Only in Landlord**: Currently, `get_rf_prediction(...)` is explicitly called inside `landlord.rs`. Other dispatchers (`mice.rs`, `greedy_weight.rs`, `epsilon_greedy.rs`, `queueing_dispatcher.rs`) still rely on static or linear regression estimators.

2. **Static Model Weights**: The ONNX model is embedded at compile-time via `include_bytes!`. Updates to the model require re-exporting the `.onnx` file and re-compiling the binary.

3. **Single Worker Scope**: Feature extraction relies on local worker `MQFQ` queues and local `CharMap`. Global multi-worker cross-node contention is not currently factored into the 7 feature inputs.

### Scope for Future Improvements
1. **Extend to Other Dispatching Policies**: Integrate `get_rf_prediction(...)` into `MICE`, `GreedyWeight`, and `EpsilonGreedy` dispatchers to evaluate ML-based latency estimations across all scheduling algorithms.
2. **Dynamic Model Reloading**: Enhance `RfModel` to support dynamic loading from an external path or network endpoint without recompiling the worker crate.
3. **Feature Engineering Expansion**: Incorporate GPU memory usage (`avg_mem`, `sum_memory_running`) or CPU queue load features into a retraining pipeline for higher accuracy during heavy multi-tenant memory saturation.
