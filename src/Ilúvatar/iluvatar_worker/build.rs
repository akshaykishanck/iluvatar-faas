use std::env;
use std::error::Error;
use std::path::{Path, PathBuf};

fn copy_file(infile: &Path) -> Result<(), Box<dyn Error>> {
    // 1. Get the absolute path to the iluvatar_worker crate root
    let manifest_dir = env::var("CARGO_MANIFEST_DIR")?;
    let full_infile_path = Path::new(&manifest_dir).join("src").join(infile);

    // 2. Get Cargo's official target output directory for this build
    // This points deep inside target/debug/build/iluvatar_worker-.../out
    let out_dir = env::var("OUT_DIR")?;

    // 3. Walk back up to the actual profile directory (target/debug/ or target/release/)
    let mut output_path = PathBuf::from(out_dir);
    while !output_path.ends_with("build") {
        output_path.pop();
    }
    output_path.pop(); // Pop "build" itself to land in target/debug/ or target/release/

    // 4. Append the file name
    let destination = output_path.join(infile.file_name().unwrap());

    // 5. Copy the file safely
    std::fs::copy(full_infile_path, destination)?;

    // Tell Cargo to rerun this script if worker.json changes
    println!("cargo:rerun-if-changed=src/{}", infile.display());

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    copy_file(Path::new("worker.json"))?;
    Ok(())
}
