//! Integration tests for the `gguf` CLI binary.

use std::process::Command;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_gguf")
}

#[test]
fn cli_prints_metadata_by_default() {
    let out = Command::new(bin())
        .arg("tests/test-le-v3.gguf")
        .output()
        .expect("run binary");
    assert!(out.status.success(), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("Metadata"));
    assert!(stdout.contains("general.architecture"));
}

#[test]
fn cli_prints_tensors_when_flag_set() {
    let out = Command::new(bin())
        .arg("tests/test-le-v3.gguf")
        .arg("--tensors")
        .output()
        .expect("run binary");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("Tensors"));
}

#[test]
fn cli_with_debug_flag() {
    // Exercises the SimpleLogger init branch.
    let out = Command::new(bin())
        .arg("tests/test-le-v3.gguf")
        .arg("--debug")
        .output()
        .expect("run binary");
    assert!(out.status.success());
}

#[test]
fn cli_fails_on_missing_file() {
    let out = Command::new(bin())
        .arg("definitely_not_here.gguf")
        .output()
        .expect("run binary");
    assert!(!out.status.success());
}

#[test]
fn cli_prints_metadata_with_array_values() {
    // test-le-v3.gguf contains tokenizer.ggml.tokens array — exercises the array
    // formatting branches of print_metadata.
    let out = Command::new(bin())
        .arg("tests/test-le-v3.gguf")
        .output()
        .expect("run binary");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("tokenizer.ggml.tokens"));
}
