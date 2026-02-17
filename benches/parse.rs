//! Benchmarks for GGUF parsing
//!
//! Run with: cargo bench

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use gguf_rs::{get_gguf_container, get_gguf_container_array_size};

/// Benchmark parsing a small GGUF file
fn bench_parse_small(c: &mut Criterion) {
    c.bench_function("parse_small_file", |b| {
        b.iter(|| {
            let mut container = get_gguf_container(black_box("tests/test-le-v3.gguf")).unwrap();
            black_box(container.decode().unwrap())
        })
    });
}

/// Benchmark parsing with full array size
fn bench_parse_full_arrays(c: &mut Criterion) {
    c.bench_function("parse_full_arrays", |b| {
        b.iter(|| {
            let mut container = get_gguf_container_array_size(
                black_box("tests/test-le-v3.gguf"),
                black_box(u64::MAX),
            )
            .unwrap();
            black_box(container.decode().unwrap())
        })
    });
}

/// Benchmark metadata access
fn bench_metadata_access(c: &mut Criterion) {
    let mut container = get_gguf_container("tests/test-le-v3.gguf").unwrap();
    let model = container.decode().unwrap();

    c.bench_function("metadata_access", |b| {
        b.iter(|| {
            black_box(model.metadata().get("general.architecture"));
            black_box(model.metadata().get("llama.block_count"));
            black_box(model.metadata().get("tokenizer.ggml.tokens"));
        })
    });
}

/// Benchmark tensor iteration
fn bench_tensor_iteration(c: &mut Criterion) {
    let mut container = get_gguf_container("tests/test-le-v3.gguf").unwrap();
    let model = container.decode().unwrap();

    c.bench_function("tensor_iteration", |b| {
        b.iter(|| {
            for tensor in model.tensors() {
                black_box(&tensor.name);
                black_box(&tensor.shape);
            }
        })
    });
}

/// Benchmark model info methods
fn bench_model_info(c: &mut Criterion) {
    let mut container = get_gguf_container("tests/test-le-v3.gguf").unwrap();
    let model = container.decode().unwrap();

    c.bench_function("model_info", |b| {
        b.iter(|| {
            black_box(model.get_version());
            black_box(model.model_family());
            black_box(model.model_parameters());
            black_box(model.file_type());
            black_box(model.num_tensor());
            black_box(model.num_kv());
        })
    });
}

criterion_group!(
    benches,
    bench_parse_small,
    bench_parse_full_arrays,
    bench_metadata_access,
    bench_tensor_iteration,
    bench_model_info,
);

criterion_main!(benches);
