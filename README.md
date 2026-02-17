# GGUF-RS

[![Crates.io](https://img.shields.io/crates/v/gguf-rs.svg)](https://crates.io/crates/gguf-rs)
[![Documentation](https://docs.rs/gguf-rs/badge.svg)](https://docs.rs/gguf-rs)
[![License](https://img.shields.io/crates/l/gguf-rs.svg)](https://github.com/zackshen/gguf/blob/main/LICENSE)
![Unit test](https://github.com/zackshen/gguf/actions/workflows/test.yml/badge.svg)
![Security Audit](https://github.com/zackshen/gguf/actions/workflows/audit.yml/badge.svg)
![Publish](https://github.com/zackshen/gguf/actions/workflows/publish.yml/badge.svg)
[![codecov](https://codecov.io/gh/zackshen/gguf/graph/badge.svg?token=REPLACE_WITH_TOKEN)](https://codecov.io/gh/zackshen/gguf)

A Rust library for parsing and reading GGUF (GGML Universal Format) files. GGUF files are binary files that contain key-value metadata and tensors, commonly used for storing quantized machine learning models.

## Features

- ✅ Decode GGUF files (v1, v2, v3)
- ✅ Access key-value metadata
- ✅ Access tensor information
- ✅ Support for little-endian and big-endian files
- ✅ CLI tool for quick inspection
- ✅ Zero-copy metadata access
- ✅ Memory-mapped file support (optional `mmap` feature)
- ✅ Async I/O support (optional `async` feature)
- ✅ Write GGUF files

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
gguf-rs = "0.1"
```

Or install the CLI tool:

```bash
cargo install gguf-rs
```

## Usage

### Library

**Basic usage:**

```rust
use gguf_rs::get_gguf_container;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Open a GGUF file
    let mut container = get_gguf_container("model.gguf")?;
    let model = container.decode()?;

    // Print model info
    println!("GGUF version: {}", model.get_version());
    println!("Architecture: {}", model.model_family());
    println!("Parameters: {}", model.model_parameters());
    println!("File type: {}", model.file_type());
    println!("Number of tensors: {}", model.num_tensor());

    Ok(())
}
```

**Access specific metadata:**

```rust
use gguf_rs::get_gguf_container;

let mut container = get_gguf_container("model.gguf")?;
let model = container.decode()?;

// Get specific metadata values
let metadata = model.metadata();
if let Some(arch) = metadata.get("general.architecture") {
    println!("Architecture: {}", arch);
}

// Check context length
if let Some(ctx_len) = metadata.get("llama.context_length") {
    println!("Context length: {}", ctx_len);
}
```

**Work with tensors:**

```rust
use gguf_rs::get_gguf_container;

let mut container = get_gguf_container("model.gguf")?;
let model = container.decode()?;

// List all tensors
for tensor in model.tensors() {
    println!("{}: shape={:?}, offset={}, size={}", 
        tensor.name, tensor.shape, tensor.offset, tensor.size);
}

// Find specific tensor
let embed_tensor = model.tensors()
    .iter()
    .find(|t| t.name.contains("token_embd"));
    
if let Some(tensor) = embed_tensor {
    println!("Embedding tensor shape: {:?}", tensor.shape);
}
```

**Read full tokenizer vocabulary:**

```rust
use gguf_rs::get_gguf_container_array_size;

// Use get_gguf_container_array_size to read full arrays
// (default get_gguf_container truncates arrays to 3 elements)
let mut container = get_gguf_container_array_size("model.gguf", u64::MAX)?;
let model = container.decode()?;

// Now you can access full tokenizer arrays
let metadata = model.metadata();
if let Some(tokens) = metadata.get("tokenizer.ggml.tokens") {
    println!("Vocabulary size: {:?}", tokens);
}
```

### CLI

Show model metadata:

```bash
gguf path_to_your_model.gguf
```

Show tensors:

```bash
gguf path_to_your_model.gguf --tensors
```

## Supported GGML Types

| Type | Description |
|------|-------------|
| F32 | 32-bit float |
| F16 | 16-bit float |
| Q4_0 | 4-bit quantization (type 0) |
| Q4_1 | 4-bit quantization (type 1) |
| Q5_0 | 5-bit quantization (type 0) |
| Q5_1 | 5-bit quantization (type 1) |
| Q8_0 | 8-bit quantization (type 0) |
| Q2_K - Q6_K | K-quant types |
| IQ series | I-quant types (IQ1_S, IQ2_XXS, etc.) |
| BF16 | Brain float 16 |

## API Documentation

Full API documentation is available at [docs.rs/gguf-rs](https://docs.rs/gguf-rs).

## Performance

- **Zero-copy metadata access**: Metadata is parsed once and stored in memory for fast repeated access
- **Lazy tensor data**: Tensor metadata is parsed, but actual tensor data is not loaded into memory
- **Array truncation**: By default, arrays in metadata are truncated to 3 elements for performance. Use `get_gguf_container_array_size()` with `u64::MAX` to read full arrays when needed

### Memory Usage

The library has minimal memory overhead:
- Metadata storage: O(n_kv + n_tensors) where n_kv = number of key-value pairs, n_tensors = number of tensors
- No tensor data is loaded into memory unless explicitly requested

### Memory-Mapped Files

For large GGUF files (multiple GB), use the `mmap` feature for efficient access:

```toml
[dependencies]
gguf-rs = { version = "0.1", features = ["mmap"] }
```

```rust,no_run
use gguf_rs::mmap::MmapGGUF;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mmap = MmapGGUF::open("large_model.gguf")?;
    let model = mmap.decode()?;
    
    println!("Architecture: {}", model.model_family());
    println!("Tensors: {}", model.num_tensor());
    
    Ok(())
}
```

Benefits of memory mapping:
- **Lazy loading**: Only accessed pages are loaded into memory
- **OS-managed paging**: The operating system handles memory management
- **Fast random access**: Direct pointer access to file data

### Async I/O

For async applications, enable the `async` feature:

```toml
[dependencies]
gguf-rs = { version = "0.1", features = ["async"] }
```

```rust,no_run
use gguf_rs::async_io::AsyncGGUF;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut container = AsyncGGUF::open("model.gguf").await?;
    let model = container.decode().await?;

    println!("Architecture: {}", model.model_family());
    println!("Tensors: {}", model.num_tensor());

    Ok(())
}
```

### Writing GGUF Files

Create and write GGUF files:

```rust,no_run
use gguf_rs::writer::{GGUFWriter, TensorInfo};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create writer for GGUF v3
    let mut writer = GGUFWriter::new("output.gguf", 3)?;

    // Add metadata
    writer.add_metadata("general.architecture", "llama");
    writer.add_metadata_u32("llama.block_count", 12);
    writer.add_metadata_f32("test.value", 3.14);

    // Add tensor info
    let tensor = TensorInfo {
        name: "token_embd.weight".to_string(),
        shape: vec![4096, 32000],
        dtype: 0, // F32
    };
    writer.add_tensor(tensor);

    // Write header and metadata
    writer.write()?;

    // Write tensor data
    let data: Vec<u8> = vec![0; 4096 * 32000 * 4]; // F32 = 4 bytes
    writer.write_tensor_data(0, &data)?;

    // Finalize
    writer.finalize()?;

    Ok(())
}
```

## Compatibility

- **Rust version**: Requires Rust 1.56+ (edition 2021)
- **GGUF versions**: Supports v1, v2, and v3
- **Byte order**: Both little-endian and big-endian files
- **Platforms**: Works on all platforms supported by Rust (Linux, macOS, Windows, BSD, etc.)

## Testing

```bash
cargo test
```

## Benchmarks

Run performance benchmarks:

```bash
cargo bench
```

Benchmarks measure:
- File parsing performance
- Metadata access speed
- Tensor iteration overhead

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and clippy
5. Submit a pull request

## Security

Please report security vulnerabilities to zackshen0526@gmail.com. See [SECURITY.md](SECURITY.md) for more information.

## GGUF Specification

This library implements the [GGUF specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md).

## License

MIT License - see [LICENSE](LICENSE) for details.

## Credits

- GGUF format by [ggml](https://github.com/ggerganov/ggml)
- Contributors: [@AvivAbachi](https://github.com/AvivAbachi), [@jbooth](https://github.com/jbooth), [@Knight-Ops](https://github.com/Knight-Ops)
