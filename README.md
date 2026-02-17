# GGUF-RS

[![Crates.io](https://img.shields.io/crates/v/gguf-rs.svg)](https://crates.io/crates/gguf-rs)
[![Documentation](https://docs.rs/gguf-rs/badge.svg)](https://docs.rs/gguf-rs)
[![License](https://img.shields.io/crates/l/gguf-rs.svg)](https://github.com/zackshen/gguf/blob/main/LICENSE)
![Unit test](https://github.com/zackshen/gguf/actions/workflows/test.yml/badge.svg)
![Security Audit](https://github.com/zackshen/gguf/actions/workflows/audit.yml/badge.svg)
![Publish](https://github.com/zackshen/gguf/actions/workflows/publish.yml/badge.svg)

A Rust library for parsing and reading GGUF (GGML Universal Format) files. GGUF files are binary files that contain key-value metadata and tensors, commonly used for storing quantized machine learning models.

## Features

- ✅ Decode GGUF files (v1, v2, v3)
- ✅ Access key-value metadata
- ✅ Access tensor information
- ✅ Support for little-endian and big-endian files
- ✅ CLI tool for quick inspection
- ✅ Zero-copy metadata access

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

    // Access metadata
    for (key, value) in model.kv() {
        println!("{}: {:?}", key, value);
    }

    // List tensors
    for tensor in model.tensors() {
        println!("Tensor: {} (shape: {:?})", tensor.name, tensor.shape);
    }

    Ok(())
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

## Testing

```bash
cargo test
```

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
