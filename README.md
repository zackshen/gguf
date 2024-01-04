# GGUF

This is a Rust project that provides functionality for decoding and working with GGUF files. GGUF files are binary files that contain key-value metadata and tensors.

## Features

- Decoding GGUF files
- Accessing key-value metadata and tensors from GGUF files
- Support for different versions of GGUF files: v1, v2, v3

## Usage

```rust
use gguf::get_gguf_container;
use std::fs::File;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut container = get_gguf_container("path_to_your_file")?;
    let model = container.decode()?;
    println!("Model Family: {}", model.model_family());
    println!("Number of Parameters: {}", model.model_parameters());
    println!("File Type: {}", model.file_type());
    println!("Number of Tensors: {}", model.num_tensor());
    Ok(())
}
```

## Testing

This project includes unit tests. Run them with `cargo test`.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## GGUF Specification

[GGUF Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)

## License

[MIT](https://choosealicense.com/licenses/mit/)
