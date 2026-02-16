# Contributing to gguf-rs

Thank you for your interest in contributing to `gguf-rs`! This document provides guidelines and instructions for contributing.

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code.

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check the existing issues as you might find that the problem has already been reported. When creating a bug report, please include:

- A clear and descriptive title
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Environment information (Rust version, OS)
- Any relevant error messages or logs

### Suggesting Enhancements

Enhancement suggestions are also welcome! When suggesting an enhancement:

- Use a clear and descriptive title
- Provide a detailed description of the proposed enhancement
- Explain why this enhancement would be useful
- If possible, provide examples or mockups

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Run tests (`cargo test`)
5. Run clippy (`cargo clippy -- -D warnings`)
6. Run fmt (`cargo fmt`)
7. Commit your changes (`git commit -m 'feat: add your feature'`)
8. Push to the branch (`git push origin feature/your-feature`)
9. Open a Pull Request

### Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `style:` for code style changes (formatting, etc.)
- `refactor:` for code refactoring
- `perf:` for performance improvements
- `test:` for adding or updating tests
- `chore:` for maintenance tasks
- `ci:` for CI/CD changes

Examples:
```
feat: add support for MXFP4 type
fix: correct tensor offset calculation
docs: update API documentation
```

### Coding Style

- Follow Rust conventions and idioms
- Use `cargo fmt` for formatting
- Run `cargo clippy -- -D warnings` before committing
- Write tests for new features
- Update documentation as needed

## Development Setup

1. Clone the repository
2. Install Rust toolchain (stable)
3. Run tests: `cargo test`
4. Run CLI tool: `cargo run --bin gguf -- <path-to-file>`

## Testing

Run the test suite:
```bash
cargo test
```

Run with verbose output:
```bash
cargo test --verbose
```

Run a specific test:
```bash
cargo test test_name
```

## Questions?

Feel free to open an issue for any questions or discussion!
