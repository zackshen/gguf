# Security Policy

## Supported Versions

Currently, all versions of `gguf-rs` are supported for security updates.

## Reporting a Vulnerability

If you discover a security vulnerability, please report it privately to us before disclosing it publicly.

### How to Report

- Send an email to: zackshen0526@gmail.com
- Use a descriptive subject line starting with `[SECURITY]`

### What to Include

Please include as much information as possible:

- Type of vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if known)

### Response

We will:

- Acknowledge receipt within 48 hours
- Provide a detailed response within 7 days
- Confirm the vulnerability and assess impact
- Work on a fix and coordinate disclosure
- Credit you in the security advisory

### Disclosure Policy

We aim to fix vulnerabilities within 14 days of confirmation. Once fixed, we will:

- Release a new version with the fix
- Publish a security advisory
- Credit the reporter (if desired)

## Security Best Practices

This library parses GGUF files, which are binary files. Always:

- Validate input files before parsing
- Be cautious with files from untrusted sources
- Consider resource limits when processing large files
- Use in a sandboxed environment if processing user-provided files

## Dependencies

We strive to keep dependencies up-to-date and secure. If you discover a security issue in a dependency, please report it following the same process.
