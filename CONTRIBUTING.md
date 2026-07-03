# Contributing to LXT-R

## Development Setup

1. Install Rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
2. Install CUDA toolkit (for FP8/loader FFI)
3. Clone and build: `git clone https://github.com/jcn363/LXT-R.git && cd LXT-R && cargo build`

## Code Style

- All shared primitives live in their own crate under `crates/`
- Each shared crate has a `factory.rs` — use it to instantiate modules
- Import from `ltx_*` crate root, never internal submodules
- No hardcoded constants — use `ltx_types::constants::*`
- No reimplemented primitives — use the shared crate version

## SSOT Rules

Before merging, verify:

- [ ] No `1e-6`, `1e-8`, `448.0`, `10000.0` hardcoded — use `ltx_types::constants::*`
- [ ] No duplicate functions — use shared primitive crates
- [ ] No duplicate types — use shared primitive crates
- [ ] All imports use `ltx_*` paths
- [ ] `cargo clippy --all-targets` passes

## Testing

Each shared primitive should have tests in `crates/<name>/tests/` with golden `.npz` files from Python for numerical comparison.

```bash
cargo test --workspace
```

## Commit Messages

Use conventional commits:
- `feat: <description>` for new features
- `fix: <description>` for bug fixes
- `refactor: <description>` for code cleanup
- `docs: <description>` for documentation
