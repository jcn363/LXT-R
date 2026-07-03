# Rust Best Practices

Apply these guidelines when writing or reviewing Rust code. Based on Apollo GraphQL's [Rust Best Practices Handbook](https://github.com/apollographql/rust-best-practices).

## Best Practices Reference

Before reviewing, familiarize yourself with Apollo's Rust best practices. Read ALL relevant chapters in the same turn in parallel. Reference these files when providing feedback:

- [Chapter 1 - Coding Styles and Idioms](https://github.com/apollographql/skills/blob/HEAD/skills/rust-best-practices/references/chapter_01.md): Borrowing vs cloning, Copy trait, Option/Result handling, iterators, comments
- [Chapter 2 - Clippy and Linting](https://github.com/apollographql/skills/blob/HEAD/skills/rust-best-practices/references/chapter_02.md): Clippy configuration, important lints, workspace lint setup
- [Chapter 3 - Performance Mindset](https://github.com/apollographql/skills/blob/HEAD/skills/rust-best-practices/references/chapter_03.md): Profiling, avoiding redundant clones, stack vs heap, zero-cost abstractions
- [Chapter 4 - Error Handling](https://github.com/apollographql/skills/blob/HEAD/skills/rust-best-practices/references/chapter_04.md): Result vs panic, thiserror vs anyhow, error hierarchies
- [Chapter 5 - Automated Testing](https://github.com/apollographql/skills/blob/HEAD/skills/rust-best-practices/references/chapter_05.md): Test naming, one assertion per test, snapshot testing
- [Chapter 6 - Generics and Dispatch](https://github.com/apollographql/skills/blob/HEAD/skills/rust-best-practices/references/chapter_06.md): Static vs dynamic dispatch, trait objects
- [Chapter 7 - Type State Pattern](https://github.com/apollographql/skills/blob/HEAD/skills/rust-best-practices/references/chapter_07.md): Compile-time state safety, when to use it
- [Chapter 8 - Comments vs Documentation](https://github.com/apollographql/skills/blob/HEAD/skills/rust-best-practices/references/chapter_08.md): When to comment, doc comments, rustdoc
- [Chapter 9 - Understanding Pointers](https://github.com/apollographql/skills/blob/HEAD/skills/rust-best-practices/references/chapter_09.md): Thread safety, Send/Sync, pointer types

## Quick Reference

### Borrowing & Ownership

- Prefer `&T` over `.clone()` unless ownership transfer is required
- Use `&str` over `String`, `&[T]` over `Vec<T>` in function parameters
- Small `Copy` types (≤24 bytes) can be passed by value
- Use `Cow<'_, T>` when ownership is ambiguous

> **llama-rs notes:** The project uses `Arc<Model>` for shared ownership across threads and `RwLock` for read-mostly state (KV cache, tensor data cache). See `CODE_STYLE.md#Thread-Safety` for the concurrency model. The matmul backend passes `&[f32]` slices per the `&[T]` over `Vec<T>` rule — see `crates/ggml-cpu/src/matmul.rs:16-24`.

### Error Handling

- Return `Result<T, E>` for fallible operations; avoid `panic!` in production
- Never use `unwrap()`/`expect()` outside tests
- Use `thiserror` for library errors, `anyhow` for binaries only
- Prefer `?` operator over match chains for error propagation

> **llama-rs notes:** Follows this exactly. Library error enum at `crates/gguf/src/errors.rs` uses `#[derive(Error)]` with `GgufResult<T>` type alias. The project makes one deliberate exception: `expect("lock poisoned")` on `RwLock`/`Mutex` access is acceptable in library code since lock poisoning is unrecoverable — see `CODE_STYLE.md#Lock-Poisoning`.

### Performance

- Always benchmark with `--release` flag
- Run `cargo clippy -- -D clippy::perf` for performance hints
- Avoid cloning in loops; use `.iter()` instead of `.into_iter()` for Copy types
- Prefer iterators over manual loops; avoid intermediate `.collect()` calls

> **llama-rs notes:** Uses criterion with `black_box` — see `crates/ggml-cpu/benches/cpu_bench.rs`. The SIMD matmul uses a tiered fallback: AVX (8-wide) → SSE4.2 (4-wide) → scalar, with 4 accumulators per iteration for instruction-level parallelism (`crates/ggml-cpu/src/simd.rs:15`). Parallelism thresholds documented in `CODE_STYLE.md#Parallelism-Thresholds`: 64 rows minimum for parallel matmul, 1024 elements for vector ops.

### Linting

Run regularly: `cargo clippy --workspace -- -D warnings`

Key lints to watch:

- `redundant_clone` - unnecessary cloning
- `large_enum_variant` - oversized variants (consider boxing)
- `needless_collect` - premature collection

Use `#[expect(clippy::lint)]` over `#[allow(...)]` with justification comment.

> **llama-rs notes:** The workspace uses: `cargo clippy --workspace -- -D warnings`. There is no workspace-level lint config — only `ggml-cuda` enables `#![deny(clippy::pedantic)]` at the crate level. See `CODE_STYLE.md#Clippy-Linting`. Uses `#[expect(dead_code)]` over `#[allow(dead_code)]` per the modern convention.

### Testing

- Name tests descriptively: `process_should_return_error_when_input_empty()`
- One assertion per test when possible
- Use doc tests (`///`) for public API examples
- Consider `cargo insta` for snapshot testing generated output

> **llama-rs notes:** Uses `describe_should_expected_behavior` naming (e.g., `dot_f32_should_compute_correct_result`). Unit tests are inline under `#[cfg(test)] mod tests { ... }`, integration tests live in `crates/<name>/tests/<name>_test.rs`, benchmarks in `crates/<name>/benches/<name>.rs`. Tests requiring external model files skip gracefully without panicking — see `CODE_STYLE.md#Test-Patterns` and example at `crates/ggml-cpu/src/lib.rs`. Benchmark harness is criterion — see `crates/ggml-cpu/benches/cpu_bench.rs`.

### Generics & Dispatch

- Prefer generics (static dispatch) for performance-critical code
- Use `dyn Trait` only when heterogeneous collections are needed
- Box at API boundaries, not internally

> **llama-rs notes:** The project uses concrete types (`Tensor`, `DType`, `Graph` in `ggml/`) rather than generic abstractions. The two backends (`CpuBackend`, `CudaBackend`) are separate structs, not a shared trait — kept intentionally simple. The SIMD dot product uses `#[inline]` and const generics for step sizes (`AVX_F32_STEP`, `SSE_F32_STEP`) — see `crates/ggml-cpu/src/simd.rs:17-23`.

### Type State Pattern

Encode valid states in the type system to catch invalid operations at compile time:

```rust
struct Connection<State> { /* ... */ _state: PhantomData<State> }
struct Disconnected;
struct Connected;

impl Connection<Connected> {
    fn send(&self, data: &[u8]) { /* only connected can send */ }
}
```

### Documentation

- `//` comments explain *why* (safety, workarounds, design rationale)
- `///` doc comments explain *what* and *how* for public APIs
- Every `TODO` needs a linked issue: `// TODO(#42): ...`
- Enable `#![deny(missing_docs)]` for libraries

> **llama-rs notes:** Every `unsafe` block must have a `// SAFETY:` comment. The project uses ASCII section separators (`// ─── Section ────────────`) to visually organize files — see `CODE_STYLE.md#Section-Separators`. The `#[must_use]` attribute is used on pure accessors (`is_available()`, `total_vram()`). Crate-level docs (`//!`) are present in all `lib.rs` files.

---

## Appendix: Market Rust Best Practices
*Ecosystem-wide best practices sourced from the Rust API Guidelines, The Rust Book, Cargo Book, Tokio docs, Microsoft Rust Guidelines, Effective Rust, and community conventions.*

### A.1 Rust API Guidelines (Official)

The Rust Library Team maintains a comprehensive set of API design guidelines. Key checkpoints beyond Apollo's coverage:

**Naming**
- Conversion methods follow `as_` (free, borrowed→borrowed), `to_` (expensive, borrowed→owned), `into_` (variable, owned→owned) prefixes ([C-CONV])
- Getters omit the `get_` prefix (e.g., `len()` not `get_len()`) except when there's a single obvious thing being gotten ([C-GETTER])
- Collection iterators use `iter` / `iter_mut` / `into_iter` ([C-ITER])
- Iterator types are named after the method that produces them (e.g., `IntoIter` for `into_iter()`) ([C-ITER-TY])
- Feature names avoid placeholder words: `serde` not `use-serde`, `std` not `use-std` ([C-FEATURE])
- Error types use consistent verb-object-error word order: `ParseAddrError` not `AddrParseError` ([C-WORD-ORDER])

**Interoperability**
- Implement common traits eagerly: `Clone`, `Debug`, `PartialEq`, `Eq`, `PartialOrd`, `Ord`, `Hash`, `Default` ([C-COMMON-TRAITS])
- Use `From` / `TryFrom` / `AsRef` / `AsMut` for conversions; avoid `Into` (has blanket impl) ([C-CONV-TRAITS])
- Collections implement `FromIterator` and `Extend` ([C-COLLECT])
- Data structures implement Serde's `Serialize`/`Deserialize` behind a `serde` feature gate ([C-SERDE])
- Types should be `Send` + `Sync` where possible; test with compile-fail assertions ([C-SEND-SYNC])
- Error types must implement `std::error::Error`, be `Send + Sync + 'static`; never use `()` as an error type ([C-GOOD-ERR])
- Generic reader/writer functions take `R: Read` and `W: Write` by value ([C-RW-VALUE])

**Predictability & Type Safety**
- Conversions live on the most specific type involved ([C-CONV-SPECIFIC])
- Functions with a clear receiver are methods, not free functions ([C-METHOD])
- Never use out-parameters; return values instead ([C-NO-OUT])
- Only smart pointers implement `Deref`/`DerefMut` ([C-DEREF])
- Constructors are static, inherent methods (e.g., `Foo::new()`) ([C-CTOR])
- Functions expose intermediate results to avoid duplicate work ([C-INTERMEDIATE])
- Caller decides where to copy and place data ([C-CALLER-CONTROL])
- Use generics to minimize assumptions about parameters ([C-GENERIC])
- Traits are object-safe if they may be useful as trait objects ([C-OBJECT])
- Use newtypes for static distinctions between interpretations of the same underlying type ([C-NEWTYPE])
- Use custom enums/structs, not bare `bool` or `Option`, to convey argument meaning ([C-CUSTOM-TYPE])
- Use `bitflags` crate for sets of flags, not enums ([C-BITFLAG])
- Use the builder pattern for complex construction, especially when side effects are involved ([C-BUILDER])
- Make traits sealed (`#[doc(hidden)]` + private supertrait) to prevent downstream implementations ([C-SEALED])

**Dependability**
- Validate arguments; prefer static enforcement (type system), then `debug_assert!`, then `_unchecked` variants ([C-VALIDATE])
- Destructors must never fail; provide a separate `close()` method that returns `Result` ([C-DTOR-FAIL])
- Destructors should not block; provide explicit shutdown methods ([C-DTOR-BLOCK])

> **llama-rs notes:** The project uses `assert_eq!` with explicit shape error messages for argument validation (e.g., `matmul.rs:25-27`). Rust API Guidelines naming: conversion methods use `as_`/`to_`/`into_` prefixes. Feature gate uses the simple syntax: `cuda = ["cudarc"]` (not `dep:` prefix). The `GgufError` type implements `std::error::Error` via `thiserror` and is `Send + Sync + 'static`. See `ARCHITECTURE.md#Core-Components` for the full crate breakdown.

### A.2 Cargo & Workspace Management

**Workspace Organization**
- Use `[workspace]` in a virtual manifest for monorepos — all members share a single `target/` and `Cargo.lock`, cutting CI build times significantly.
- Declare shared dependencies in `[workspace.dependencies]` for version consistency across all workspace members.
- Use `[workspace.package]` to share metadata (version, authors, license, edition) across members.
- Use `[workspace.lints]` to unify lint configuration.
- Use `resolver = "2"` (edition 2021+) or `resolver = "3"` (edition 2024, MSRV-aware) for correct feature unification.
- Run `cargo check --workspace`, `cargo test --workspace`, `cargo clippy --workspace` for complete coverage.
- Split large crates into workspace members to improve incremental compilation and build parallelism.
- For monorepos, organize crates by domain (e.g., `crates/network/`, `crates/storage/`).

**Features (Compile-Time Configuration)**
- Features must be **additive**: enabling feature A + feature B must always work. Never use mutually exclusive features.
- Name features after **what they enable**, not what they depend on: `postgres` not `sqlx`.
- Make commonly-needed features default; gate niche functionality behind features.
- Use `dep:` syntax (Rust 1.60+) to prevent optional deps from creating implicit features.
- Test with `--no-default-features` in CI to catch breakage.
- Document available features at the crate level (`//!` in `lib.rs`) or in README.

> **llama-rs notes:** Workspace uses `resolver = "2"` (not `"3"`) despite edition 2024 — this is a deliberate choice for compatibility. Features follow the naming rule: `cuda` feature enables CUDA (`dep:cudarc` syntax). The `cuda` feature is now enabled by default (`default = ["cuda"]` in `ggml-cuda/Cargo.toml`). Workspace uses `[workspace.dependencies]` and `[workspace.package]` for shared configuration — see `Cargo.toml`. Note: no `[workspace.lints]` section; lint config is per-crate.

**Profiles & Build Optimization**
- Use `lto = "thin"` for most release builds; `lto = "fat"` for final distribution builds.
- Define custom profiles for specific needs: `profiling` (debug symbols + release), `dist` (max optimization).
- Use `debug = 1` in release for line-number-only backtraces without full perf loss.
- Configure `.cargo/config.toml` for target-specific settings, network timeouts, and build parallelization.

> **llama-rs notes:** Release profile: `lto = true`, `panic = "abort"` (no explicit `opt-level`, `codegen-units`, or `strip`). See `Cargo.toml`. The `.cargo/config.toml` sets `target-cpu=bdver1` for AMD Opteron 3280 — see `CODE_STYLE.md`. No custom profiling profile defined (uses default release).

> **llama-rs notes:** Crates are organized under `crates/` by domain following the recommended layout. Uses modern `submodule.rs` style (no `mod.rs`). Struct definitions placed in `lib.rs`, `impl` blocks in named files (`model.rs`, `reader.rs`) — see `CODE_STYLE.md#File-Organization`. Visibility follows narrowest-modifier rule: `pub(crate)` for internal utilities, `pub` only for intentional public API. Re-exports via `pub use` in `lib.rs` decouple file structure from API surface.

### A.3 Module Organization & Visibility

**File Layout**
- One module per file; directories for submodules (`src/front_of_house/hosting.rs`).
- Prefer the modern `submodule.rs` style over `submodule/mod.rs` (avoids many files named `mod.rs`).
- Use `mod` declarations in the crate root to establish the module tree; the compiler resolves paths.

```
src/
├── lib.rs              # crate root, mod declarations + pub use re-exports
├── front_of_house/
│   ├── mod.rs          # declares submodules, re-exports public API
│   └── hosting.rs
└── back_of_house/
    ├── mod.rs
    └── kitchen.rs
```

**Visibility Best Practices**
- Default to private; widen visibility incrementally using the narrowest modifier that works.
- `pub(crate)` — internal to the crate, hidden from external consumers (ideal for shared internal utilities).
- `pub(super)` — visible only to the parent module (for helpers used by the immediate parent).
- `pub(in path)` — fine-grained scoping to a specific ancestor module.
- Use `pub use` (re-exporting) to create clean public APIs that decouple the physical file structure from the logical namespace. Consumers write `use crate::Vector;` not `use crate::math::vector::Vector;`.
- Avoid namespace pollution — re-export only what forms the intentional public API.
- Never glob-import (`use foo::*`) from external crates — only from internal modules where intentional.

### A.4 Async & Concurrency (Tokio Ecosystem)

> **llama-rs notes:** The `llama-server` binary uses axum + tokio for the HTTP API (`GET /health`, `POST /completion` with SSE streaming). Server code in `crates/llama-server/src/main.rs`. For CPU parallelism, the project uses `std::thread::scope` (not `tokio::task`) in the matmul backend for raw pointer access — see `crates/ggml-cpu/src/matmul.rs:41-50`. Data-parallel operations (dequantization, tensor loading) use `rayon`. The inference engine (`crates/llama/src/`) is synchronous — async only enters at the HTTP server layer. See `ARCHITECTURE.md#5.-llama---Inference-Engine`.

**Runtime Choice**
- Tokio is the de-facto standard async runtime (2025+); async-std is in maintenance mode.
- Use `#[tokio::main]` for application entry points; configure runtime with `Builder` for fine-grained control (worker threads, blocking threads, event interval).

**Task Management**
- **Never block the async runtime**: CPU-bound work → `tokio::task::spawn_blocking`; file I/O → use Tokio's async I/O or `spawn_blocking`.
- **Avoid blocking the executor**: no `std::thread::sleep` (use `tokio::time::sleep`), no `std::sync::Mutex` held across `.await`.
- Use `JoinSet` for managing groups of related tasks (structured concurrency) — tracks handles, collects results, catches panics.
- Use `CancellationToken` for graceful shutdown — propagate through the task tree.
- Use `Semaphore` or `buffer_unordered(N)` to limit concurrency and provide backpressure.
- For CPU-bound data-parallel work, combine `spawn_blocking` with `rayon`.

**Synchronization**
- `std::sync::Mutex` — use for short, synchronous critical sections (faster). Never hold across `.await`.
- `tokio::sync::Mutex` — use when you must hold the lock across an `.await` point (higher overhead due to waker machinery).
- Prefer the **actor pattern** (task + channel) over shared `Arc<Mutex<T>>` for complex state — uses `mpsc::channel` for natural backpressure and eliminates lock contention.
- Use **bounded channels** (`mpsc::channel(N)`) rather than unbounded (`mpsc::unbounded_channel()`) to prevent OOM under load.

**Common Pitfalls (Anti-Patterns)**

| Pitfall | Fix |
|---------|-----|
| Blocking the runtime (long sync op in async fn) | `spawn_blocking` or `yield_now` |
| Holding `std::sync::Mutex` across `.await` | Use `tokio::sync::Mutex` or shorten the lock scope |
| Unbounded spawn without limits | `Semaphore` or `JoinSet` with capacity |
| Sequential awaits that should be concurrent | `tokio::join!` or `JoinSet` |
| `select!` with complex multi-step branches | Move complex logic into a spawned task |
| Using `Unpin` incorrectly | Understand `Pin` when working with self-referential futures |

**Production Async**
- Use `tracing` for structured, async-aware observability (spans cross task boundaries).
- Enable `tokio_unstable` and use `tokio-console` to visualize task states and detect blocking.
- Use `spawn_blocking` for FFI calls and file I/O; never `unwrap()` on channel sends (handle `Err` for closed receiver).

### A.5 Unsafe Code & FFI Guidelines

**When to Use Unsafe**
- Valid reasons only: (1) novel abstractions (new smart pointer/allocator), (2) performance (proven via benchmarks), (3) FFI and platform calls.
- **Never** use ad-hoc `unsafe` to shorten safe code, bypass `Send`/`Sync`, or bypass lifetimes.
- All `unsafe` code must be hardened against adversarial closures, misbehaving `Deref`/`Clone`/`Drop` impls.
- Run Miri on all unsafe code paths; use `cargo fuzz` with sanitizers for FFI targets.

> **llama-rs notes:** The project uses `unsafe` primarily for SIMD intrinsics (AVX/SSE4.2) in `crates/ggml-cpu/src/simd.rs` and for raw pointer access in the matmul threadpool. Every `unsafe` block has a `// SAFETY:` comment. Unsafe is encapsulated behind safe public APIs (`dot_f32`, `matmul_f32`). No `unsafe fn` declarations — all unsafe blocks are local. See `CODE_STYLE.md#Do's-and-Don'ts`.

**The Three Rules of Sound Unsafe**
1. **Document invariants** — every `// SAFETY:` comment explains why the operation is valid.
2. **Encapsulate** — unsafe lives inside a safe API; callers cannot trigger UB through safe code.
3. **Minimize** — only the smallest possible block is `unsafe`; avoid `unsafe fn` when a safe wrapper suffices.

**FFI Patterns**
- Use `unsafe extern` blocks (required since Rust 2024). Mark items explicitly `safe` or `unsafe`.
- Use `#[repr(C)]` for types crossing FFI boundaries; never use `#[repr(Rust)]` types for FFI.
- Use sized integer types (`i32`, `u32`, etc.) — never `int`, `long`, etc.
- Use `CString`/`CStr` for string interop; use `Box::from_raw` / `Box::into_raw` for heap-allocated memory.
- Allocate and free on the same side of the FFI boundary. Provide symmetric init/destroy functions.
- Encapsulate unsafe FFI code in safe Rust wrappers. Box at the API boundary, not internally.
- Prevent panics from crossing FFI boundaries (`catch_unwind` before the boundary).
- Never use empty enums as FFI types (compiler treats them as uninhabited → UB).

> **llama-rs notes:** Test naming follows `describe_should_expected_behavior`. Unit tests are inline in `#[cfg(test)] mod tests` blocks. Integration tests in `crates/<name>/tests/<name>_test.rs`. Benchmarks use criterion in `crates/<name>/benches/<name>.rs` — see `crates/ggml-cpu/benches/cpu_bench.rs` and `crates/llama/benches/profiling.rs`. Tests requiring model files skip gracefully (check `model_path.exists()`). See `CODE_STYLE.md#Testing` for full patterns.

### A.6 Testing Ecosystem

**Multi-Layer Strategy**
- **Unit tests**: cover business logic, error paths, edge cases (`#[cfg(test)] mod tests` in-source). Run on every commit.
- **Integration tests**: in `tests/` directory, test public API across module boundaries. Each file is a separate crate (consolidate to avoid compile overhead).
- **Doc tests**: `///` code blocks test public API examples. Use `?` not `unwrap()`.
- **Property-based tests** (`proptest` / `quickcheck`): test invariants with random inputs. One property can replace dozens of example tests.
- **Fuzzing** (`cargo-fuzz` / `arbtest`): coverage-guided, catches crashers humans miss. Run locally for long sessions, in CI with `-max_total_time`.
- **Snapshot tests** (`insta`): for complex output validation (serialization, rendering, codegen).
- **Mutation tests** (`cargo-mutants`): verify tests catch behavior changes. Run periodically on core modules.

**Testing Patterns**
```rust
// Property-based test with proptest
use proptest::prelude::*;

proptest! {
    #[test]
    fn sort_should_preserve_length(input in prop::collection::vec(any::<i32>(), 0..100)) {
        let sorted = sort(input.clone());
        prop_assert_eq!(sorted.len(), input.len());
        // Property: output contains same elements
        prop_assert!(input.iter().all(|x| sorted.contains(x)));
    }
}
```

**Benchmarking**
- Use **Criterion** (`criterion` crate) for statistically rigorous benchmarks with regression detection.
- Always benchmark with `--release` flag; use `black_box` to prevent dead-code elimination.
- Track benchmarks across commits; compare against baselines.
- Profile before optimizing: measure in release mode, use `perf` / `flamegraph` for CPU, `DHAT` for heap.

**Best Practices**
- Name tests descriptively: `fn describe_should_expected_behavior()`.
- Prefer one assertion per test for clear failure messages.
- Use `#[ignore]` for slow tests; run them separately.
- Use `rstest` for test fixtures and parameterized tests.
- Test error paths and panic conditions (`#[should_panic]` + `#[should_panic(expected = "...")]`).
- Use `#[cfg(test)]` to exclude test-only helpers from production builds.

> **llama-rs notes:** All external dependencies are declared in `[workspace.dependencies]` for version consistency. Uses `cargo deny` for license compliance — policy at `deny.toml` allows MIT, Apache-2.0, and Unlicense only. Internal crates use `path = "crates/<name>"` references. Version specs use caret (default, e.g., `"1.5"`). The `Cargo.lock` is committed (application workspace). See `Cargo.toml` for the full dependency list.

### A.7 Dependency Management & SemVer

**Version Specification**
- Prefer `"1.2.3"` (caret, the default) — allows SemVer-compatible updates, specifies a minimum.
- Avoid exact pinning (`"=1.2.3"`) — blocks security fixes and creates resolution conflicts.
- Avoid overly narrow requirements (`~1.2` blocks minor updates that should be compatible).
- Use `[workspace.dependencies]` for shared version declarations.
- For libraries, specify the actual minimum version required; verify with `cargo +nightly -Zminimal-versions`.

**SemVer Verification**
- Use `cargo-semver-checks` in CI to automatically detect breaking changes before publishing.
- Be aware of subtle breaking changes: adding an enum variant, changing auto-trait impls, adding public items (breaks glob imports).
- Use `cargo deny` for license compliance, duplicate detection, and advisory checking.
- Use `cargo udeps` to find unused dependencies.
- Use `cargo tree` to visualize the dependency graph and detect duplicates.

**Best Practices**
- Keep `Cargo.lock` committed for applications (reproducible builds); libraries don't ship it.
- Use `cargo update` deliberately; use `--package` to update specific dependencies.
- Set `package.rust-version` in `Cargo.toml` for MSRV; use `resolver = "3"` (edition 2024) for Rust-version-aware resolution.
- Be conservative with pre-release dependencies in published libraries.
- Follow **minimum necessary versions**: prefer well-known crates for complex functionality, but don't depend on a crate for a single function you could write yourself.

> **llama-rs notes:** Crate-level `//!` docs present in all `lib.rs` files. All `Cargo.toml` files include `description`, `version`, `edition`, `rust-version`, `license`, `repository`. The workspace uses `[workspace.package]` for shared metadata. Changelog not yet maintained. All crates are unpublished (path-only workspace). See `CODE_STYLE.md#Do's-and-Don'ts` for doc conventions.

### A.8 Documentation & Metadata

**Crate-Level Documentation**
- Every published crate must have thorough crate-level docs (`//!` in `lib.rs`) including a usage example.
- `Cargo.toml` must include: `description`, `license`, `repository`, `readme`, `keywords`, `categories`.
- `homepage` should be a dedicated website (not redundant with `repository` or `documentation`).
- Publish release notes (CHANGELOG.md) documenting all significant changes, with breaking changes clearly marked.
- Use annotated Git tags for releases.

**Rustdoc Conventions**
- Every public item needs a doc comment.
- Examples use `?` (not `try!`, not `unwrap()`) so users can copy them verbatim.
- Document `Errors`, `Panics`, and `Safety` sections in function docs where applicable.
- Link to related items in prose ([C-LINK] — per RFC 1574, "Link all the things").
- Use `#[doc(hidden)]` to hide implementation details that would clutter rustdoc (e.g., `From<PrivateError>` impls).
- Avoid over-documenting internal impls that users don't need to see.

---

*Sources: [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/), [The Rust Book](https://doc.rust-lang.org/book/), [The Cargo Book](https://doc.rust-lang.org/cargo/), [Microsoft Pragmatic Rust Guidelines](https://microsoft.github.io/rust-guidelines/), [Effective Rust](https://effective-rust.com/), [Tokio Docs](https://tokio.rs/), [Rust Patterns Book](https://www.rust-patterns.com/), [Async Rust From Futures to Production (Microsoft)](https://microsoft.github.io/RustTraining/async-book/)*
