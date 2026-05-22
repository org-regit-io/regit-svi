# regit-svi — Task runner
# Run `just` to see available recipes.

# Quality gate — run all checks
check: fmt-check lint test doc
    @echo "All checks passed."

# Format check
fmt-check:
    cargo fmt --all --check

# Format (fix)
fmt:
    cargo fmt --all

# Lint — zero warnings, all targets
lint:
    cargo clippy --all-targets -- -D warnings

# Run all tests
test:
    cargo test

# Build documentation
doc:
    cargo doc --no-deps

# Run the library quickstart example
example:
    cargo run --example quickstart

# Run benchmarks
bench:
    cargo bench

# Dependency / licence audit (requires cargo-deny)
deny:
    cargo deny check

# WASM build smoke test — the crate is pure math and WASM-clean
wasm:
    cargo build --target wasm32-unknown-unknown --release

# Run property tests with extra cases
proptest:
    PROPTEST_CASES=5000 cargo test prop_

# Full CI pipeline
ci: fmt-check lint test doc deny wasm

# Run Miri for undefined-behaviour checks
miri:
    cargo +nightly miri test --lib
