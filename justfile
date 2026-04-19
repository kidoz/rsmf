# RSMF Development Tasks

# Default task: list all commands
default:
    @just --list

# Format all code in the workspace
fmt:
    cargo fmt --all

# Lint all code in the workspace
lint:
    cargo clippy --workspace --all-targets -- -D warnings
    cargo clippy -p rsmf-wgpu --all-targets -- -D warnings
    cargo clippy -p rsmf-cuda --all-targets -- -D warnings
    cargo clippy -p rsmf-metal --all-targets -- -D warnings

# Run all tests in the workspace (excluding heavy GPU/optional crates by default)
test:
    cargo test --workspace

# Run all tests including GPU crates (requires local SDKs)
test-full:
    cargo test --workspace
    cargo test -p rsmf-wgpu
    # CUDA/Metal tests are hardware specific, run individually if needed.

# Check that everything compiles, including optional GPU crates
check:
    cargo check --workspace
    cargo check -p rsmf-wgpu
    cargo check -p rsmf-cuda
    cargo check -p rsmf-metal
    cargo check -p rsmf-python

# Build the project in release mode
build:
    cargo build --release --workspace

# Run benchmarks
bench:
    cargo bench -p rsmf-bench

# Run benchmarks with WGPU enabled
bench-gpu:
    cargo bench -p rsmf-bench --features wgpu

# Clean build artifacts
clean:
    cargo clean

# Quick local install of the rsmf CLI
install:
    cargo install --path crates/rsmf-cli

# Run the classification comparison example
example-compare:
    cargo run -p rsmf-core --example compare_classification --features compression

# Pack the large embedding model (stress test)
test-pack-large:
    cargo run -p rsmf-cli -- pack --from-npy tmp/data/embeddings_sentence-transformers_all-mpnet-base-v2.npy --quantize-int8 cpu_generic --compress-tensors --out embeddings_optimized.rsmf
    cargo run -p rsmf-cli -- inspect embeddings_optimized.rsmf
