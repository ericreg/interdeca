# The Python package's Maturin backend builds Rust during an ordinary sync.
set windows-shell := ["cmd.exe", "/c"]

# Install development tools and the optimized production extension.
rust-setup:
    uv sync --locked --inexact

# Explicit rebuild after Rust edits; Python edits remain editable.
rust-build:
    uv sync --locked --inexact --reinstall-package interdeca

# Unoptimized build for debugger use, not performance measurements.
rust-dev:
    uv run --no-sync maturin develop --uv --locked --profile dev --manifest-path rust/Cargo.toml

# Check formatting and the application's existing non-Slicer tests.
rust-check: rust-build
    cargo fmt --manifest-path rust/Cargo.toml --check
    uv run --no-sync pytest tests --ignore=tests/end_to_end

# Build a wheel containing Python and the native extension for this platform.
rust-wheel:
    uv build --wheel
