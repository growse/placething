default:
    just --list

# Build debug binary
build:
    cargo build

# Build optimised release binary
build-release:
    cargo build --release

# Run the server (release) on default port 8000
run: build-release
    ./target/release/server

# Run the server (release) on a custom port
run-port port: build-release
    ./target/release/server --port {{port}}

# Run the server (debug)
run-dev:
    cargo run -- --port 8000

# Transform a text file through the running server
transform file:
    curl -sf -X POST http://localhost:8000/transform \
        -H "Content-Type: text/plain" \
        --data-binary @{{file}}

# Run Rust tests
test:
    cargo test

# Check code compiles without producing a binary
check:
    cargo check

# Lint with clippy
lint:
    cargo clippy -- -D warnings

# Remove build artefacts
clean:
    cargo clean
