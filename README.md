Example of neural network training with data splitting, written in Rust.

Demonstrates an ability to accelerate neural network training by splitting the training data.
In this case the same model instances are trained on own data parts in different threads,
and resulting application runs an ensemble of models.

Build:

```bash
cargo build --release --example single_training
cargo build --release --example parallel_training
```

Run:

```bash
./target/release/examples/single_training
./target/release/examples/parallel_training
```
