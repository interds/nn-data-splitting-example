#[path = "../src/multi_nn.rs"]
mod multi_nn;
#[path = "../src/nn.rs"]
mod nn;
#[path = "../src/utils.rs"]
mod utils;

use indicatif::HumanDuration;
use std::time::Instant;

const NUMBER_OF_SPLITS: usize = 10;

fn main() {
    let mut examples: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();

    utils::load_examples("./data/fen.txt", &mut examples);

    let start = Instant::now();
    multi_nn::train_multiple(&examples, NUMBER_OF_SPLITS);
    println!("Training took {}", HumanDuration(start.elapsed()));

    let example_document = "[1-0] 2bQ4/p4kb1/6n1/q1p1p3/1rn1P3/N3BP2/1PP5/2KR2R1 w".to_string();
    let result = multi_nn::run(example_document, NUMBER_OF_SPLITS);
    if result == "a3c4" {
        println!("OK")
    } else {
        println!("Failed: {} != a3c4", result)
    }
}
