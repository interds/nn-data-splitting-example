#[path = "../src/multi_nn.rs"]
mod multi_nn;
#[path = "../src/nn.rs"]
mod nn;
#[path = "../src/utils.rs"]
mod utils;

use crate::nn::{HaltCondition, NN};
use indicatif::{HumanDuration, ProgressBar, ProgressStyle};
use std::fs::File;
use std::io::{Read, Write};
use std::time::Instant;

const FILENAME: &str = "./data/net-single.json";
const EPOCHS: u32 = 100;

fn main() {
    let mut examples: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();

    utils::load_examples("./data/fen.txt", &mut examples);

    let start = Instant::now();
    train(FILENAME, &examples);
    println!("Training took {}.", HumanDuration(start.elapsed()));

    let example_document = "[1-0] 2bQ4/p4kb1/6n1/q1p1p3/1rn1P3/N3BP2/1PP5/2KR2R1 w".to_string();
    let mut f = File::open(FILENAME).unwrap();
    let mut buffer = String::new();
    f.read_to_string(&mut buffer).unwrap();
    let net: NN = NN::from_json(&buffer);

    let results = net.run(&utils::str2vec(example_document, 80));
    let result = utils::vec2str(&results);

    if result == "a3c4" {
        println!("OK")
    } else {
        println!("Failed: {} != a3c4", result)
    }
}

fn train(filename: &str, examples: &Vec<(Vec<f64>, Vec<f64>)>) {
    let mut net: NN = if std::fs::metadata(filename).is_err() {
        NN::new(&[80, 120, 120, 4])
    } else {
        let mut f = File::open(filename).unwrap();
        let mut buffer = String::new();
        f.read_to_string(&mut buffer).unwrap();
        NN::from_json(&buffer)
    };

    let example_count = examples.len();
    let bar = ProgressBar::new((example_count * EPOCHS as usize) as u64);
    bar.set_style(ProgressStyle::default_bar()
        .template("{prefix} [{elapsed_precise}] {bar:40} {percent:>3}% - {pos}/{len} - {per_sec} - eta {eta} - {msg}"));

    net.train(&examples)
        .halt_condition(HaltCondition::Epochs(EPOCHS))
        .momentum(0.9)
        .rate(0.07)
        .go(Box::new(
            move |s, epochs, i, _, training_error_rate| match s {
                1 => {
                    bar.set_prefix(&format!("epoch {:>3}", epochs + 1));
                    bar.tick();
                }
                2 => {
                    if i % 1_000 == 0 {
                        bar.set_position((example_count * epochs as usize + i) as u64);
                    }
                }
                3 => {
                    bar.set_message(&format!("error rate={}", training_error_rate));
                    bar.set_position((example_count * epochs as usize + i) as u64);
                }
                4 => bar.finish(),
                _ => (),
            },
        ));

    let mut f = File::create(filename).unwrap();
    f.write(net.to_json().as_bytes()).unwrap();
}
