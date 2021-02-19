//! Utility functions.

use indicatif::{ProgressBar, ProgressStyle};
use std::fs::File;
use std::io::{BufRead, BufReader};

/// Convert Vec to String.
pub fn str2vec(s: String, size: usize) -> Vec<f64> {
    let mut v = s
        .as_bytes()
        .iter()
        .map(|&x| (x as f64) / 256.0)
        .collect::<Vec<f64>>();
    v.resize(size, 0.0);
    v
}

/// Convert String to Vec.
pub fn vec2str(v: &Vec<f64>) -> String {
    v.iter()
        .map(|&x| ((x * 256.0).round() as u8) as char)
        .collect::<String>()
        .trim()
        .to_owned()
}

/// Generate the training data.
pub fn load_examples(filename: &str, examples: &mut Vec<(Vec<f64>, Vec<f64>)>) {
    let count;
    {
        let r = File::open(filename).expect("Unable to open file");
        let f = BufReader::new(r);
        count = f.lines().count();
    }
    let r = File::open(filename).expect("Unable to open file");
    let f = BufReader::new(r);
    let mut capacity = 0;

    let bar = ProgressBar::new(count as u64);
    bar.set_style(ProgressStyle::default_bar().template(
        "Load: [{elapsed_precise}] {bar:40} {percent}% - {pos}/{len} - {per_sec} - ETA {eta}",
    ));

    for line in f.lines() {
        let line = line.expect("Unable to read line");
        let a1: Vec<&str> = line.rsplit('-').collect();

        if a1.len() >= 3 {
            let label = a1[0].trim().to_owned();
            let document = (a1[2].to_owned() + "-" + &a1[1].to_owned())
                .trim()
                .to_owned();

            examples.push((str2vec(document, 80), str2vec(label, 4)));
            capacity += 1;
        }
        if capacity % 5_000 == 0 {
            bar.inc(5_000);
        }
    }

    bar.finish();

    println!("{} examples loaded.", capacity);
}
