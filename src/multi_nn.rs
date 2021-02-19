extern crate scoped_threadpool;

use crate::nn::{HaltCondition, NN};
use crate::utils::{str2vec, vec2str};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use scoped_threadpool::Pool;
use std::fs::File;
use std::io::{Read, Write};
use std::sync::Arc;

const EPOCHS: u32 = 100;

fn split_examples(
    examples: &Vec<(Vec<f64>, Vec<f64>)>,
    n: usize,
) -> Vec<Vec<(Vec<f64>, Vec<f64>)>> {
    let mut results: Vec<Vec<(Vec<f64>, Vec<f64>)>> = Vec::new();
    for _ in 0..n {
        results.push(Vec::new());
    }
    for x in examples {
        let s = vec2str(&x.0[..9].to_owned());
        let s = s.as_bytes();
        let mut num: usize = 0;
        for i in s {
            num += *i as usize;
        }
        let idx = num % n;
        results[idx].push(x.clone());
    }
    results
}

pub fn train_multiple(examples: &Vec<(Vec<f64>, Vec<f64>)>, n: usize) {
    let mut pool = Pool::new(n as u32);
    let s1 = split_examples(&examples, n);
    let m = MultiProgress::new();
    pool.scoped(|scoped| {
        for i in 0..n {
            let example_count = s1[i].len();
            let bar = ProgressBar::new((example_count * EPOCHS as usize) as u64);
            bar.set_style(ProgressStyle::default_bar()
                .template(&(format!("Net #{}: ", i) + "{prefix} [{elapsed_precise}] {bar:40} {percent:>3}% - {pos}/{len} - {per_sec} - eta {eta} - {msg}")));
            let pb = m.add(bar);
            let v = Arc::new(&s1[i]);
            scoped.execute(move || {
                let filename = format!("./data/net-{}.json", i);
                let mut net: NN = if std::fs::metadata(filename.clone()).is_err() {
                    NN::new(&[80, 120, 120, 4])
                } else {
                    let mut f = File::open(filename.clone()).unwrap();
                    let mut buffer = String::new();
                    f.read_to_string(&mut buffer).unwrap();
                    NN::from_json(&buffer)
                };

                net.train(&v)
                .halt_condition(HaltCondition::Epochs(EPOCHS))
                .momentum(0.9)
                .rate(0.07)
                .go(Box::new(move |s, epochs, i, _, training_error_rate| {
                    match s {
                        1 => {
                            pb.set_prefix(&format!("epoch {:>3}", epochs + 1));
                            pb.tick();
                        },
                        2 => if i % 1_000 == 0 {
                            pb.set_position((example_count * epochs as usize + i) as u64);
                        },
                        3 => {
                            pb.set_message(&format!("error rate={}", training_error_rate));
                            pb.set_position((example_count * epochs as usize + i) as u64);
                        },
                        4 => pb.finish(),
                        _ => ()
                    }
                }));

                let mut f = File::create(filename).unwrap();
                f.write(net.to_json().as_bytes()).unwrap();
            });
        }
        m.join().unwrap();
        scoped.join_all();
    });
}

pub fn run(example_document: String, n: usize) -> String {
    let s = &example_document.as_bytes()[0..9];
    let mut num: usize = 0;
    for i in s {
        num += *i as usize;
    }
    let idx = num % n;
    println!("Selected net #{}", idx);

    let mut f = File::open(format!("./data/net-{}.json", idx)).unwrap();
    let mut buffer = String::new();
    f.read_to_string(&mut buffer).unwrap();
    let net: NN = NN::from_json(&buffer);

    let results = net.run(&str2vec(example_document, 80));
    vec2str(&results)
}
