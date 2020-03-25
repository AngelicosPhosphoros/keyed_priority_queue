extern crate criterion;

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use keyed_priority_queue::KeyedPriorityQueue;

mod generators;
use crate::generators::{choose_some, gen_random_usizes, get_random_strings};

pub fn bench_set_priority(c: &mut Criterion) {
    let base_keys = gen_random_usizes(500_000, 0);
    let base_values = gen_random_usizes(500_000, 7);

    let mut group = c.benchmark_group("set_priority_usize");
    for &size in &[10_000, 500_000] {
        assert!(base_keys.len() >= size);

        let test_keys: Vec<_> = choose_some(&base_keys[..size], 500, 500);
        let test_vals: Vec<_> = gen_random_usizes(500, 564);

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let base_queue: KeyedPriorityQueue<_, _> = base_keys[..size]
                .iter()
                .cloned()
                .zip(base_values[..size].iter().cloned())
                .collect();
            b.iter_batched(
                || {
                    (
                        base_queue.clone(),
                        Vec::<Result<usize, _>>::with_capacity(test_keys.len()),
                    )
                },
                |(mut queue, mut results_store)| {
                    for (&k, &v) in test_keys.iter().zip(test_vals.iter()) {
                        results_store.push(queue.set_priority(&k, v));
                    }
                    (queue, results_store)
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();

    let mut group = c.benchmark_group("set_priority_string");
    let base_keys = get_random_strings(50_000, 0);
    let base_values = get_random_strings(50_000, 7);

    for &size in &[1_000, 50_000] {
        assert!(base_keys.len() >= size);

        let test_keys: Vec<_> = choose_some(&base_keys[..size], 500, 500);
        let test_vals: Vec<_> = get_random_strings(500, 564);

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let base_queue: KeyedPriorityQueue<_, _> = base_keys[..size]
                .iter()
                .cloned()
                .zip(base_values[..size].iter().cloned())
                .collect();
            b.iter_batched(
                || {
                    (
                        base_queue.clone(),
                        Vec::<Result<String, _>>::with_capacity(test_keys.len()),
                    )
                },
                |(mut queue, mut results_store)| {
                    for (k, v) in test_keys.iter().zip(test_vals.iter()) {
                        results_store.push(queue.set_priority(k, v.clone()));
                    }
                    (queue, results_store)
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_set_priority);
criterion_main!(benches);
