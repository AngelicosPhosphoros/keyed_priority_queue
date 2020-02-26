extern crate criterion;

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use keyed_priority_queue::KeyedPriorityQueue;

mod generators;
use crate::generators::{choose_some, gen_random_usizes, get_random_strings};

/* BECNHES */

pub fn bench_get_priority(c: &mut Criterion) {
    let base_keys = gen_random_usizes(500_000, 0);
    let base_values = gen_random_usizes(500_000, 7);

    let mut group = c.benchmark_group("get_priority_usize");
    for &size in &[10_000, 500_000] {
        assert!(base_keys.len() >= size);

        let test_keys: Vec<_> = choose_some(&base_keys[..size], 500, 500);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let base_queue: KeyedPriorityQueue<_, _> = base_keys[..size]
                .iter()
                .cloned()
                .zip(base_values[..size].iter().cloned())
                .collect();
            b.iter_batched(
                || base_queue.clone(),
                |queue| {
                    for k in test_keys.iter() {
                        black_box(queue.get_priority(k));
                    }
                    queue
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();

    let mut group = c.benchmark_group("get_priority_string");
    let base_keys = get_random_strings(50_000, 0);
    let base_values = get_random_strings(50_000, 7);

    for &size in &[1_000, 50_000] {
        assert!(base_keys.len() >= size);

        let test_keys: Vec<_> = choose_some(&base_keys[..size], 500, 500);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let base_queue: KeyedPriorityQueue<_, _> = base_keys[..size]
                .iter()
                .cloned()
                .zip(base_values[..size].iter().cloned())
                .collect();
            b.iter_batched(
                || base_queue.clone(),
                |queue| {
                    for k in test_keys.iter() {
                        black_box(queue.get_priority(k));
                    }
                    queue
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_get_priority);
criterion_main!(benches);
