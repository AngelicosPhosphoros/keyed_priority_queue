extern crate criterion;

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use keyed_priority_queue::KeyedPriorityQueue;

mod generators;
use crate::generators::{gen_random_usizes, get_random_strings};

pub fn bench_pop(c: &mut Criterion) {
    let base_keys = gen_random_usizes(500_000, 0);
    let base_values = gen_random_usizes(500_000, 7);

    let mut group = c.benchmark_group("pop_usize");
    for &size in &[100_000, 200_000, 300_000, 400_000, 500_000] {
        assert!(base_keys.len() >= size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let base_queue: KeyedPriorityQueue<usize, usize> = base_keys[..size]
                .iter()
                .cloned()
                .zip(base_values[..size].iter().cloned())
                .collect();
            b.iter_batched(
                || base_queue.clone(),
                |mut queue| {
                    for _ in 0..1000 {
                        queue.pop();
                    }
                    queue
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();

    let mut group = c.benchmark_group("pop_string");
    let base_keys = get_random_strings(50_000, 0);
    let base_values = get_random_strings(50_000, 7);

    for &size in &[10_000, 20_000, 30_000, 40_000, 50_000] {
        assert!(base_keys.len() >= size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let base_queue: KeyedPriorityQueue<String, String> = base_keys[..size]
                .iter()
                .cloned()
                .zip(base_values[..size].iter().cloned())
                .collect();
            b.iter_batched(
                || base_queue.clone(),
                |mut queue| {
                    for _ in 0..1000 {
                        queue.pop();
                    }
                    queue
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_pop);
criterion_main!(benches);
