extern crate criterion;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use keyed_priority_queue::KeyedPriorityQueue;

mod generators;
use crate::generators::{gen_random_usizes, get_random_strings};

pub fn bench_from_iter(c: &mut Criterion) {
    let base_keys = gen_random_usizes(100_000, 0);
    let base_values = gen_random_usizes(100_000, 7);

    let mut group = c.benchmark_group("from_iter_usize");
    for &size in &[20_000, 40_000, 60_000, 80_000, 100_000] {
        assert!(base_keys.len() >= size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let queue: KeyedPriorityQueue<_, _> = base_keys[..size]
                    .iter()
                    .cloned()
                    .zip(base_values[..size].iter().cloned())
                    .collect();
                black_box(queue)
            });
        });
    }

    group.finish();

    let mut group = c.benchmark_group("from_iter_string");
    let base_keys = get_random_strings(50_000, 0);
    let base_values = get_random_strings(50_000, 7);

    for &size in &[10_000, 20_000, 30_000, 40_000, 50_000] {
        assert!(base_keys.len() >= size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let queue: KeyedPriorityQueue<_, _> = base_keys[..size]
                    .iter()
                    .cloned()
                    .zip(base_values[..size].iter().cloned())
                    .collect();
                black_box(queue)
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_from_iter);
criterion_main!(benches);
