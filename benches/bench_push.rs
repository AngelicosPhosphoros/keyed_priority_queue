extern crate criterion;

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use keyed_priority_queue::KeyedPriorityQueue;

mod generators;
use crate::generators::{
    gen_random_usizes, generate_worst_push_data, get_random_strings, get_unique_random_strings,
};

pub fn bench_push(c: &mut Criterion) {
    let base_keys = gen_random_usizes(500_000, 0);
    let base_values = gen_random_usizes(500_000, 7);

    let extra_keys = gen_random_usizes(1000, 8);
    let extra_values = gen_random_usizes(1000, 20);
    let extra: Vec<_> = extra_keys
        .into_iter()
        .zip(extra_values.into_iter())
        .collect();

    let mut group = c.benchmark_group("push_usizes_random");
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
                    for (k, v) in extra.iter().cloned() {
                        queue.push(k, v);
                    }
                    queue
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();

    let mut group = c.benchmark_group("push_strings_random");
    let base_keys = get_random_strings(50_000, 0);
    let base_values = get_random_strings(50_000, 7);

    let extra_keys = get_random_strings(1000, 8);
    let extra_values = get_random_strings(1000, 20);
    let extra: Vec<_> = extra_keys
        .into_iter()
        .zip(extra_values.into_iter())
        .collect();

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
                    for (k, v) in extra.iter().cloned() {
                        queue.push(k, v);
                    }
                    queue
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();

    let mut base_keys: Vec<usize> = (0..520_000).into_iter().collect();
    let base_values = gen_random_usizes(520_000, 7);

    let extra_keys: Vec<_> = base_keys[500_000..].into();
    base_keys.truncate(500_000);
    let (base_values, extra_values) = generate_worst_push_data(base_values, 20_000, 987987);
    let extra: Vec<_> = extra_keys
        .into_iter()
        .zip(extra_values.into_iter())
        .collect();

    let mut group = c.benchmark_group("push_usizes_worst");
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
                    for (k, v) in extra.iter().cloned() {
                        queue.push(k, v);
                    }
                    queue
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();

    let mut base_keys: Vec<String> = get_unique_random_strings(55_000, 987987);
    let base_values = get_unique_random_strings(55_000, 23423);

    let extra_keys: Vec<_> = base_keys[50_000..].iter().cloned().collect();
    base_keys.truncate(50_000);
    let (base_values, extra_values) =
        generators::generate_worst_push_data(base_values, 5_000, 987987);
    let extra: Vec<_> = extra_keys
        .into_iter()
        .zip(extra_values.into_iter())
        .collect();

    let mut group = c.benchmark_group("push_strings_worst");
    for &size in &[10_000, 20_000, 30_000, 40_000, 50_000] {
        assert!(base_keys.len() >= size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let base_queue: KeyedPriorityQueue<_, _> = base_keys[..size]
                .iter()
                .cloned()
                .zip(base_values[..size].iter().cloned())
                .collect();
            b.iter_batched(
                || base_queue.clone(),
                |mut queue| {
                    for (k, v) in extra.iter().cloned() {
                        queue.push(k, v);
                    }
                    queue
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, bench_push);
criterion_main!(benches);
