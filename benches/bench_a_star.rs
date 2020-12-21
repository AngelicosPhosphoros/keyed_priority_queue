use std::cmp::Reverse;
use std::ops::Index;

#[derive(Eq, PartialEq, Debug, Hash, Copy, Clone, Ord, PartialOrd)]
struct Position {
    row: usize,
    column: usize,
}

struct Field {
    rows: usize,
    columns: usize,
    costs: Box<[u32]>,
}

impl Index<Position> for Field {
    type Output = u32;

    fn index(&self, index: Position) -> &Self::Output {
        &self.costs[self.columns * index.row + index.column]
    }
}

struct Neighbours {
    len: usize,
    items: [Position; 8],
}

fn get_neighbors(pos: Position, field: &Field) -> Neighbours {
    let mut items = [pos; 8];
    let mut length = 0usize;
    if pos.row > 0 {
        items[length].row -= 1;
        length += 1;
    }
    if pos.row + 1 < field.rows {
        items[length].row += 1;
        length += 1;
    }
    if pos.column > 0 {
        items[length].column -= 1;
        length += 1;
    }
    if pos.column + 1 < field.columns {
        items[length].column += 1;
        length += 1;
    }

    if pos.row > 0 && pos.column > 0 {
        items[length].row -= 1;
        items[length].column -= 1;
        length += 1
    }

    if pos.row > 0 && pos.column + 1 < field.columns {
        items[length].row -= 1;
        items[length].column += 1;
        length += 1
    }

    if pos.row + 1 < field.rows && pos.column > 0 {
        items[length].row += 1;
        items[length].column -= 1;
        length += 1
    }

    if pos.row + 1 < field.rows && pos.column + 1 < field.columns {
        items[length].row += 1;
        items[length].column += 1;
        length += 1
    }

    Neighbours { len: length, items }
}

mod std_a_star {
    use super::*;
    use fxhash::{FxHashMap, FxHashSet};
    use std::collections::BinaryHeap;

    pub(crate) fn find_path(
        start: Position,
        target: Position,
        field: &Field,
    ) -> Option<Vec<Position>> {
        if start == target {
            return Some(vec![start]);
        }
        let calc_heuristic = |pos: Position| -> usize {
            ((target.row as isize - pos.row as isize).abs()
                + (target.column as isize - pos.column as isize).abs()) as usize
        };

        fn restore_path(
            pos: Position,
            parentize: &FxHashMap<Position, Position>,
            start: Position,
        ) -> Vec<Position> {
            let mut result = Vec::new();
            let mut current = pos;
            loop {
                result.push(current);
                if current == start {
                    result.reverse();
                    return result;
                }
                current = parentize[&current];
            }
        }

        // Child to its parent
        let mut parentize: FxHashMap<Position, Position> = FxHashMap::default();
        // Already checked
        let mut closed_set: FxHashSet<Position> = FxHashSet::default();
        // Total cost (with estimate), real cost and positions
        let mut available: BinaryHeap<Reverse<(usize, usize, Position)>> = BinaryHeap::new();
        // Position to minimal total cost. Used to decide is need to enter new val into heap
        let mut remembered_nodes: FxHashMap<Position, usize> = FxHashMap::default();
        available.push(Reverse((0 + calc_heuristic(start), 0, start)));
        while let Some(Reverse((_, current_cost, current_pos))) = available.pop() {
            if current_pos == target {
                return Some(restore_path(current_pos, &parentize, start));
            }

            // No more need to remember
            remembered_nodes.remove(&current_pos);
            // Case when we handled this point earlier
            // Can be, because we add point to `available` multiple times if our calculated real cost changes
            if closed_set.contains(&current_pos) {
                continue;
            }

            closed_set.insert(current_pos);

            let neighbours = get_neighbors(current_pos, &field);
            for next in neighbours.items[..neighbours.len]
                .iter()
                .cloned()
                .filter(|x| !closed_set.contains(x))
            {
                let real_cost = field[next] as usize + current_cost;
                let total = real_cost + calc_heuristic(next);
                if !remembered_nodes.contains_key(&next) || remembered_nodes[&next] > total {
                    remembered_nodes.insert(next, total);
                    parentize.insert(next, current_pos);
                    available.push(Reverse((total, real_cost, next)));
                }
            }
        }
        None
    }
}

mod keyed_a_star {
    use super::*;
    use fxhash::{FxHashMap, FxHashSet};
    use keyed_priority_queue::{Entry, KeyedPriorityQueue};
    use std::hash::BuildHasher;

    pub(crate) fn find_path<HasherParam: BuildHasher + Default>(
        start: Position,
        target: Position,
        field: &Field,
    ) -> Option<Vec<Position>> {
        if start == target {
            return Some(vec![start]);
        }
        let calc_heuristic = |pos: Position| -> usize {
            ((target.row as isize - pos.row as isize).abs()
                + (target.column as isize - pos.column as isize).abs()) as usize
        };

        fn restore_path(
            pos: Position,
            parentize: &FxHashMap<Position, Position>,
            start: Position,
        ) -> Vec<Position> {
            let mut result = Vec::new();
            let mut current = pos;
            loop {
                result.push(current);
                if current == start {
                    result.reverse();
                    return result;
                }
                current = parentize[&current];
            }
        }

        // Child to its parent
        let mut parentize: FxHashMap<Position, Position> = FxHashMap::default();
        // Already checked
        let mut closed_set: FxHashSet<Position> = FxHashSet::default();
        // Positions sortered by total cost and real cost.
        // We prefer items with lower real cost if total are same.
        #[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
        struct Cost {
            total: usize,
            real: usize,
        }
        let mut available = KeyedPriorityQueue::<Position, Reverse<Cost>, HasherParam>::default();
        available.push(
            start,
            Reverse(Cost {
                total: calc_heuristic(start),
                real: 0,
            }),
        );
        while let Some((current_pos, Reverse(current_cost))) = available.pop() {
            if current_pos == target {
                return Some(restore_path(current_pos, &parentize, start));
            }

            closed_set.insert(current_pos);

            let neighbours = get_neighbors(current_pos, &field);
            for next in neighbours.items[..neighbours.len]
                .iter()
                .cloned()
                .filter(|x| !closed_set.contains(x))
            {
                let real = field[next] as usize + current_cost.real;
                let total = current_cost.real + calc_heuristic(next);
                let cost = Cost { total, real };
                match available.entry(next) {
                    Entry::Vacant(entry) => {
                        entry.set_priority(Reverse(cost));
                        parentize.insert(next, current_pos);
                    }
                    Entry::Occupied(entry) if *entry.get_priority() < Reverse(cost) => {
                        entry.set_priority(Reverse(cost));
                        parentize.insert(next, current_pos);
                    }
                    _ => {}
                }
            }
        }
        None
    }
}

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn generate_field(size: usize) -> Field {
    const SEED: u64 = 546579634698731;
    use rand::prelude::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let dist = rand::distributions::Uniform::new_inclusive(1u32, 10u32);
    let vec: Vec<u32> = (0..size * size)
        .into_iter()
        .map(|_| rng.sample(dist))
        .collect();
    Field {
        columns: size,
        rows: size,
        costs: vec.into(),
    }
}

fn find_path_benchmark(c: &mut Criterion) {
    let field = generate_field(100);
    let mut group = c.benchmark_group("A_Stars");
    for &end in &[1, 5, 10, 25, 45, 49, 99] {
        let start = Position { row: 0, column: 0 };
        let stop_at = Position {
            row: end,
            column: end,
        };
        group.bench_with_input(
            BenchmarkId::new("STD A Star", end),
            &(start, stop_at, &field),
            |b, &(start, target, field)| b.iter(|| std_a_star::find_path(start, target, field)),
        );
        group.bench_with_input(
            BenchmarkId::new("Keyed A Star", end),
            &(start, stop_at, &field),
            |b, &(start, target, field)| {
                b.iter(|| {
                    keyed_a_star::find_path::<std::collections::hash_map::RandomState>(
                        start, target, field,
                    )
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("Keyed A Star FxHash", end),
            &(start, stop_at, &field),
            |b, &(start, target, field)| {
                b.iter(|| keyed_a_star::find_path::<fxhash::FxBuildHasher>(start, target, field))
            },
        );
    }
    const BIG_SIZE: usize = 500;
    let field_eq = Field {
        columns: BIG_SIZE,
        rows: BIG_SIZE,
        costs: vec![1; BIG_SIZE * BIG_SIZE].into_boxed_slice(),
    };

    let start = Position { row: 0, column: 0 };
    let stop_at = Position {
        row: BIG_SIZE - 1,
        column: BIG_SIZE - 1,
    };
    group.bench_with_input(
        BenchmarkId::new("STD A Star Ones field", BIG_SIZE),
        &(start, stop_at, &field),
        |b, _| b.iter(|| std_a_star::find_path(start, stop_at, &field_eq)),
    );
    group.bench_with_input(
        BenchmarkId::new("Keyed A Star Ones field", BIG_SIZE),
        &(start, stop_at, &field),
        |b, _| {
            b.iter(|| {
                keyed_a_star::find_path::<std::collections::hash_map::RandomState>(
                    start, stop_at, &field_eq,
                )
            })
        },
    );
    group.bench_with_input(
        BenchmarkId::new("Keyed A Star Ones field FxHash", BIG_SIZE),
        &(start, stop_at, &field),
        |b, _| {
            b.iter(|| keyed_a_star::find_path::<fxhash::FxBuildHasher>(start, stop_at, &field_eq))
        },
    );

    group.finish();
}

criterion_group!(benches, find_path_benchmark);
criterion_main!(benches);
