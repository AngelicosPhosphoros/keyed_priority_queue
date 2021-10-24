//! This is priority queue that supports elements priority modification and early removal.
//!
//! It uses HashMap and own implementation of binary heap to achieve this.
//!
//! Each entry has associated *key* and *priority*.
//! Keys must be unique, and hashable; priorities must implement Ord trait.
//!
//! Popping returns element with biggest priority.
//! Pushing adds element to queue.
//! Also it is possible to change priority or remove item by key.
//!
//! Pop, push, change priority, remove by key have ***O(log n)*** time complexity;
//! peek, lookup by key are ***O(1)***.
//!
//! # Examples
//!
//! This is implementation of [A* algorithm][a_star] for 2D grid.
//! Each cell in grid has the cost.
//! This algorithm finds shortest path to target using heuristics.
//!
//! Let open set be the set of position where algorithm can move in next step.
//! Sometimes better path for node in open set is found
//! so the priority of it needs to be updated with new value.
//!
//! This example shows how to change priority in [`KeyedPriorityQueue`] when needed.
//!
//! [a_star]: https://en.wikipedia.org/wiki/A*_search_algorithm
//! [`KeyedPriorityQueue`]: struct.KeyedPriorityQueue.html
//!
//! ```
//! use keyed_priority_queue::{KeyedPriorityQueue, Entry};
//! use std::cmp::Reverse;
//! use std::collections::HashSet;
//! use std::ops::Index;
//!
//! struct Field {
//!     rows: usize,
//!     columns: usize,
//!     costs: Box<[u32]>,
//! }
//!
//! #[derive(Eq, PartialEq, Debug, Hash, Copy, Clone)]
//! struct Position {
//!     row: usize,
//!     column: usize,
//! }
//!
//! impl Index<Position> for Field {
//!     type Output = u32;
//!
//!     fn index(&self, index: Position) -> &Self::Output {
//!         &self.costs[self.columns * index.row + index.column]
//!     }
//! }
//!
//! // From cell we can move upper, right, bottom and left
//! fn get_neighbors(pos: Position, field: &Field) -> Vec<Position> {
//!     let mut items = Vec::with_capacity(4);
//!     if pos.row > 0 {
//!         items.push(Position { row: pos.row - 1, column: pos.column });
//!     }
//!     if pos.row + 1 < field.rows {
//!         items.push(Position { row: pos.row + 1, column: pos.column });
//!     }
//!     if pos.column > 0 {
//!         items.push(Position { row: pos.row, column: pos.column - 1 });
//!     }
//!     if pos.column + 1 < field.columns {
//!         items.push(Position { row: pos.row, column: pos.column + 1 });
//!     }
//!     items
//! }
//!
//! fn find_path(start: Position, target: Position, field: &Field) -> Option<u32> {
//!     if start == target {
//!         return Some(field[start]);
//!     }
//!     let calc_heuristic = |pos: Position| -> u32 {
//!         ((target.row as isize - pos.row as isize).abs()
//!             + (target.column as isize - pos.column as isize).abs()) as u32
//!     };
//!
//!     // Already handled this points
//!     let mut closed_set: HashSet<Position> = HashSet::new();
//!     // Positions sortered by total cost and real cost.
//!     // We prefer items with lower real cost if total ones are same.
//!     #[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
//!     struct Cost {
//!         total: u32,
//!         real: u32,
//!     }
//!     // Queue that contains all nodes that available for next step
//!     // Min-queue required so Reverse struct used as priority.
//!     let mut available = KeyedPriorityQueue::<Position, Reverse<Cost>>::new();
//!     available.push(
//!         start,
//!         Reverse(Cost {
//!             total: calc_heuristic(start),
//!             real: 0,
//!         }),
//!     );
//!     while let Some((current_pos, Reverse(current_cost))) = available.pop() {
//!         // We have reached target
//!         if current_pos == target {
//!             return Some(current_cost.real);
//!         }
//!
//!         closed_set.insert(current_pos);
//!
//!         for next in get_neighbors(current_pos, &field).into_iter()
//!             .filter(|x| !closed_set.contains(x))
//!             {
//!                 let real = field[next] + current_cost.real;
//!                 let total = current_cost.real + calc_heuristic(next);
//!                 let cost = Cost { total, real };
//!                 // Entire this interaction will make only one hash lookup
//!                 match available.entry(next) {
//!                     Entry::Vacant(entry) => {
//!                         // Add new position to queue
//!                         entry.set_priority(Reverse(cost));
//!                     }
//!                     Entry::Occupied(entry) if *entry.get_priority() < Reverse(cost) => {
//!                         // Have found better path to node in queue
//!                         entry.set_priority(Reverse(cost));
//!                     }
//!                     _ => { /* Have found worse path. */ }
//!                 };
//!             }
//!     }
//!     None
//! }
//!
//!let field = Field {
//!    rows: 4,
//!    columns: 4,
//!    costs: vec![
//!        1, 3, 3, 6, //
//!        4, 4, 3, 8, //
//!        3, 1, 2, 4, //
//!        4, 8, 9, 4, //
//!    ].into_boxed_slice(),
//!};
//!
//!let start = Position { row: 0, column: 0 };
//!let end = Position { row: 3, column: 3 };
//!assert_eq!(find_path(start, end, &field), Some(18));
//! ```
//!

mod editable_binary_heap;
mod keyed_priority_queue;
mod mediator;

pub use crate::keyed_priority_queue::{
    Entry, KeyedPriorityQueue, KeyedPriorityQueueBorrowIter, KeyedPriorityQueueIterator,
    OccupiedEntry, SetPriorityNotFoundError, VacantEntry,
};

#[doc = include_str!("../../Readme.md")]
#[cfg(doctest)]
pub struct ReadmeDoctests;
