# Keyed Priority Queue

[![Crates.io](https://img.shields.io/crates/v/keyed_priority_queue)](https://crates.io/crates/keyed_priority_queue)
[![Build Status](https://travis-ci.org/AngelicosPhosphoros/keyed_priority_queue.svg?branch=master)](https://travis-ci.org/AngelicosPhosphoros/keyed_priority_queue)
[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE.md)
[![Average time to resolve an issue](https://isitmaintained.com/badge/resolution/AngelicosPhosphoros/keyed_priority_queue.svg)](https://isitmaintained.com/project/AngelicosPhosphoros/keyed_priority_queue "Average time to resolve an issue")
[![Percentage of issues still open](https://isitmaintained.com/badge/open/AngelicosPhosphoros/keyed_priority_queue.svg)](https://isitmaintained.com/project/AngelicosPhosphoros/keyed_priority_queue "Percentage of issues still open")


A Rust library with priority queue that supports changing of priority item in queue or early removal.
To change priority you need to use some key.

## Usage

Add this to your `Cargo.toml`:
```toml
[dependencies]
keyed_priority_queue = "0.1.*"
```

The example of code:

```rust
use keyed_priority_queue::KeyedPriorityQueue;

let mut queue = KeyedPriorityQueue::new();

// Currently queue is empty
assert_eq!(queue.peek(), None);

queue.push("Second", 4);
queue.push("Third", 3);
queue.push("First", 5);
queue.push("Fourth", 2);
queue.push("Fifth", 1);

// Peek return references to most important pair.
assert_eq!(queue.peek(), Some((&"First", &5)));

assert_eq!(queue.len(), 5);

// We can clone queue if both key and priority is clonable
let mut queue_clone = queue.clone();

// We can run consuming iterator on queue,
// and it will return items in decreasing order
for (key, priority) in queue_clone{
    println!("Priority of key {} is {}", key, priority);
}

// Popping always will return the biggest element
assert_eq!(queue.pop(), Some(("First", 5)));
// We can change priority of item by key:
queue.set_priority(&"Fourth", 10);
// And get it
assert_eq!(queue.get_priority(&"Fourth"), Some(&10));
// Now biggest element is Fourth
assert_eq!(queue.pop(), Some(("Fourth", 10)));
// We can also decrease priority!
queue.set_priority(&"Second", -1);
assert_eq!(queue.pop(), Some(("Third", 3)));
assert_eq!(queue.pop(), Some(("Fifth", 1)));
assert_eq!(queue.pop(), Some(("Second", -1)));
// Now queue is empty
assert_eq!(queue.pop(), None);

// We can clear queue
queue.clear();
assert!(queue.is_empty());
```

