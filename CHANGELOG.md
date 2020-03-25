# Change Log

## 2020-03-25: 0.2.1
Fixed typo in Readme.md

## 2020-03-25: 0.2.0
## Changes
### API
- Trait `Clone` is no more required for keys (Since it stored only once)
- Renamed method `remove_item` to `remove`
- Added method `remove_entry` which returns both key and priority
- `push` operation returns old priority now if same key already exists
- Method `set_priority` returns `Result<TPriority, SetPriorityNotFoundError>` with old priority instead of panicing on missing keys
- Added Entry API to allow whole cycle `Find -> Read -> Update` with just one hashmap lookup.
- Added borrowing unordered iterator (by method `iter`) over which will iterate over whole queue in O(n)
- Improved documentation by a little
- Added `#[forbid(unsafe_code)]`
### Implementation
- Now uses IndexMap from [indexmap](https://crates.io/crates/indexmap) crate internally


## 2020-02-25: 0.1.3
## Changes
- Removed unsafe implementations of Sync + Send because they are deduced by compiler
- Made some optimizations which reduce timings by 50% but increase memory usage in worst case on 30%
- Added benchmarks

## 2019-11-24: 0.1.2
### Added
- Now items in queue can be looked up borrow using result, e.g. if `String` struct used as key, `&str` can be passed as lookup key.

## 2019-10-27: 0.1.1
### Added
- Now `KeyedPriorityQueue` implements `Default` trait

### Changes
- Some clippy fixes
