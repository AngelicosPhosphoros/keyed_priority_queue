use rand::prelude::{Rng, SeedableRng};
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;

const STRING_SIZE: usize = 100;

#[allow(dead_code)]
pub(crate) fn gen_random_usizes(n: usize, seed: u64) -> Vec<usize> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let dist = rand::distributions::Uniform::new_inclusive(1usize, 40_000_000usize);
    let mut res = Vec::with_capacity(n);
    for _ in 0..n {
        res.push(rng.sample(dist))
    }
    res
}

#[allow(dead_code)]
pub(crate) fn get_random_strings(n: usize, seed: u64) -> Vec<String> {
    let alphabet: Vec<char> = (0u8..0x7f)
        .into_iter()
        .filter(|x| x.is_ascii_alphanumeric())
        .map(|x| x as char)
        .collect();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut res = Vec::with_capacity(n);
    for _ in 0..n {
        let s: String = alphabet[..]
            .choose_multiple(&mut rng, STRING_SIZE)
            .collect();
        res.push(s);
    }
    res
}

#[allow(dead_code)]
pub(crate) fn get_unique_random_strings(n: usize, seed: u64) -> Vec<String> {
    use std::collections::HashSet;

    let alphabet: Vec<char> = (0u8..0x7f)
        .into_iter()
        .filter(|x| x.is_ascii_alphanumeric())
        .map(|x| x as char)
        .collect();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut res = HashSet::with_capacity(n);
    while res.len() < n {
        let s: String = alphabet[..]
            .choose_multiple(&mut rng, STRING_SIZE)
            .collect();
        res.insert(s);
    }
    res.into_iter().collect()
}

#[allow(dead_code)]
pub(crate) fn choose_some<T>(vals: &[T], num: usize, seed: u64) -> Vec<T>
where
    T: Clone,
{
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    vals.choose_multiple(&mut rng, num).cloned().collect()
}

/// Splits the data in to parts:
/// 1. random shuffled first output for filling queue
/// 2. `number_for_push` maximal items sorted in ascending order
/// ## Panics
/// If number_for_push bigger than data.len()
#[allow(dead_code)]
pub(crate) fn generate_worst_push_data<T: Ord + Clone>(
    mut data: Vec<T>,
    number_for_push: usize,
    seed: u64,
) -> (Vec<T>, Vec<T>) {
    if number_for_push > data.len() {
        panic!(
            "number_for_push {} MUST be less or equal data length {}",
            number_for_push,
            data.len()
        );
    }
    data.sort_unstable();
    let remain_length = data.len() - number_for_push;
    let for_pushes = data[remain_length..].to_vec();
    data.truncate(remain_length);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    data.shuffle(&mut rng);
    (data, for_pushes)
}
