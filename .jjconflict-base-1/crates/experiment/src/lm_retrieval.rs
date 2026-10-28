//! Fixed byte-vocabulary associative recall diagnostic; no corpus text or training labels are reused.
use crate::{
    config::Config,
    data::{Example, Rng},
};

pub fn examples(c: &Config) -> Vec<Example> {
    let count = ((c.context - 8) / 4).clamp(1, 16);
    (0..c.eval_examples)
        .map(|index| {
            let mut rng = Rng(c.eval_seed.wrapping_add(index as u64 * 7919));
            let mut keys: Vec<u8> = (b'A'..=b'Z').collect();
            rng.shuffle(&mut keys);
            keys.truncate(count);
            let values: Vec<u8> = (0..count).map(|_| b'a' + rng.index(26) as u8).collect();
            let query = index % count;
            let mut bytes = Vec::with_capacity(c.context);
            for (&key, &value) in keys.iter().zip(&values) {
                bytes.extend_from_slice(&[key, b':', value, b';']);
            }
            bytes.resize(c.context - 3, b' ');
            bytes.extend_from_slice(&[b' ', keys[query], b':']);
            let mut target = vec![0; c.context];
            target[c.context - 1] = values[query] as i32;
            let mut weight = vec![0.0; c.context];
            weight[c.context - 1] = 1.0;
            let mut groups = vec![None; c.context];
            groups[c.context - 1] = Some((count, c.context - 1 - query * 4));
            Example {
                input: bytes.into_iter().map(i32::from).collect(),
                target,
                weight,
                groups,
            }
        })
        .collect()
}
