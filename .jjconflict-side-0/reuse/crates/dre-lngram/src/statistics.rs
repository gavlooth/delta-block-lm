//! Host statistics of the discrete route codes (spec §3.3, §11.5): route entropy, symbol and key utilization,
//! lookup hit diversity and whole-code entropy, overall or per group of positions (e.g. per script).

use std::collections::{BTreeMap, HashMap, HashSet};

/// Route codes of a batch copied to the host. Flat position index is `b · tokens + t`.
#[derive(Debug, Clone)]
pub struct HostRouteCodes {
    pub batch: usize,
    pub tokens: usize,
    pub routes: usize,
    /// Symbols per route, `K`.
    pub alphabet: usize,
    pub orders: Vec<usize>,
    /// `false` at padding, `[B · T]`.
    pub valid: Vec<bool>,
    /// Route symbols, `[B · T · R]`.
    pub symbols: Vec<i64>,
    /// Per order: global table row, `-1` where the position has no full n-gram, `[B · T · R]`.
    pub addresses: Vec<Vec<i64>>,
}

/// Code statistics over a set of valid positions. Entropies are in bits.
#[derive(Debug, Clone, PartialEq)]
pub struct CodeStatistics {
    /// Valid positions included.
    pub positions: usize,
    /// Mean over routes of the symbol entropy (maximum `bits_per_route`).
    pub route_entropy_mean: f64,
    /// Smallest symbol entropy of any route; near 0 means a collapsed route.
    pub route_entropy_min: f64,
    /// Mean over routes of `distinct symbols / K`.
    pub symbol_utilization: f64,
    /// Distinct whole codes (all routes of one position together).
    pub distinct_codes: usize,
    /// Entropy of the whole-code distribution (maximum `log2(positions)`).
    pub code_entropy: f64,
    pub orders: Vec<OrderUtilization>,
}

/// Table use of one n-gram order.
#[derive(Debug, Clone, PartialEq)]
pub struct OrderUtilization {
    pub order: usize,
    /// Lookups made (positions with a full n-gram × routes).
    pub lookups: usize,
    /// Distinct table rows hit.
    pub distinct_rows: usize,
    /// Rows of the table, `R · K^n`.
    pub table_rows: usize,
    /// `distinct_rows / table_rows`.
    pub key_utilization: f64,
    /// Entropy of the hit-row distribution (lookup hit diversity).
    pub hit_entropy: f64,
}

impl HostRouteCodes {
    /// Statistics over the valid positions for which `include(flat_position)` is true.
    pub fn statistics(&self, include: impl Fn(usize) -> bool) -> CodeStatistics {
        let selected: Vec<usize> =
            (0..self.batch * self.tokens).filter(|&position| self.valid[position] && include(position)).collect();
        let routes = self.routes;

        let mut route_entropies = Vec::with_capacity(routes);
        let mut used_symbols = 0usize;
        for route in 0..routes {
            let mut counts = vec![0usize; self.alphabet];
            for &position in &selected {
                counts[self.symbols[position * routes + route] as usize] += 1;
            }
            used_symbols += counts.iter().filter(|&&count| count > 0).count();
            route_entropies.push(entropy(counts.iter().copied()));
        }

        let mut codes: HashMap<&[i64], usize> = HashMap::new();
        for &position in &selected {
            *codes.entry(&self.symbols[position * routes..(position + 1) * routes]).or_default() += 1;
        }

        let orders = self
            .orders
            .iter()
            .zip(&self.addresses)
            .map(|(&order, addresses)| {
                let mut hits: HashMap<i64, usize> = HashMap::new();
                for &position in &selected {
                    for &address in &addresses[position * routes..(position + 1) * routes] {
                        if address >= 0 {
                            *hits.entry(address).or_default() += 1;
                        }
                    }
                }
                let table_rows = routes * self.alphabet.pow(order as u32);
                OrderUtilization {
                    order,
                    lookups: hits.values().sum(),
                    distinct_rows: hits.len(),
                    table_rows,
                    key_utilization: hits.len() as f64 / table_rows as f64,
                    hit_entropy: entropy(hits.values().copied()),
                }
            })
            .collect();

        let routes_f = routes.max(1) as f64;
        CodeStatistics {
            positions: selected.len(),
            route_entropy_mean: route_entropies.iter().sum::<f64>() / routes_f,
            route_entropy_min: route_entropies.iter().copied().fold(f64::INFINITY, f64::min),
            symbol_utilization: used_symbols as f64 / (routes_f * self.alphabet as f64),
            distinct_codes: codes.len(),
            code_entropy: entropy(codes.values().copied()),
            orders,
        }
    }

    /// Statistics per group label (e.g. script id per position, `[B · T]`); padding is skipped.
    pub fn statistics_by_group(&self, groups: &[u32]) -> BTreeMap<u32, CodeStatistics> {
        assert_eq!(groups.len(), self.batch * self.tokens, "one group label per position");
        let labels: HashSet<u32> =
            groups.iter().zip(&self.valid).filter(|(_, valid)| **valid).map(|(group, _)| *group).collect();
        labels.into_iter().map(|label| (label, self.statistics(|position| groups[position] == label))).collect()
    }
}

/// Shannon entropy in bits of a histogram.
fn entropy(counts: impl Iterator<Item = usize> + Clone) -> f64 {
    let total: usize = counts.clone().sum();
    if total == 0 {
        return 0.0;
    }
    counts
        .filter(|&count| count > 0)
        .map(|count| {
            let p = count as f64 / total as f64;
            p * (1.0 / p).log2()
        })
        .sum()
}
