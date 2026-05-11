//! Sum statistics over graph attributes.
use crate::graph::Graph;
use crate::partition::Partition;
use crate::recom::RecomProposal;
use std::collections::HashMap;

/// Parses `val` as an `i32`, accepting whole-number float strings (e.g. `"1044.0"`).
/// Panics with a clear message if the value cannot be represented as an integer.
fn parse_as_int(val: &str, col: &str, n: usize) -> i32 {
    val.parse::<i32>().unwrap_or_else(|_| {
        val.parse::<f64>()
            .ok()
            .filter(|&f| f.fract() == 0.0)
            .map(|f| f as i32)
            .unwrap_or_else(|| {
                panic!(
                    "Could not parse value '{}' as integer for column '{}' at node {}",
                    val, col, n
                )
            })
    })
}

/// Computes sums over all statistics for all districts in a proposal.
pub fn partition_sums(graph: &Graph, partition: &Partition) -> HashMap<String, Vec<i32>> {
    graph
        .attr
        .iter()
        .map(|(key, _)| (key.clone(), partition_attr_sums(graph, partition, key)))
        .collect()
}

/// Computes sums over a single statistic for all districts in a proposal.
pub fn partition_attr_sums(graph: &Graph, partition: &Partition, attr: &str) -> Vec<i32> {
    let values = graph.attr.get(attr).unwrap();
    // TODO: check this invariant elsewhere.
    assert!(values.len() == graph.neighbors.len());
    partition
        .dist_nodes
        .iter()
        .map(|nodes| {
            nodes
                .iter()
                .map(|&n| parse_as_int(&values[n], attr, n))
                .sum()
        })
        .collect()
}

/// Computes sums over statistics for the two new districts in a proposal.
pub fn proposal_sums(graph: &Graph, proposal: &RecomProposal) -> HashMap<String, (i32, i32)> {
    return graph
        .attr
        .iter()
        .map(|(key, values)| {
            let a_sum: i32 = proposal
                .a_nodes
                .iter()
                .map(|&n| parse_as_int(&values[n], key, n))
                .sum();
            let b_sum: i32 = proposal
                .b_nodes
                .iter()
                .map(|&n| parse_as_int(&values[n], key, n))
                .sum();
            return (key.clone(), (a_sum, b_sum));
        })
        .collect();
}
