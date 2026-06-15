//! Load-time graph reordering for the `bendl` file writer.
//!
//! When `--bendl-graph-order` is set, the graph JSON is reordered here, before
//! the chain is built, and the chain then runs in the reordered node space. The
//! writer embeds these exact bytes as the Graph asset and dumps
//! `partition.assignments` verbatim, so the stream is positionally aligned to
//! the embedded graph with no per-step permutation.
//!
//! Note on identity: the crate reorder renumbers each node's `id` to its new
//! position (`0..N-1`), but every other node attribute travels with the node.
//! The embedded graph is therefore self-consistent with the positional stream,
//! and original-node identity is recovered through a preserved identifying
//! attribute (e.g. GEOID), not through `id`. The crate's old-to-new permutation
//! map is discarded rather than embedded.

use std::io::{self, Cursor};

use binary_ensemble::json::graph::{
    sort_json_file_by_key, sort_json_file_by_ordering, GraphOrderingMethod,
};
use serde_json::Value;

/// The ordering applied to the embedded graph, parsed from
/// `--bendl-graph-order`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BendlGraphOrder {
    /// No reordering (default): the embedded graph is the input verbatim.
    None,
    /// Reverse Cuthill-McKee topological ordering.
    Rcm,
    /// Multilevel-cluster topological ordering.
    Mlc,
    /// Sort nodes by the named node attribute.
    Key(String),
}

impl BendlGraphOrder {
    /// Parse the `--bendl-graph-order` value: `none`, `rcm`, `mlc`, or
    /// `key:<attr>`.
    pub fn parse(value: &str) -> Result<BendlGraphOrder, String> {
        match value {
            "none" => Ok(BendlGraphOrder::None),
            "rcm" => Ok(BendlGraphOrder::Rcm),
            "mlc" => Ok(BendlGraphOrder::Mlc),
            other => match other.strip_prefix("key:") {
                Some(attr) if !attr.is_empty() => Ok(BendlGraphOrder::Key(attr.to_string())),
                Some(_) => Err(
                    "--bendl-graph-order 'key:' requires an attribute name, e.g. 'key:GEOID'"
                        .to_string(),
                ),
                None => Err(format!(
                    "invalid --bendl-graph-order '{}': expected one of none, rcm, mlc, key:<attr>",
                    other
                )),
            },
        }
    }

    /// Whether no reordering was requested.
    pub fn is_none(&self) -> bool {
        matches!(self, BendlGraphOrder::None)
    }

    /// The provenance label recorded in the bundle metadata
    /// (`bendl_graph_order`).
    pub fn label(&self) -> String {
        match self {
            BendlGraphOrder::None => "none".to_string(),
            BendlGraphOrder::Rcm => "rcm".to_string(),
            BendlGraphOrder::Mlc => "mlc".to_string(),
            BendlGraphOrder::Key(attr) => format!("key:{}", attr),
        }
    }
}

/// Reorder `graph_bytes` (NetworkX adjacency JSON) per `order` and return the
/// JSON bytes to use for both the chain and the embedded Graph asset.
///
/// `None` returns the input unchanged (so the embedded bytes hash equal to the
/// source). The crate sort functions take `Read`/`Write`, so the input is
/// wrapped in a `Cursor` and the output collected into a `Vec`; their returned
/// old-to-new permutation map is discarded (the embedded reordered graph plus
/// the positional stream are self-consistent on their own).
pub fn reorder_graph_json(graph_bytes: &[u8], order: &BendlGraphOrder) -> io::Result<Vec<u8>> {
    match order {
        BendlGraphOrder::None => Ok(graph_bytes.to_vec()),
        BendlGraphOrder::Rcm => order_by_method(graph_bytes, GraphOrderingMethod::ReverseCuthillMckee),
        BendlGraphOrder::Mlc => order_by_method(graph_bytes, GraphOrderingMethod::MultiLevelCluster),
        BendlGraphOrder::Key(attr) => {
            // The crate sorter tolerates a missing key (sorting absent values as
            // the string "null"), which would silently reorder against bogus
            // values. Pre-validate against FRCW's strict `columns` semantics:
            // the key must be present on every node.
            prevalidate_key(graph_bytes, attr)?;
            let mut reordered_bytes = Vec::new();
            sort_json_file_by_key(Cursor::new(graph_bytes), &mut reordered_bytes, attr)?;
            Ok(reordered_bytes)
        }
    }
}

fn order_by_method(graph_bytes: &[u8], method: GraphOrderingMethod) -> io::Result<Vec<u8>> {
    let mut reordered_bytes = Vec::new();
    sort_json_file_by_ordering(Cursor::new(graph_bytes), &mut reordered_bytes, method)?;
    Ok(reordered_bytes)
}

/// Error unless every node in the graph JSON carries `attr`.
pub fn prevalidate_key(graph_bytes: &[u8], attr: &str) -> io::Result<()> {
    let data: Value = serde_json::from_slice(graph_bytes)?;
    let nodes = data.get("nodes").and_then(Value::as_array).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "graph JSON has no 'nodes' array to reorder",
        )
    })?;
    for (index, node) in nodes.iter().enumerate() {
        let present = node.as_object().is_some_and(|obj| obj.contains_key(attr));
        if !present {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "--bendl-graph-order 'key:{}' requires attribute '{}' on every node, \
                     but node {} is missing it",
                    attr, attr, index
                ),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path_graph_json() -> Vec<u8> {
        serde_json::json!({
            "directed": false,
            "multigraph": false,
            "graph": [],
            "nodes": [
                {"id": 0, "population": 1, "geoid": "c"},
                {"id": 1, "population": 1, "geoid": "a"},
                {"id": 2, "population": 1, "geoid": "b"}
            ],
            "adjacency": [
                [{"id": 1}],
                [{"id": 0}, {"id": 2}],
                [{"id": 1}]
            ]
        })
        .to_string()
        .into_bytes()
    }

    #[test]
    fn parse_recognizes_every_form() {
        assert_eq!(BendlGraphOrder::parse("none").unwrap(), BendlGraphOrder::None);
        assert_eq!(BendlGraphOrder::parse("rcm").unwrap(), BendlGraphOrder::Rcm);
        assert_eq!(BendlGraphOrder::parse("mlc").unwrap(), BendlGraphOrder::Mlc);
        assert_eq!(
            BendlGraphOrder::parse("key:GEOID").unwrap(),
            BendlGraphOrder::Key("GEOID".to_string())
        );
        assert!(BendlGraphOrder::parse("key:").is_err());
        assert!(BendlGraphOrder::parse("bogus").is_err());
    }

    #[test]
    fn label_round_trips_the_provenance_value() {
        assert_eq!(BendlGraphOrder::None.label(), "none");
        assert_eq!(BendlGraphOrder::Rcm.label(), "rcm");
        assert_eq!(BendlGraphOrder::Mlc.label(), "mlc");
        assert_eq!(BendlGraphOrder::Key("X".into()).label(), "key:X");
    }

    #[test]
    fn none_returns_input_verbatim() {
        let bytes = path_graph_json();
        let out = reorder_graph_json(&bytes, &BendlGraphOrder::None).unwrap();
        assert_eq!(out, bytes, "order=none must not re-serialize the graph");
    }

    #[test]
    fn key_sort_orders_by_attribute_and_renumbers_ids() {
        let bytes = path_graph_json();
        let out = reorder_graph_json(&bytes, &BendlGraphOrder::Key("geoid".into())).unwrap();
        let data: Value = serde_json::from_slice(&out).unwrap();
        let nodes = data["nodes"].as_array().unwrap();
        // The node attributes are reordered ascending by geoid: a, b, c.
        let geoids: Vec<&str> = nodes.iter().map(|n| n["geoid"].as_str().unwrap()).collect();
        assert_eq!(geoids, vec!["a", "b", "c"]);
        // The crate renumbers `id` to the new position (0..N-1); it does not
        // carry the original id through. Identity lives in the other attributes.
        let ids: Vec<i64> = nodes.iter().map(|n| n["id"].as_i64().unwrap()).collect();
        assert_eq!(ids, vec![0, 1, 2]);
    }

    #[test]
    fn key_missing_on_a_node_errors_before_sorting() {
        let bytes = serde_json::json!({
            "directed": false,
            "multigraph": false,
            "graph": [],
            "nodes": [
                {"id": 0, "population": 1, "geoid": "a"},
                {"id": 1, "population": 1}
            ],
            "adjacency": [[{"id": 1}], [{"id": 0}]]
        })
        .to_string()
        .into_bytes();
        let err = reorder_graph_json(&bytes, &BendlGraphOrder::Key("geoid".into())).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert!(err.to_string().contains("geoid"));
    }
}
