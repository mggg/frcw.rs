use super::RecomVariant;
use crate::buffers::SubgraphBuffer;
use crate::partition::Partition;
use crate::spanning_tree::SpanningTreeError;
use serde_json::json;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) const MST_DEBUG_DIR_ENV: &str = "RUSTRECOM_MST_DEBUG_DIR";

pub(crate) fn debug_directory() -> Option<PathBuf> {
    std::env::var_os(MST_DEBUG_DIR_ENV)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn variant_name(variant: RecomVariant) -> &'static str {
    match variant {
        RecomVariant::Reversible => "reversible",
        RecomVariant::CutEdgesUST => "cut-edges-ust",
        RecomVariant::DistrictPairsUST => "district-pairs-ust",
        RecomVariant::CutEdgesRMST => "cut-edges-mst",
        RecomVariant::DistrictPairsRMST => "district-pairs-mst",
        RecomVariant::CutEdgesRegionAware => "cut-edges-region-aware",
        RecomVariant::DistrictPairsRegionAware => "district-pairs-region-aware",
    }
}

fn diagnostic_path(directory: &Path, runner: &str, worker_index: usize) -> PathBuf {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    directory.join(format!(
        "rustrecom-mst-failure-{}-{}-worker-{}-{}.json",
        std::process::id(),
        runner,
        worker_index,
        timestamp
    ))
}

#[allow(clippy::too_many_arguments)]
fn write_diagnostic(
    directory: &Path,
    runner: &str,
    worker_index: usize,
    rng_seed: u64,
    variant: RecomVariant,
    error: &SpanningTreeError,
    previous_assignment: Option<&[u32]>,
    partition: &Partition,
    dist_a: usize,
    dist_b: usize,
    subgraph: &SubgraphBuffer,
) -> io::Result<PathBuf> {
    fs::create_dir_all(directory)?;
    let path = diagnostic_path(directory, runner, worker_index);
    let parent_edges = subgraph
        .graph
        .edges
        .iter()
        .map(|edge| [subgraph.raw_nodes[edge.0], subgraph.raw_nodes[edge.1]])
        .collect::<Vec<_>>();
    let parent_adjacency = subgraph
        .graph
        .neighbors
        .iter()
        .map(|neighbors| {
            neighbors
                .iter()
                .map(|&node| subgraph.raw_nodes[node])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let document = json!({
        "schema_version": 1,
        "failure": {
            "error": error,
            "message": error.to_string(),
            "stage": "before spanning-tree split; no candidate partition exists",
        },
        "runner": runner,
        "worker_index": worker_index,
        "rng_seed": rng_seed,
        "recom_variant": variant_name(variant),
        "attempted_districts": {
            "a": dist_a,
            "b": dist_b,
            "indexing": "internal zero-based",
        },
        "previous_partition": previous_assignment.map(|assignments| {
            json!({ "assignments": assignments })
        }),
        "current_partition": {
            "assignments": &partition.assignments,
        },
        "candidate_partition": null,
        "merged_subgraph": {
            "node_count": subgraph.graph.pops.len(),
            "edge_count": subgraph.graph.edges.len(),
            "parent_graph_node_indices": &subgraph.raw_nodes,
            "parent_graph_edges": parent_edges,
            "parent_graph_adjacency": parent_adjacency,
        },
    });
    fs::write(&path, serde_json::to_vec_pretty(&document)?)?;
    Ok(path)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn failure_message(
    debug_directory: Option<&Path>,
    runner: &str,
    worker_index: usize,
    rng_seed: u64,
    variant: RecomVariant,
    error: &SpanningTreeError,
    previous_assignment: Option<&[u32]>,
    partition: &Partition,
    dist_a: usize,
    dist_b: usize,
    subgraph: &SubgraphBuffer,
) -> String {
    let diagnostic = match debug_directory {
        Some(directory) => match write_diagnostic(
            directory,
            runner,
            worker_index,
            rng_seed,
            variant,
            error,
            previous_assignment,
            partition,
            dist_a,
            dist_b,
            subgraph,
        ) {
            Ok(path) => format!(" Diagnostic written to '{}'.", path.display()),
            Err(write_error) => format!(
                " Could not write the requested diagnostic to '{}': {}.",
                directory.display(),
                write_error
            ),
        },
        None => format!(
            " Set {} to a directory to write a diagnostic JSON file.",
            MST_DEBUG_DIR_ENV
        ),
    };
    format!("{}{}", error, diagnostic)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn diagnostic_includes_previous_and_current_partitions() {
        let graph = Graph::rect_grid(2, 2);
        let partition = Partition::from_assignments(&graph, &vec![0, 0, 1, 1]).unwrap();
        let mut subgraph = SubgraphBuffer::new(4, 4);
        partition.subgraph(&graph, &mut subgraph, 0, 1);
        let previous = vec![0, 1, 0, 1];
        let error = SpanningTreeError::Disconnected {
            node_count: 4,
            graph_edge_count: 2,
            expected_tree_edges: 3,
            actual_tree_edges: 2,
        };
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "rustrecom_mst_diagnostic_unit_{}_{}",
            std::process::id(),
            timestamp
        ));

        let message = failure_message(
            Some(&directory),
            "test",
            2,
            99,
            RecomVariant::CutEdgesRMST,
            &error,
            Some(&previous),
            &partition,
            0,
            1,
            &subgraph,
        );
        assert!(message.contains("Diagnostic written to"));
        let paths = fs::read_dir(&directory)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect::<Vec<_>>();
        assert_eq!(paths.len(), 1);
        let document: serde_json::Value =
            serde_json::from_slice(&fs::read(&paths[0]).unwrap()).unwrap();
        assert_eq!(
            document["previous_partition"]["assignments"],
            json!(previous)
        );
        assert_eq!(
            document["current_partition"]["assignments"],
            json!([0, 0, 1, 1])
        );

        fs::remove_file(&paths[0]).unwrap();
        fs::remove_dir(directory).unwrap();
    }
}
