//! Utility functions for loading graph and partition data.
use crate::graph::{Edge, Graph};
use crate::partition::Partition;
use serde_json::Result as SerdeResult;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io;

/// Returns `(district_label, components, node_count)` for each disconnected district.
fn disconnected_districts(graph: &Graph, partition: &Partition) -> Vec<(usize, usize, usize)> {
    let mut disconnected = Vec::<(usize, usize, usize)>::new();
    for (dist_idx, nodes) in partition.dist_nodes.iter().enumerate() {
        if nodes.len() <= 1 {
            continue;
        }
        let node_set = nodes.iter().copied().collect::<HashSet<usize>>();
        let mut visited = HashSet::<usize>::with_capacity(nodes.len());
        let mut stack = Vec::<usize>::with_capacity(nodes.len());
        let mut components = 0;
        for &start in nodes.iter() {
            if visited.contains(&start) {
                continue;
            }
            components += 1;
            visited.insert(start);
            stack.push(start);
            while let Some(node) = stack.pop() {
                for &neighbor in graph.neighbors[node].iter() {
                    if node_set.contains(&neighbor) && !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        stack.push(neighbor);
                    }
                }
            }
        }
        if components > 1 {
            disconnected.push((dist_idx + 1, components, nodes.len()));
        }
    }
    disconnected
}

/// Loads graph and partition data in the NetworkX `adjacency_data` format
/// used by [GerryChain](https://github.com/mggg/gerrychain). Returns a
/// [serde_json::Result] containing a [graph::Graph] and
/// a [partition::Partition] upon a successful load.
///
/// # Arguments
///
/// * `path` - the path of the graph JSON file.
/// * `pop_col` - The column in the graph JSON corresponding to total node
///    population. This column should be integer-valued.
/// * `assignment_col` - A column in the graph JSON corresponding to a
///    a seed partition. This column should be integer-valued and 1-indexed.
/// * `columns` - Node metadata columns that must exist on every node. Missing
///    keys panic -- use this for columns a typo in whose name should be caught
///    at load time.
/// * `partial_columns` - Node metadata columns that may be missing on some
///    nodes. Missing keys are stored as the string `"null"`, matching the
///    serialization of an explicit JSON `null`. Use this for columns that
///    legitimately apply to only a subset of nodes (e.g. `boundary_perim`
///    on boundary nodes only).
/// * `edge_float_cols` - Edge attribute columns to load as `f64` (e.g. `"shared_perim"`).
///    Pass an empty `Vec` if no edge attributes are needed.
pub fn from_networkx(
    path: &str,
    pop_col: &str,
    assignment_col: &str,
    columns: Vec<String>,
    partial_columns: Vec<String>,
    edge_float_cols: Vec<String>,
) -> SerdeResult<(Graph, Partition)> {
    let (graph, data) =
        match graph_from_networkx(path, pop_col, columns, partial_columns, edge_float_cols) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };

    let raw_nodes = data["nodes"].as_array().expect("Could not get nodes array");
    let assignments: Vec<u32> = raw_nodes
        .iter()
        .enumerate()
        .map(|(i, node)| match &node[assignment_col] {
            serde_json::Value::Number(num) => num.as_u64().expect(
                format!(
                    "When geting assignment, failed to unwrap the value {} as a u32",
                    num
                )
                .as_str(),
            ) as u32,
            serde_json::Value::String(ref s) => s.parse::<u32>().expect(
                format!(
                    "When getting assignment, failed to unwrap the value {} as a u32",
                    s
                )
                .as_str(),
            ),
            _ => panic!(
                "{}{}{}",
                "Unexpected entry type in assignment column. ",
                format!(
                    "Found {:?} at index {:?} in column {:?}.",
                    node[assignment_col], i, assignment_col
                ),
                "Please make sure that all entries can be interpreted as positive numbers."
            ),
        })
        .collect();
    let partition = Partition::from_assignments(&graph, &assignments).map_err(|e| {
        serde_json::Error::io(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Could not create partition from assignment column '{}': {}",
                assignment_col, e
            ),
        ))
    })?;

    let disconnected = disconnected_districts(&graph, &partition);
    if !disconnected.is_empty() {
        let max_items = 8;
        let preview = disconnected
            .iter()
            .take(max_items)
            .map(|(label, components, node_count)| {
                format!(
                    "district {} ({} components, {} nodes)",
                    label, components, node_count
                )
            })
            .collect::<Vec<String>>()
            .join(", ");
        let more = if disconnected.len() > max_items {
            format!(", and {} more", disconnected.len() - max_items)
        } else {
            "".to_string()
        };
        return Err(serde_json::Error::io(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Assignment column '{}' is disconnected in {} district(s): {}{}.",
                assignment_col,
                disconnected.len(),
                preview,
                more
            ),
        )));
    }

    return Ok((graph, partition));
}

/// Loads graph data in the NetworkX `adjacency_data` format
/// used by [GerryChain](https://github.com/mggg/gerrychain). Returns a
/// [serde_json::Result] containing a [graph::Graph] and the raw
/// graph JSON tree upon a successful load.
///
/// # Arguments
///
/// * `path` - the path of the graph JSON file.
/// * `pop_col` - The column in the graph JSON corresponding to total node
///    population. This column should be integer-valued.
/// * `columns` - Node metadata columns that must exist on every node.
///    Missing keys panic.
/// * `partial_columns` - Node metadata columns that may be missing on some
///    nodes. Missing keys are stored as the string `"null"`.
/// * `edge_float_cols` - Edge attribute columns to load as `f64` (e.g. `"shared_perim"`).
///    Pass an empty `Vec` if no edge attributes are needed.
pub fn graph_from_networkx(
    path: &str,
    pop_col: &str,
    columns: Vec<String>,
    partial_columns: Vec<String>,
    edge_float_cols: Vec<String>,
) -> SerdeResult<(Graph, Value)> {
    // TODO: should load from a generic buffer.
    let raw = fs::read_to_string(path).expect("Could not load graph");
    let data: Value = serde_json::from_str(&raw)?;

    let raw_nodes = data["nodes"].as_array().unwrap();
    let raw_adj = data["adjacency"].as_array().unwrap();
    let num_nodes = raw_nodes.len();
    let mut node_id_to_index = HashMap::<String, usize>::with_capacity(num_nodes);
    for (index, node) in raw_nodes.iter().enumerate() {
        let id = node
            .as_object()
            .and_then(|obj| obj.get("id"))
            .unwrap_or_else(|| panic!("Node {} is missing an 'id' field.", index));
        let key = serde_json::to_string(id).expect("Could not serialize node id.");
        if node_id_to_index.insert(key.clone(), index).is_some() {
            panic!("Duplicate node id in graph JSON: {}", key);
        }
    }

    let mut pops = Vec::<u32>::with_capacity(num_nodes);
    let mut neighbors = Vec::<Vec<usize>>::with_capacity(num_nodes);
    let mut edges = Vec::<Edge>::new();
    let mut edges_start = vec![0 as usize; num_nodes];
    let mut attr = HashMap::new();
    for col in columns.iter().chain(partial_columns.iter()) {
        attr.insert(col.clone(), Vec::<String>::with_capacity(num_nodes));
    }

    for (index, (node, adj)) in raw_nodes.iter().zip(raw_adj.iter()).enumerate() {
        edges_start[index] = edges.len();
        let node_neighbors: Vec<usize> = adj
            .as_array()
            .unwrap()
            .into_iter()
            .map(|n| {
                let neighbor_id =
                    n.as_object()
                        .and_then(|obj| obj.get("id"))
                        .unwrap_or_else(|| {
                            panic!("Node {} has an adjacency entry without an 'id'.", index)
                        });
                let neighbor_key =
                    serde_json::to_string(neighbor_id).expect("Could not serialize neighbor id.");
                *node_id_to_index.get(&neighbor_key).unwrap_or_else(|| {
                    panic!(
                        "Node {} has adjacency to unknown node id {}.",
                        index, neighbor_key
                    )
                })
            })
            .collect();
        for col in columns.iter() {
            if let Some(data) = attr.get_mut(col) {
                match node.get(col) {
                    Some(value) => data.push(
                        value.as_str().map(str::to_owned).unwrap_or_else(|| value.to_string())
                    ),
                    None => panic!(
                        "Node {} is missing required attribute '{}'. \
                         If this column applies only to some nodes, pass it \
                         as a partial column instead.",
                        index, col
                    ),
                }
            }
        }
        for col in partial_columns.iter() {
            if let Some(data) = attr.get_mut(col) {
                match node.get(col) {
                    Some(value) => data.push(
                        value.as_str().map(str::to_owned).unwrap_or_else(|| value.to_string())
                    ),
                    None => data.push("null".to_string()),
                }
            }
        }
        let new_pop = match &node[pop_col] {
            serde_json::Value::Number(num) => num.as_f64().expect(
                format!(
                    "When getting population, failed to unwrap the value {} as a f64",
                    num
                )
                .as_str(),
            ) as u32,
            serde_json::Value::String(ref s) => s.parse::<u32>().unwrap(),
            _ => panic!(
                "{}{}{}",
                "Unexpected entry type in population column. ",
                format!(
                    "Found {:?} at index {:?} in column {:?}.",
                    node[pop_col], index, pop_col
                ),
                "Please make sure that all entries can be interpreted as positive numbers."
            ),
        };
        pops.push(new_pop);
        neighbors.push(node_neighbors.clone());

        for neighbor in &node_neighbors {
            if neighbor > &index {
                let edge = Edge(index, *neighbor);
                edges.push(edge.clone());
            }
        }
    }

    // Load float-valued edge attributes (e.g. shared_perim).
    // For each edge Edge(u, v) (u < v), find the adjacency entry in raw_adj[u]
    // that points to v and extract the requested float columns.
    let mut edge_attr: HashMap<String, Vec<f64>> = HashMap::new();
    if !edge_float_cols.is_empty() {
        for col in edge_float_cols.iter() {
            let col_vals: Vec<f64> = edges
                .iter()
                .map(|Edge(u, v)| {
                    raw_adj[*u]
                        .as_array()
                        .unwrap()
                        .iter()
                        .find(|entry| {
                            let nid = serde_json::to_string(
                                entry.as_object().unwrap().get("id").unwrap(),
                            )
                            .unwrap();
                            node_id_to_index.get(&nid).copied() == Some(*v)
                        })
                        .and_then(|entry| entry[col.as_str()].as_f64())
                        .unwrap_or(0.0)
                })
                .collect();
            edge_attr.insert(col.clone(), col_vals);
        }
    }

    let total_pop = pops.iter().sum();
    let graph = Graph {
        pops: pops,
        neighbors: neighbors,
        edges: edges.clone(),
        edges_start: edges_start.clone(),
        total_pop: total_pop,
        attr: attr,
        edge_attr: edge_attr,
        int_attr: std::collections::HashMap::new(),
    };
    return Ok((graph, data));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn write_temp_graph(data: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        path.push(format!("frcw_init_test_{}_{}.json", std::process::id(), ts));
        fs::write(&path, data).unwrap();
        path
    }

    fn connected_component_count(graph: &Graph) -> usize {
        let mut visited = vec![false; graph.neighbors.len()];
        let mut count = 0;
        for start in 0..graph.neighbors.len() {
            if visited[start] {
                continue;
            }
            count += 1;
            let mut stack = vec![start];
            visited[start] = true;
            while let Some(node) = stack.pop() {
                for &neighbor in graph.neighbors[node].iter() {
                    if !visited[neighbor] {
                        visited[neighbor] = true;
                        stack.push(neighbor);
                    }
                }
            }
        }
        count
    }

    #[test]
    fn graph_from_networkx_one_indexed_ids() {
        let json = serde_json::json!({
            "directed": false,
            "multigraph": false,
            "graph": [],
            "nodes": [
                {"id": 1, "population": 1},
                {"id": 2, "population": 1},
                {"id": 3, "population": 1},
                {"id": 4, "population": 1}
            ],
            "adjacency": [
                [{"id": 2}],
                [{"id": 1}, {"id": 3}],
                [{"id": 2}, {"id": 4}],
                [{"id": 3}]
            ]
        });
        let path = write_temp_graph(&json.to_string());
        let path_str = path.to_string_lossy();
        let (graph, _) =
            graph_from_networkx(&path_str, "population", vec![], vec![], vec![]).unwrap();
        fs::remove_file(path).unwrap();

        assert_eq!(
            graph.neighbors,
            vec![vec![1], vec![0, 2], vec![1, 3], vec![2]]
        );
        assert_eq!(graph.edges, vec![Edge(0, 1), Edge(1, 2), Edge(2, 3)]);
        assert_eq!(connected_component_count(&graph), 1);
    }

    #[test]
    fn graph_from_networkx_shuffled_node_ids() {
        let json = serde_json::json!({
            "directed": false,
            "multigraph": false,
            "graph": [],
            "nodes": [
                {"id": 10, "population": 1},
                {"id": 30, "population": 1},
                {"id": 20, "population": 1},
                {"id": 40, "population": 1}
            ],
            "adjacency": [
                [{"id": 20}],
                [{"id": 20}, {"id": 40}],
                [{"id": 10}, {"id": 30}],
                [{"id": 30}]
            ]
        });
        let path = write_temp_graph(&json.to_string());
        let path_str = path.to_string_lossy();
        let (graph, _) =
            graph_from_networkx(&path_str, "population", vec![], vec![], vec![]).unwrap();
        fs::remove_file(path).unwrap();

        assert_eq!(
            graph.neighbors,
            vec![vec![2], vec![2, 3], vec![0, 1], vec![1]]
        );
        assert_eq!(graph.edges, vec![Edge(0, 2), Edge(1, 2), Edge(1, 3)]);
        assert_eq!(connected_component_count(&graph), 1);
    }

    #[test]
    fn from_networkx_rejects_disconnected_seed_partition() {
        let json = serde_json::json!({
            "directed": false,
            "multigraph": false,
            "graph": [],
            "nodes": [
                {"id": 0, "population": 1, "district": 1},
                {"id": 1, "population": 1, "district": 2},
                {"id": 2, "population": 1, "district": 1},
                {"id": 3, "population": 1, "district": 2}
            ],
            "adjacency": [
                [{"id": 1}],
                [{"id": 0}, {"id": 2}],
                [{"id": 1}, {"id": 3}],
                [{"id": 2}]
            ]
        });
        let path = write_temp_graph(&json.to_string());
        let path_str = path.to_string_lossy();
        let err =
            from_networkx(&path_str, "population", "district", vec![], vec![], vec![]).unwrap_err();
        fs::remove_file(path).unwrap();

        let msg = err.to_string();
        assert!(msg.contains("disconnected"));
        assert!(msg.contains("Assignment column 'district'"));
        assert!(msg.contains("district 1"));
    }
}
