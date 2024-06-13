use crate::graph::Graph;
use crate::partition::Partition;
use crate::recom::RecomProposal;
#[cfg(feature = "linalg")]
use crate::stats::subgraph_spanning_tree_count;
use crate::stats::{partition_sums, proposal_sums, SelfLoopCounts, SelfLoopReason};
use pcompress::diff::Diff;
use pcompress::encode::export_diff;
use serde_json::{json, to_value, Value};
use std::io::{BufWriter, Result, Write};

/// A standard interface for writing steps and statistics to stdout.
/// TODO: allow direct output to a file (e.g. in Parquet format).
/// TODO: move outside of this module.
pub trait StatsWriter: Send {
    /// Prints data from the initial partition.
    fn init(&mut self, graph: &Graph, partition: &Partition) -> Result<()>;

    /// Prints deltas generated from an accepted proposal
    /// which has been applied to `partition`.
    fn step(
        &mut self,
        step: u64,
        graph: &Graph,
        partition: &Partition,
        proposal: &RecomProposal,
        counts: &SelfLoopCounts,
    ) -> Result<()>;

    /// Cleans up after the last step (useful for testing).
    fn close(&mut self) -> Result<()>;
}

/// Writes chain statistics in TSV (tab-separated values) format.
/// Each step in the chain is a line; no statistics are saved about the
/// initial partition.
///
/// Rows in the output contain the following columns:
///   * `step` - The step count at the accepted proposal (including self-loops).
///   * `non_adjacent` - The number of self-loops due to non-adjacency.
///   * `no_split` - The number of self-loops due to the lack of an ε-balanced split.
///   * `seam_length` - The number of self-loops due to seam length rejection
///     (Reversible ReCom only).
///   * `a_label` - The label of the `a`-district in the proposal.
///   * `b_label` - The label of the `b`-district in the proposal.
///   * `a_pop` - The population of the new `a`-district.
///   * `b_pop` - The population of the new `b`-district.
///   * `a_nodes` - The list of node indices in the new `a`-district.
///   * `b_nodes` - The list of node indices in the new `b`-district.
pub struct TSVWriter {
    // The output stream that we would like to write to.
    output: Box<dyn Write + Send>,
}

/// Writes assignments in space-delimited format (with step number prefix).
pub struct AssignmentsOnlyWriter {
    /// Determines whether to canonicalize assignment vectors.
    canonicalize: bool,
    /// The output stream that we would like to write to.
    output: Box<dyn Write + Send>,
}

/// Writes out assignments in sandardizable format for generic
/// ensembles of plans. Will produce a JSONL file with the line
/// formatting:
///
/// ```json
/// {"assignment": <assignment-vector>, "sample": <sample-number>}
/// ```
///
/// where the sample number is indexed from 1. This will also
/// create a copy JSONL line for each step in a self loop
/// to ensure that we can properly collect statistical data.
pub struct CanonicalWriter {
    /// The previous assignment vector. Used to fill in self loops.
    previous_assignment: Vec<u32>,
    /// The output stream that we would like to write to.
    output: Box<dyn Write + Send>,
}

pub struct BenWriter {
    previous_assignment: Vec<u32>,
    output: Box<dyn Write + Send>,
}

/// Writes assignments in Max Fan's `pcompress` binary format.
pub struct PcompressWriter {
    /// A buffered writer used internally by pcompress.
    writer: BufWriter<Box<dyn Write + Send>>,
    /// Diff buffer (reused across steps).
    diff: Diff,
}

/// Writes statistics in JSONL (JSON Lines) format.
pub struct JSONLWriter {
    /// Determines whether node deltas should be saved for each step.
    nodes: bool,
    /// Determines whether to compute spanning tree counts for each step.
    spanning_tree_counts: bool,
    /// Determines whether to compute cut edge counts for each step.
    cut_edges_count: bool,
    // The output stream that we would like to write to.
    output: Box<dyn Write + Send>,
}

impl TSVWriter {
    pub fn new(output: Box<dyn Write + Send>) -> TSVWriter {
        TSVWriter { output: output }
    }
}

impl AssignmentsOnlyWriter {
    pub fn new(canonicalize: bool, output: Box<dyn Write + Send>) -> AssignmentsOnlyWriter {
        AssignmentsOnlyWriter {
            output: output,
            canonicalize: canonicalize,
        }
    }

    /// Canonicalizes the assignment vector.
    /// Meaning that instead of generating an assigmment
    /// vector of the form
    /// [2, 2, 2, 2, 3, 0, 0, 0, 3, 3, 1, 1, 3, 1, 1, 1]
    /// the values are reassigned by order of appearance
    /// indexed by 1, so the previous vector would become
    /// [1, 1, 1, 1, 2, 3, 3, 3, 2, 2, 4, 4, 2, 4, 4, 4]
    fn canonicalize_assignments(&self, partition: &Partition) -> Vec<u32> {
        let mut canon = partition.assignments.clone();
        let mut dist_mapping = vec![0; partition.num_dists as usize];
        let mut cur_dist = 1;
        for (idx, &assn) in partition.assignments.iter().enumerate() {
            if dist_mapping[assn as usize] == 0 {
                dist_mapping[assn as usize] = cur_dist;
                cur_dist += 1;
            }
            canon[idx] = dist_mapping[assn as usize];
        }
        canon
    }
}

impl CanonicalWriter {
    pub fn new(output: Box<dyn Write + Send>) -> CanonicalWriter {
        CanonicalWriter {
            previous_assignment: Vec::new(),
            output: output,
        }
    }
}

impl BenWriter {
    pub fn new(output: Box<dyn Write + Send>) -> BenWriter {
        BenWriter {
            previous_assignment: Vec::new(),
            output: output,
        }
    }
}

impl JSONLWriter {
    pub fn new(
        nodes: bool,
        spanning_tree_counts: bool,
        cut_edges_count: bool,
        output: Box<dyn Write + Send>,
    ) -> JSONLWriter {
        JSONLWriter {
            nodes: nodes,
            spanning_tree_counts: spanning_tree_counts,
            cut_edges_count: cut_edges_count,
            output: output,
        }
    }

    #[cfg(feature = "linalg")]
    /// Adds initial spanning tree count statistics to `stats`.
    fn init_spanning_tree_counts(graph: &Graph, partition: &Partition, stats: &mut Value) {
        stats.as_object_mut().unwrap().insert(
            "spanning_tree_counts".to_string(),
            partition
                .dist_nodes
                .iter()
                .map(|nodes| subgraph_spanning_tree_count(graph, nodes))
                .collect(),
        );
    }

    #[cfg(not(feature = "linalg"))]
    /// Dummy function---spanning tree counts depend on linear algebra libraries.
    fn init_spanning_tree_counts(_graph: &Graph, _partition: &Partition, _stats: &mut Value) {}

    #[cfg(feature = "linalg")]
    /// Adds step spanning tree count statistics to `stats`.
    fn step_spanning_tree_counts(graph: &Graph, proposal: &RecomProposal, stats: &mut Value) {
        stats.as_object_mut().unwrap().insert(
            "spanning_tree_counts".to_string(),
            json!((
                subgraph_spanning_tree_count(graph, &proposal.a_nodes),
                subgraph_spanning_tree_count(graph, &proposal.b_nodes)
            )),
        );
    }

    #[cfg(not(feature = "linalg"))]
    /// Dummy function---spanning tree counts depend on linear algebra libraries.
    fn step_spanning_tree_counts(_graph: &Graph, _proposal: &RecomProposal, _stats: &mut Value) {}
}

impl StatsWriter for TSVWriter {
    fn init(&mut self, _graph: &Graph, _partition: &Partition) -> Result<()> {
        // TSV column header.
        print!("step\tnon_adjacent\tno_split\tseam_length\ta_label\tb_label\t");
        println!("a_pop\tb_pop\ta_nodes\tb_nodes");
        Ok(())
    }

    fn step(
        &mut self,
        step: u64,
        _graph: &Graph,
        _partition: &Partition,
        proposal: &RecomProposal,
        counts: &SelfLoopCounts,
    ) -> Result<()> {
        self.output
            .write_all(
                format!(
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:?}\t{:?}\n",
                    step,
                    counts.get(SelfLoopReason::NonAdjacent),
                    counts.get(SelfLoopReason::NoSplit),
                    counts.get(SelfLoopReason::SeamLength),
                    proposal.a_label,
                    proposal.b_label,
                    proposal.a_pop,
                    proposal.b_pop,
                    proposal.a_nodes,
                    proposal.b_nodes
                )
                .as_bytes(),
            )
            .expect("Failed to write to output");
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

impl StatsWriter for JSONLWriter {
    fn init(&mut self, graph: &Graph, partition: &Partition) -> Result<()> {
        // TSV column header.
        let mut stats = json!({
            "num_dists": partition.num_dists,
            "populations": partition.dist_pops,
            "sums": partition_sums(graph, partition)
        });
        if self.spanning_tree_counts {
            JSONLWriter::init_spanning_tree_counts(graph, partition, &mut stats);
        }
        if self.cut_edges_count {
            let mut partition = partition.clone();
            let cut_edges_count = partition.cut_edges(graph).len();
            stats.as_object_mut().unwrap().insert(
                "num_cut_edges".to_string(),
                to_value(cut_edges_count).unwrap(),
            );
        }
        self.output
            .write_all(format!("{}\n", json!({ "init": stats }).to_string()).as_bytes())
            .expect("Failed to write to output");
        Ok(())
    }

    fn step(
        &mut self,
        step: u64,
        graph: &Graph,
        partition: &Partition,
        proposal: &RecomProposal,
        counts: &SelfLoopCounts,
    ) -> Result<()> {
        let mut step = json!({
            "step": step,
            "dists": (proposal.a_label, proposal.b_label),
            "populations": (proposal.a_pop, proposal.b_pop),
            "sums": proposal_sums(graph, proposal),
            "counts": counts,
        });
        if self.nodes {
            step.as_object_mut().unwrap().insert(
                "nodes".to_string(),
                json!((proposal.a_nodes.clone(), proposal.b_nodes.clone())),
            );
        }
        if self.spanning_tree_counts {
            JSONLWriter::step_spanning_tree_counts(graph, proposal, &mut step);
        }
        if self.cut_edges_count {
            let mut partition = partition.clone();
            let cut_edges_count = partition.cut_edges(graph).len();
            step.as_object_mut().unwrap().insert(
                "num_cut_edges".to_string(),
                to_value(cut_edges_count).unwrap(),
            );
        }
        self.output
            .write_all(format!("{}\n", json!({ "step": step }).to_string()).as_bytes())
            .expect("Failed to write to output");
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

impl StatsWriter for AssignmentsOnlyWriter {
    fn init(&mut self, _graph: &Graph, partition: &Partition) -> Result<()> {
        if self.canonicalize {
            self.output
                .write_all(format!("0,{:?}\n", self.canonicalize_assignments(partition)).as_bytes())
                .expect("Failed to write to output");
        } else {
            self.output
                .write_all(format!("0,{:?}\n", partition.assignments).as_bytes())
                .expect("Failed to write to output");
        }
        Ok(())
    }

    fn step(
        &mut self,
        step: u64,
        _graph: &Graph,
        partition: &Partition,
        _proposal: &RecomProposal,
        _counts: &SelfLoopCounts,
    ) -> Result<()> {
        if self.canonicalize {
            self.output
                .write_all(
                    format!("{},{:?}\n", step, self.canonicalize_assignments(partition)).as_bytes(),
                )
                .expect("Failed to write to output");
        } else {
            self.output
                .write_all(format!("{},{:?}\n", step, partition.assignments).as_bytes())
                .expect("Failed to write to output");
        }
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

impl StatsWriter for CanonicalWriter {
    fn init(&mut self, _graph: &Graph, partition: &Partition) -> Result<()> {
        self.previous_assignment = partition
            .assignments
            .clone()
            .iter()
            .map(|x| x + 1)
            .collect();
        self.output
            .write_all(
                format!(
                    "{}\n",
                    json!({
                        "assignment": self.previous_assignment,
                        "sample": 1,
                    })
                )
                .as_bytes(),
            )
            .expect("Failed to write to output");
        Ok(())
    }

    fn step(
        &mut self,
        step: u64,
        _graph: &Graph,
        partition: &Partition,
        _proposal: &RecomProposal,
        counts: &SelfLoopCounts,
    ) -> Result<()> {
        let tot_count = counts.sum();
        for i in step - tot_count as u64 + 1..step + 1 {
            self.output
                .write_all(
                    format!(
                        "{}\n",
                        json!({
                            "assignment": self.previous_assignment,
                            "sample": i,
                        })
                    )
                    .as_bytes(),
                )
                .expect("Failed to write to output");
        }
        self.previous_assignment = partition
            .assignments
            .clone()
            .iter()
            .map(|x| x + 1)
            .collect();
        self.output
            .write_all(
                format!(
                    "{}\n",
                    json!({
                        "assignment": self.previous_assignment,
                        "sample": step+1,
                    })
                )
                .as_bytes(),
            )
            .expect("Failed to write to output");
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

impl StatsWriter for BenWriter {
    fn init(&mut self, _graph: &Graph, partition: &Partition) -> Result<()> {
        self.previous_assignment = partition
            .assignments
            .clone()
            .iter()
            .map(|x| x + 1)
            .collect();
        self.output
            .write_all(b"MKVCHAIN BEN FILE")
            .expect("Failed to write to output");
        self.output
            .write_all(
                ben::encode::encode_ben_vec_from_assign(
                    (&self.previous_assignment)
                        .iter()
                        .map(|&x| x as u16)
                        .collect(),
                )
                .as_slice(),
            )
            .expect("Failed to write to output");
        Ok(())
    }

    fn step(
        &mut self,
        _step: u64,
        _graph: &Graph,
        partition: &Partition,
        _proposal: &RecomProposal,
        counts: &SelfLoopCounts,
    ) -> Result<()> {
        // The first step plus the number of self loops
        let tot_count = counts.sum() + 1;
        self.output.write_all(&(tot_count as u16).to_be_bytes())?;
        self.previous_assignment = partition
            .assignments
            .clone()
            .iter()
            .map(|x| x + 1)
            .collect();
        let new_vec = ben::encode::encode_ben_vec_from_assign(
            (&self.previous_assignment)
                .iter()
                .map(|&x| x as u16)
                .collect(),
        );
        self.output
            .write_all(new_vec.as_slice())
            .expect("Failed to write to output");
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        // The very last step is always counted as 1 since we hit that
        // step and then we stop drawing.
        self.output.write_all(&[0u8, 1u8])?;
        Ok(())
    }
}

impl PcompressWriter {
    pub fn new(output: Box<dyn Write + Send>) -> PcompressWriter {
        PcompressWriter {
            writer: BufWriter::new(output),
            diff: Diff::new(),
        }
    }
}

impl StatsWriter for PcompressWriter {
    fn init(&mut self, _graph: &Graph, partition: &Partition) -> Result<()> {
        for (node, &dist) in partition.assignments.iter().enumerate() {
            self.diff.add(dist as usize, node);
        }
        export_diff(&mut self.writer, &self.diff);
        Ok(())
    }

    fn step(
        &mut self,
        _step: u64,
        _graph: &Graph,
        _partition: &Partition,
        proposal: &RecomProposal,
        counts: &SelfLoopCounts,
    ) -> Result<()> {
        // Write out self-loops first. The counts here
        // are the number of self-loops since the last
        // accepted proposal (i.e. the number of times the
        // last proposal was repeated before acceptance).
        self.diff.reset();
        for _ in 0..counts.sum() {
            export_diff(&mut self.writer, &self.diff);
        }

        // Write out the actual delta.
        self.diff.reset();
        for &node in proposal.a_nodes.iter() {
            self.diff.add(proposal.a_label, node);
        }
        for &node in proposal.b_nodes.iter() {
            self.diff.add(proposal.b_label, node);
        }
        export_diff(&mut self.writer, &self.diff);

        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        self.writer.flush()
    }
}
