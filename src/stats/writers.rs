use crate::graph::Graph;
use crate::partition::Partition;
use crate::recom::RecomProposal;
#[cfg(feature = "linalg")]
use crate::stats::subgraph_spanning_tree_count;
use crate::stats::{partition_sums, proposal_sums, SelfLoopCounts, SelfLoopReason};
use binary_ensemble::io::bundle::format::{AssignmentFormat, KnownAssetKind};
use binary_ensemble::io::bundle::{
    AddAssetOptions, BendlStreamSession, BendlWriteError, BendlWriter,
};
use binary_ensemble::io::writer::BenStreamWriter;
use binary_ensemble::BenVariant;
use pcompress::diff::Diff;
use pcompress::encode::export_diff;
use serde_json::{json, to_value, Value};
use std::fs::File;
use std::io::{self, BufWriter, Result, Write};

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

    /// Prints self-loops after the last accepted proposal.
    fn self_loop(
        &mut self,
        _step: u64,
        _graph: &Graph,
        _partition: &Partition,
        _counts: &SelfLoopCounts,
    ) -> Result<()> {
        Ok(())
    }

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
    /// The last assignment vector written.
    previous_assignment: Vec<u32>,
    /// The last chain step written.
    last_step: u64,
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

/// Shared TwoDelta frame bookkeeping for the `ben` and `bendl` writers.
///
/// Holds the single buffered plan whose repeat count is not yet known and the
/// trailing self-loop tally at the final plan, and accumulates the true total
/// sample count. Both writers delegate their frame emission here so the count
/// logic (including the `u16`-overflow split below) lives in one place.
struct TwoDeltaFrameBuffer {
    /// The plan whose frame is buffered, held as the `u16` labels BEN packs. Its
    /// repeat count is only known once the chain moves off it (the next `step`)
    /// or the chain ends (the final flush).
    pending_assignment: Vec<u16>,
    /// Self-loop count accumulated at the final plan after the last accepted
    /// step, folded into the final segment's count by the final flush.
    trailing_self_loops: u64,
    /// Running sum of every emitted frame count: the true total sample count,
    /// stamped into the BENDL header so readers can trust it.
    total_samples: u64,
}

impl TwoDeltaFrameBuffer {
    fn new() -> TwoDeltaFrameBuffer {
        TwoDeltaFrameBuffer {
            pending_assignment: Vec::new(),
            trailing_self_loops: 0,
            total_samples: 0,
        }
    }

    /// Buffer the initial plan (from `init`); its count is filled later.
    fn set_initial(&mut self, assignment: Vec<u16>) {
        self.pending_assignment = assignment;
    }

    /// The chain left the pending plan via an accepted step: emit the pending
    /// plan with count `self_loops + 1` (the rejections spent there plus the
    /// move to `next`), then buffer `next`.
    fn push_step<W: Write>(
        &mut self,
        writer: &mut BenStreamWriter<W>,
        next_assignment: Vec<u16>,
        self_loops: u64,
    ) -> Result<()> {
        let count = self_loops + 1;
        let pending_assignment = std::mem::replace(&mut self.pending_assignment, next_assignment);
        self.write_counted(writer, pending_assignment, count)
    }

    /// Accumulate trailing self-loops sampled at the final plan.
    fn add_self_loops(&mut self, count: u64) {
        self.trailing_self_loops += count;
    }

    /// Emit the final pending plan with count `trailing_self_loops + 1` (the
    /// trailing rejections plus the arriving step counted at this plan; mirrors
    /// the `+ 1` in `push_step`).
    fn flush_final<W: Write>(&mut self, writer: &mut BenStreamWriter<W>) -> Result<()> {
        let count = self.trailing_self_loops + 1;
        let pending_assignment = std::mem::take(&mut self.pending_assignment);
        self.write_counted(writer, pending_assignment, count)
    }

    /// Write `assignment` with repeat `count`, splitting counts above
    /// `u16::MAX` into repeated same-assignment frames. A second `write_frame`
    /// with an unchanged assignment encodes a TwoDelta `Repeat`, and the chunk
    /// sum is preserved, so the decoded length still matches `total_samples`.
    /// Silent `as u16` truncation is forbidden: it would make the BENDL header
    /// `sample_count` disagree with the decodable stream length.
    fn write_counted<W: Write>(
        &mut self,
        writer: &mut BenStreamWriter<W>,
        assignment: Vec<u16>,
        count: u64,
    ) -> Result<()> {
        debug_assert!(count > 0, "frame count must be positive");
        const MAX_FRAME_COUNT: u64 = u16::MAX as u64;
        let mut remaining = count;
        // `write_frame` consumes the assignment Vec, so clone for every chunk
        // except the last.
        while remaining > MAX_FRAME_COUNT {
            writer.write_frame(assignment.clone(), u16::MAX)?;
            remaining -= MAX_FRAME_COUNT;
        }
        writer.write_frame(assignment, remaining as u16)?;
        self.total_samples += count;
        Ok(())
    }
}

pub struct BenWriter {
    /// Streaming TwoDelta encoder. `None` until [`StatsWriter::init`] builds it
    /// (which emits the banner); [`StatsWriter::close`] finalizes it.
    stream: Option<BenStreamWriter<Box<dyn Write + Send>>>,
    /// Output sink, held until `init` wraps it in the `stream` encoder.
    pending_output: Option<Box<dyn Write + Send>>,
    /// Shared TwoDelta frame bookkeeping.
    frames: TwoDeltaFrameBuffer,
}

/// Maps a [`BendlWriteError`] into the [`io::Error`] the [`StatsWriter`] trait
/// requires.
fn bendl_to_io(error: BendlWriteError) -> io::Error {
    io::Error::new(io::ErrorKind::Other, error)
}

/// Writes a self-describing `.bendl` file: the dual graph and run metadata as
/// front-loaded assets, plus the same TwoDelta BEN stream the `ben` writer
/// produces. BENDL needs a seekable sink, so this writer owns a concrete
/// `BufWriter<File>` rather than the boxed `output_buffer` other writers use.
pub struct BendlBenStreamWriter {
    /// Buffered output file, taken when `init` opens the bundle writer.
    pending_file: Option<BufWriter<File>>,
    /// The dual-graph JSON bytes embedded as the Graph asset (reordered if the
    /// caller requested an ordering).
    graph_bytes: Vec<u8>,
    /// The run-metadata JSON bytes embedded as the Metadata asset.
    metadata_bytes: Vec<u8>,
    /// Streaming TwoDelta encoder over the bundle's stream session; `None` until
    /// `init` builds it.
    stream: Option<BenStreamWriter<BendlStreamSession<BufWriter<File>>>>,
    /// Shared TwoDelta frame bookkeeping.
    frames: TwoDeltaFrameBuffer,
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

/// Writes objective scores for every step of a tilted chain.
pub struct ScoresWriter {
    /// The output stream that we would like to write to.
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
            previous_assignment: Vec::new(),
            last_step: 0,
        }
    }

    /// Canonicalizes the assignment vector.
    /// Meaning that instead of generating an assigmment
    /// vector of the form
    /// [2, 2, 2, 2, 3, 0, 0, 0, 3, 3, 1, 1, 3, 1, 1, 1]
    /// the values are reassigned by order of appearance
    /// indexed by 0, so the previous vector would become
    /// [0, 0, 0, 0, 1, 2, 2, 2, 1, 1, 3, 3, 1, 3, 3, 3]
    fn canonicalize_assignments(&self, partition: &Partition) -> Vec<u32> {
        let mut canon = partition.assignments.clone();
        // `None` distinguishes an unseen district from label 0.
        let mut dist_mapping: Vec<Option<u32>> = vec![None; partition.num_dists as usize];
        let mut cur_dist = 0;
        for (idx, &assn) in partition.assignments.iter().enumerate() {
            let label = *dist_mapping[assn as usize].get_or_insert_with(|| {
                let next = cur_dist;
                cur_dist += 1;
                next
            });
            canon[idx] = label;
        }
        canon
    }

    fn assignment(&self, partition: &Partition) -> Vec<u32> {
        if self.canonicalize {
            self.canonicalize_assignments(partition)
        } else {
            partition.assignments.clone()
        }
    }

    fn write_assignment(&mut self, step: u64, assignment: &[u32]) -> Result<()> {
        self.output
            .write_all(format!("{},{:?}\n", step, assignment).as_bytes())
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
            stream: None,
            pending_output: Some(output),
            frames: TwoDeltaFrameBuffer::new(),
        }
    }
}

impl BendlBenStreamWriter {
    /// `output` is the already-wrapped buffered output file; `graph_bytes` and
    /// `metadata_bytes` are the Graph and Metadata asset payloads to embed.
    pub fn new(
        output: BufWriter<File>,
        graph_bytes: Vec<u8>,
        metadata_bytes: Vec<u8>,
    ) -> BendlBenStreamWriter {
        BendlBenStreamWriter {
            pending_file: Some(output),
            graph_bytes,
            metadata_bytes,
            stream: None,
            frames: TwoDeltaFrameBuffer::new(),
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

impl ScoresWriter {
    pub fn new(output: Box<dyn Write + Send>) -> ScoresWriter {
        ScoresWriter { output }
    }

    /// Writes the CSV header and the initial score row at step 0.
    ///
    /// When `initial_district_scores` is non-empty, the header is extended
    /// with one column per district (`d_0,d_1,...,d_{N-1}`) and every row
    /// will carry per-district values. When empty, the bare `step,score`
    /// header is emitted and `step` must be called with an empty slice for
    /// every chain step.
    pub fn init(&mut self, score: f64, initial_district_scores: &[f64]) -> Result<()> {
        if initial_district_scores.is_empty() {
            self.output.write_all(b"step,score\n")?;
        } else {
            let mut header = String::from("step,score");
            for i in 0..initial_district_scores.len() {
                header.push_str(&format!(",d_{}", i));
            }
            header.push('\n');
            self.output.write_all(header.as_bytes())?;
        }
        self.step(0, score, initial_district_scores)
    }

    /// Writes the objective score for one chain step, plus any per-district
    /// scores. Pass an empty slice to emit the bare `step,score` format.
    pub fn step(&mut self, step: u64, score: f64, district_scores: &[f64]) -> Result<()> {
        if district_scores.is_empty() {
            self.output
                .write_all(format!("{},{}\n", step, score).as_bytes())
        } else {
            let mut row = format!("{},{}", step, score);
            for d in district_scores {
                row.push(',');
                row.push_str(&format!("{}", d));
            }
            row.push('\n');
            self.output.write_all(row.as_bytes())
        }
    }

    /// Flushes any buffered score rows to the underlying writer.
    pub fn flush(&mut self) -> Result<()> {
        self.output.flush()
    }

    pub fn close(&mut self) -> Result<()> {
        self.output.flush()
    }
}

impl StatsWriter for TSVWriter {
    fn init(&mut self, _graph: &Graph, _partition: &Partition) -> Result<()> {
        // TSV column header.
        self.output.write_all(
            b"step\tnon_adjacent\tno_split\tseam_length\ttilted_rejection\ta_label\tb_label\ta_pop\tb_pop\ta_nodes\tb_nodes\n",
        )?;
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
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:?}\t{:?}\n",
                    step,
                    counts.get(SelfLoopReason::NonAdjacent),
                    counts.get(SelfLoopReason::NoSplit),
                    counts.get(SelfLoopReason::SeamLength),
                    counts.get(SelfLoopReason::TiltedRejection),
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
        self.output.flush()
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
        self.output.flush()
    }
}

impl StatsWriter for AssignmentsOnlyWriter {
    fn init(&mut self, _graph: &Graph, partition: &Partition) -> Result<()> {
        let assignment = self.assignment(partition);
        self.write_assignment(0, &assignment)?;
        self.previous_assignment = assignment;
        self.last_step = 0;
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
        debug_assert_eq!(step, self.last_step + counts.sum() as u64 + 1);
        let previous_assignment = self.previous_assignment.clone();
        for self_loop_step in self.last_step + 1..step {
            self.write_assignment(self_loop_step, &previous_assignment)?;
        }
        let assignment = self.assignment(partition);
        self.write_assignment(step, &assignment)?;
        self.previous_assignment = assignment;
        self.last_step = step;
        Ok(())
    }

    fn self_loop(
        &mut self,
        step: u64,
        _graph: &Graph,
        _partition: &Partition,
        counts: &SelfLoopCounts,
    ) -> Result<()> {
        debug_assert_eq!(step, self.last_step + counts.sum() as u64);
        let previous_assignment = self.previous_assignment.clone();
        for self_loop_step in self.last_step + 1..=step {
            self.write_assignment(self_loop_step, &previous_assignment)?;
        }
        self.last_step = step;
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        self.output.flush()
    }
}

impl StatsWriter for CanonicalWriter {
    fn init(&mut self, _graph: &Graph, partition: &Partition) -> Result<()> {
        self.previous_assignment = partition.assignments.clone();
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
        for i in step - tot_count as u64..step {
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
        self.previous_assignment = partition.assignments.clone();
        self.output
            .write_all(
                format!(
                    "{}\n",
                    json!({
                        "assignment": self.previous_assignment,
                        "sample": step,
                    })
                )
                .as_bytes(),
            )
            .expect("Failed to write to output");
        Ok(())
    }

    fn self_loop(
        &mut self,
        step: u64,
        _graph: &Graph,
        _partition: &Partition,
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
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        self.output.flush()
    }
}

impl StatsWriter for BenWriter {
    fn init(&mut self, _graph: &Graph, partition: &Partition) -> Result<()> {
        let output = self
            .pending_output
            .take()
            .expect("BenWriter output already taken");
        // `for_ben` emits the TWODELTA banner immediately.
        self.stream = Some(BenStreamWriter::for_ben(output, BenVariant::TwoDelta)?);
        self.frames
            .set_initial(partition.assignments.iter().map(|&x| x as u16).collect());
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
        let next_assignment = partition.assignments.iter().map(|&x| x as u16).collect();
        let stream = self.stream.as_mut().expect("BenWriter stepped before init");
        self.frames.push_step(stream, next_assignment, counts.sum() as u64)
    }

    fn self_loop(
        &mut self,
        _step: u64,
        _graph: &Graph,
        _partition: &Partition,
        counts: &SelfLoopCounts,
    ) -> Result<()> {
        // Trailing rejections occur at the final plan and are folded into its
        // segment count by `close`. The runner emits one `self_loop` call per
        // pending batch; accumulate in case there are several.
        self.frames.add_self_loops(counts.sum() as u64);
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        let stream = self.stream.as_mut().expect("BenWriter closed before init");
        self.frames.flush_final(stream)?;
        stream.finish()?;
        Ok(())
    }
}

impl StatsWriter for BendlBenStreamWriter {
    fn init(&mut self, _graph: &Graph, partition: &Partition) -> Result<()> {
        let output = self
            .pending_file
            .take()
            .expect("BendlBenStreamWriter output already taken");
        let mut bundle = BendlWriter::new(output, AssignmentFormat::Ben)?;
        // Graph and Metadata are the only assets: the embedded (possibly
        // reordered) graph is self-consistent with the positional stream, so no
        // NodePermutationMap is needed.
        bundle
            .add_known_asset(
                KnownAssetKind::Graph,
                &self.graph_bytes,
                AddAssetOptions::defaults().json(),
            )
            .map_err(bendl_to_io)?;
        bundle
            .add_known_asset(
                KnownAssetKind::Metadata,
                &self.metadata_bytes,
                AddAssetOptions::defaults().json(),
            )
            .map_err(bendl_to_io)?;
        let session = bundle.into_stream_session().map_err(bendl_to_io)?;
        // `for_ben` emits the TWODELTA banner into the stream session.
        self.stream = Some(BenStreamWriter::for_ben(session, BenVariant::TwoDelta)?);
        self.frames
            .set_initial(partition.assignments.iter().map(|&x| x as u16).collect());
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
        let next_assignment = partition.assignments.iter().map(|&x| x as u16).collect();
        let stream = self
            .stream
            .as_mut()
            .expect("BendlBenStreamWriter stepped before init");
        self.frames
            .push_step(stream, next_assignment, counts.sum() as u64)
    }

    fn self_loop(
        &mut self,
        _step: u64,
        _graph: &Graph,
        _partition: &Partition,
        counts: &SelfLoopCounts,
    ) -> Result<()> {
        self.frames.add_self_loops(counts.sum() as u64);
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        let mut stream = self
            .stream
            .take()
            .expect("BendlBenStreamWriter closed before init");
        self.frames.flush_final(&mut stream)?;
        let total_samples = self.frames.total_samples;
        // `finish_into_inner` flushes any pending BEN frame; `finish_into_writer`
        // stamps the true total into the header before `finish` writes the directory.
        let session = stream.finish_into_inner()?;
        let sample_count = i64::try_from(total_samples).map_err(|_| {
            io::Error::new(
                io::ErrorKind::Other,
                "BENDL sample count exceeds the i64 header field",
            )
        })?;
        let bundle = session.finish_into_writer(sample_count);
        bundle.finish().map_err(bendl_to_io)?;
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
