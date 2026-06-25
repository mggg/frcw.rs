//! Writer-thread loops that consume packets from bounded channels and forward
//! them to the user-supplied stats / scores writers.

use crossbeam_channel::Receiver;

use super::super::RecomProposal;
use super::packets::{BurstScorePacket, BurstStatsPacket};
use crate::graph::Graph;
use crate::partition::Partition;
use crate::stats::{ScoresWriter, SelfLoopCounts, StatsWriter};

/// Starts a chain-statistics writer thread.
///
/// Short-bursts emits a full partition on every output record (the runner
/// already pays the partition clone for every accepted step), so the
/// writer thread receives partitions directly and synthesizes an empty
/// proposal on each `writer.step` call. Writers that need proposal-level data
/// (TSV, JSONL, pcompress) will see empty/zeroed proposal fields; prefer the
/// `assignments`, `canonicalized-assignments`, `canonical`, or `ben` writers
/// when running short bursts.
///
/// # Arguments
///
/// * `graph` - Graph passed to `writer.step` for context.
/// * `partition` - Writer-owned starting partition; `init` is called against it.
/// * `writer` - Stats writer receiving accepted partitions.
/// * `recv` - Channel receiving asynchronous stats write packets.
pub(super) fn start_burst_stats_writer(
    graph: &Graph,
    partition: Partition,
    writer: &mut dyn StatsWriter,
    recv: Receiver<BurstStatsPacket>,
) {
    writer.init(graph, &partition).unwrap();
    let dummy = RecomProposal {
        a_label: 0,
        b_label: 0,
        a_pop: 0,
        b_pop: 0,
        a_nodes: Vec::new(),
        b_nodes: Vec::new(),
    };
    let counts = SelfLoopCounts::default();
    let mut next = recv.recv().unwrap();
    while !next.terminate {
        if let Some(partition) = next.partition.as_ref() {
            writer
                .step(next.step, graph, partition, &dummy, &counts)
                .unwrap();
        }
        next = recv.recv().unwrap();
    }
    writer.close().unwrap();
}

/// Starts a score writer thread.
///
/// Mirrors the tilted-run score writer: caches the most recent
/// `district_scores` vector and replays it for any packet that omits a fresh
/// vector. An empty `initial_district_scores` switches the writer into the
/// bare `step,score` output.
///
/// # Arguments
///
/// * `writer` - Score writer receiving per-event objective scores.
/// * `initial_score` - Objective score of the starting partition.
/// * `initial_district_scores` - Per-district score vector for the starting
///   partition. Empty switches to the bare `step,score` header.
/// * `recv` - Channel receiving asynchronous score write packets.
pub(super) fn start_burst_score_writer(
    writer: &mut ScoresWriter,
    initial_score: f64,
    initial_district_scores: Vec<f64>,
    recv: Receiver<BurstScorePacket>,
) {
    writer
        .init(initial_score, &initial_district_scores)
        .unwrap();
    writer.flush().unwrap();
    let mut last_districts = initial_district_scores;
    let mut next = recv.recv().unwrap();
    while !next.terminate {
        if let Some(new_districts) = next.district_scores.take() {
            last_districts = new_districts;
        }
        writer.step(next.step, next.score, &last_districts).unwrap();
        writer.flush().unwrap();
        next = recv.recv().unwrap();
    }
    writer.close().unwrap();
}
