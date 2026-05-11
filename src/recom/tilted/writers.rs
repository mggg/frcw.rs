//! Writer-thread loops that consume packets from bounded channels and
//! forward them to the user-supplied stats / scores writers.

use crossbeam_channel::Receiver;

use crate::graph::Graph;
use crate::partition::Partition;
use crate::stats::{ScoresWriter, StatsWriter};

use super::packets::{TiltedScorePacket, TiltedStatsPacket};

/// Starts a chain-statistics writer thread.
///
/// # Arguments
///
/// * `graph` - Writer-owned graph clone.
/// * `partition` - Writer-owned starting partition, updated from accepted proposals.
/// * `writer` - Stats writer receiving accepted proposals and self-loop counts.
/// * `recv` - Channel receiving asynchronous stats write packets.
pub(super) fn start_tilted_stats_writer(
    graph: &Graph,
    mut partition: Partition,
    writer: &mut dyn StatsWriter,
    recv: Receiver<TiltedStatsPacket>,
) {
    writer.init(graph, &partition).unwrap();
    let mut next = recv.recv().unwrap();
    while !next.terminate {
        if let Some(proposal) = next.proposal {
            partition.update(&proposal);
            writer
                .step(next.step, graph, &partition, &proposal, &next.counts)
                .unwrap();
        } else {
            writer
                .self_loop(next.step, graph, &partition, &next.counts)
                .unwrap();
        }
        next = recv.recv().unwrap();
    }
    writer.close().unwrap();
}

/// Starts a score writer thread.
///
/// # Arguments
///
/// * `writer` - Score writer receiving per-step objective scores.
/// * `initial_score` - Objective score of the starting partition.
/// * `initial_district_scores` - Per-district score vector for the starting
///   partition. An empty vector switches the writer to the legacy three-column
///   `step,score,best_score` output; a non-empty vector emits
///   `step,score,best_score,d_0,...,d_{N-1}` and the writer caches it so
///   pure-rejection packets can replay the last-known district scores.
/// * `recv` - Channel receiving asynchronous score write packets.
pub(super) fn start_tilted_score_writer(
    writer: &mut ScoresWriter,
    initial_score: f64,
    initial_district_scores: Vec<f64>,
    recv: Receiver<TiltedScorePacket>,
) {
    writer
        .init(initial_score, &initial_district_scores)
        .unwrap();
    let mut last_districts = initial_district_scores;
    let mut next = recv.recv().unwrap();
    while !next.terminate {
        if let Some(new_districts) = next.district_scores.take() {
            last_districts = new_districts;
        }
        for step in next.first_step..=next.last_step {
            writer
                .step(step, next.score, next.best_score, &last_districts)
                .unwrap();
        }
        next = recv.recv().unwrap();
    }
    writer.close().unwrap();
}
