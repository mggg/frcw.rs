//! Functions for generating random spanning trees.
use crate::buffers::SpanningTreeBuffer;
use crate::graph::{Edge, Graph};
use rand::rngs::SmallRng;
use rand::Rng;
use std::cmp::{max, min};

pub trait SpanningTreeSampler {
    /// Samples a random tree of `graph` using `rng`; inserts the tree into `buf`.
    fn random_spanning_tree(
        &mut self,
        graph: &Graph,
        buf: &mut SpanningTreeBuffer,
        rng: &mut SmallRng,
    );

    /// Variant that lets region-aware samplers read node attributes from the
    /// parent graph without requiring them to be copied onto `graph`. The
    /// `raw_nodes` slice maps subgraph node indices to parent graph node indices.
    ///
    /// The default implementation ignores the parent graph and delegates to
    /// [`random_spanning_tree`]. Samplers that need attributes (e.g.
    /// [`RegionAwareSampler`]) override this.
    fn random_spanning_tree_with_parent(
        &mut self,
        graph: &Graph,
        _parent: &Graph,
        _raw_nodes: &[usize],
        buf: &mut SpanningTreeBuffer,
        rng: &mut SmallRng,
    ) {
        self.random_spanning_tree(graph, buf, rng);
    }
}
pub use crate::spanning_tree::rmst::{RMSTSampler, RegionAwareSampler};
pub use crate::spanning_tree::ust::USTSampler;

/// Spanning tree sampling from the uniform distribution.
mod ust {
    use super::*;
    use crate::buffers::RandomRangeBuffer;

    /// A reusable buffer for Wilson's algorithm.
    pub struct USTBuffer {
        /// Boolean representation of the subset of nodes in the spanning tree.
        pub in_tree: Vec<bool>,
        /// The next node in the spanning tree (for a chosen ordering).
        pub next: Vec<i64>,
        /// The edges in the MST.
        pub edges: Vec<usize>,
    }

    impl USTBuffer {
        /// Creates a buffer for a spanning tree of a subgraph
        /// within a graph of size `n`.
        pub fn new(n: usize) -> USTBuffer {
            return USTBuffer {
                in_tree: vec![false; n],
                next: vec![-1 as i64; n],
                edges: Vec::<usize>::with_capacity(n - 1),
            };
        }

        /// Resets the buffer.
        pub fn clear(&mut self) {
            self.in_tree.fill(false);
            self.next.fill(-1);
            self.edges.clear();
        }
    }

    /// Samples random spanning trees from the uniform distribution.
    pub struct USTSampler {
        /// A buffer for Wilson's algorithm.
        ust_buf: USTBuffer,
        /// A reservoir of random bytes (used for quickly selecting random node neighbors).
        range_buf: RandomRangeBuffer,
    }

    impl USTSampler {
        /// Creates a UST sampler (and underlying buffers) for a graph of approximate
        /// size `n`. (A reservoir of random bytes is initialized using `rng`.)
        pub fn new(n: usize, rng: &mut SmallRng) -> USTSampler {
            USTSampler {
                ust_buf: USTBuffer::new(n),
                range_buf: RandomRangeBuffer::new(rng),
            }
        }
    }

    impl SpanningTreeSampler for USTSampler {
        /// Draws a random spanning tree of a graph from the uniform distribution.
        /// Returns nothing; The MST buffer `buf` is updated in place.
        ///
        /// We use Wilson's algorithm [1] (which is, in essence, a self-avoiding random
        /// walk) to generate the tree.
        ///
        /// # Arguments
        /// * `graph` - The graph to form a spanning tree from. The maximum degree
        ///   of the graph must be <=256; otherwise, sampling from the uniform
        ///   distribution is not guaranteed.
        /// * `buf` - The buffer to insert the spanning tree into.
        /// * `rng` - A random number generator (used to select the spanning tree
        ///   root and refresh the random byte reservoir).
        ///
        /// # References
        /// [1]  Wilson, David Bruce. "Generating random spanning trees more quickly
        ///      than the cover time." Proceedings of the twenty-eighth annual ACM
        ///      symposium on Theory of computing. 1996.
        fn random_spanning_tree(
            &mut self,
            graph: &Graph,
            buf: &mut SpanningTreeBuffer,
            rng: &mut SmallRng,
        ) {
            buf.clear();
            self.ust_buf.clear();
            let n = graph.pops.len();
            let root = rng.random_range(0..n);
            self.ust_buf.in_tree[root] = true;
            for i in 0..n {
                let mut u = i;
                while !self.ust_buf.in_tree[u] {
                    let neighbors = &graph.neighbors[u];
                    let neighbor =
                        neighbors[self.range_buf.range(rng, neighbors.len() as u8) as usize];
                    self.ust_buf.next[u] = neighbor as i64;
                    u = neighbor;
                }
                u = i;
                while !self.ust_buf.in_tree[u] {
                    self.ust_buf.in_tree[u] = true;
                    u = self.ust_buf.next[u] as usize;
                }
            }

            for (curr, &prev) in self.ust_buf.next.iter().enumerate() {
                if prev >= 0 {
                    let a = min(curr, prev as usize);
                    let b = max(curr, prev as usize);
                    let mut edge_idx = graph.edges_start[a];
                    while graph.edges[edge_idx].0 == a {
                        if graph.edges[edge_idx].1 == b {
                            self.ust_buf.edges.push(edge_idx);
                            break;
                        }
                        edge_idx += 1;
                    }
                }
            }
            if self.ust_buf.edges.len() != n - 1 {
                panic!(
                    "expected to have {} edges in MST but got {}",
                    n - 1,
                    self.ust_buf.edges.len()
                );
            }

            for &edge in self.ust_buf.edges.iter() {
                let Edge(src, dst) = graph.edges[edge];
                buf.st[src].push(dst);
                buf.st[dst].push(src);
            }
        }
    }
}

/// Spanning tree sampling via random edge weights.
mod rmst {
    use super::*;
    use petgraph::unionfind::UnionFind;
    use rand::seq::SliceRandom;

    /// Samples random spanning trees by sampling random edge weights and finding
    /// the minimum spanning tree.
    pub struct RMSTSampler {
        /// Buffer for randomly ordered edges.
        edges_by_weight: Vec<Edge>,
    }

    /// Weighted random-MST sampler: samples random edge weights, adds optional
    /// per-region surcharges (from node attributes) and optional per-edge
    /// additions (from edge attributes), then finds the minimum spanning tree.
    ///
    /// With empty `region_weights` and non-empty `edge_weight_keys` this is a
    /// plain weighted RMST; with `region_weights` it is the region-aware sampler.
    pub struct RegionAwareSampler {
        /// Buffer for random edge weights.
        weights: Vec<f64>,
        /// Buffer for random edge weights with indices.
        weights_with_indices: Vec<(usize, f64)>,
        /// Buffer for randomly ordered edges.
        edges_by_weight: Vec<Edge>,
        /// Sampler configuration (column -> weight).
        region_weights: Vec<(String, f64)>,
        /// Per-edge attribute columns whose values are added to edge weights.
        /// An edge missing a key contributes 0 (the loader stores 0 for it).
        edge_weight_keys: Vec<String>,
    }

    impl RMSTSampler {
        /// Initializes a random MST sampler for a graph with approximate size `n`.
        pub fn new(n: usize) -> RMSTSampler {
            RMSTSampler {
                edges_by_weight: Vec::<Edge>::with_capacity(8 * n),
            }
        }
    }

    impl RegionAwareSampler {
        /// Initializes a weighted random MST sampler for a graph with approximate
        /// size `n`. `region_weights` are per-region surcharges (may be empty);
        /// `edge_weight_keys` are per-edge attribute columns added to edge weights
        /// (may be empty).
        pub fn new(
            n: usize,
            region_weights: Vec<(String, f64)>,
            edge_weight_keys: Vec<String>,
        ) -> RegionAwareSampler {
            RegionAwareSampler {
                weights: Vec::<f64>::with_capacity(8 * n),
                weights_with_indices: Vec::<(usize, f64)>::with_capacity(8 * n),
                edges_by_weight: Vec::<Edge>::with_capacity(8 * n),
                region_weights: region_weights,
                edge_weight_keys: edge_weight_keys,
            }
        }
    }

    /// Given an edge order (`edges_by_weight`), uses a greedy algorithm
    /// analogous to Kruskal's algorithm to find the "minimum" spanning tree
    /// according to the given edge order.
    fn greedy_spanning_tree(
        graph: &Graph,
        buf: &mut SpanningTreeBuffer,
        edges_by_weight: &Vec<Edge>,
    ) {
        buf.clear();

        // Initialize a union-find data structure to keep track of connected
        // components of the graph.
        // TODO: buffer this?
        let mut uf = UnionFind::<usize>::new(graph.neighbors.len());

        // Apply Kruskal's algorithm: add edges until the graph is connected.
        let n_edges = graph.pops.len() - 1;
        let mut n_unions = 0;
        for &Edge(src, dst) in edges_by_weight.iter() {
            if n_unions == n_edges {
                break;
            }
            if !uf.equiv(src, dst) {
                uf.union(src, dst);
                buf.st[src].push(dst);
                buf.st[dst].push(src);
                n_unions += 1;
            }
        }
        if n_unions != n_edges {
            panic!(
                "expected to have {} edges in MST but got {}",
                n_edges, n_unions
            );
        }
    }

    impl SpanningTreeSampler for RMSTSampler {
        /// Draws a random spanning tree of a graph by sampling random edge weights
        /// and finding the minimum spanning tree (using Kruskal's algorithm).
        /// Returns nothing; The MST buffer `buf` is updated in place.
        ///
        /// # Arguments
        /// * `graph` - The graph to form a spanning tree from.
        /// * `buf` - The buffer to insert the spanning tree into.
        /// * `rng` - A random number generator (used to generate random edge weights).
        fn random_spanning_tree(
            &mut self,
            graph: &Graph,
            buf: &mut SpanningTreeBuffer,
            rng: &mut SmallRng,
        ) {
            self.edges_by_weight.clone_from(&graph.edges);
            self.edges_by_weight.shuffle(rng);
            greedy_spanning_tree(graph, buf, &self.edges_by_weight);
        }
    }

    impl RegionAwareSampler {
        /// Core sampling routine shared by [`SpanningTreeSampler::random_spanning_tree`]
        /// and [`SpanningTreeSampler::random_spanning_tree_with_parent`]. Reads
        /// region attribute values from `attr_source` using `attr_indices[edge.X]`
        /// to look up node attribute positions.
        fn sample_with_attr_source(
            &mut self,
            graph: &Graph,
            attr_source: &Graph,
            attr_indices: &[usize],
            buf: &mut SpanningTreeBuffer,
            rng: &mut SmallRng,
        ) {
            // An allocation-free scheme for weight sampling: maintain three
            // separate buffers.
            //   * Efficiently generate raw weights by edge index
            //     using the `weights` buffer. (RNG sampling + offsets)
            //   * Convert these weights to (edge index, edge weight) pairs in
            //     the `weights_with_indices` buffer. Sort this buffer in place.
            //   * Copy edges from the graph into the `edges_by_weight` based
            //     on the edge index order in `weights_with_indices`.
            let n_edges = graph.edges.len();
            self.weights.resize(n_edges, 0.0);
            rng.fill(&mut self.weights[..]);
            for (region_col, region_weight) in self.region_weights.iter() {
                let col = &attr_source.attr[region_col];
                for (idx, edge) in graph.edges.iter().enumerate() {
                    let a = &col[attr_indices[edge.0]];
                    let b = &col[attr_indices[edge.1]];
                    if a == "null" || a.is_empty() || b == "null" || b.is_empty() || a != b {
                        self.weights[idx] += region_weight;
                    }
                }
            }

            // Add per-edge attribute values. Edge attributes are indexed against
            // the parent (`attr_source`) edge list, so each subgraph edge is
            // mapped back to its parent edge index via the `edges_start` scan
            // (the same lookup used in `USTSampler`).
            for key in self.edge_weight_keys.iter() {
                let vals = attr_source
                    .edge_attr
                    .get(key)
                    .unwrap_or_else(|| panic!("Missing edge attribute '{}'", key));
                for (idx, edge) in graph.edges.iter().enumerate() {
                    let a = min(attr_indices[edge.0], attr_indices[edge.1]);
                    let b = max(attr_indices[edge.0], attr_indices[edge.1]);
                    let mut e = attr_source.edges_start[a];
                    while attr_source.edges[e].0 == a {
                        if attr_source.edges[e].1 == b {
                            self.weights[idx] += vals[e];
                            break;
                        }
                        e += 1;
                    }
                }
            }

            self.weights_with_indices.clear();
            for (idx, &weight) in self.weights.iter().enumerate() {
                self.weights_with_indices.push((idx, weight));
            }

            // Sort the edges so that the largest weight is last.
            self.weights_with_indices
                .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            self.edges_by_weight.clear();
            for (edge_idx, _) in self.weights_with_indices.iter() {
                self.edges_by_weight.push(graph.edges[*edge_idx]);
            }

            greedy_spanning_tree(graph, buf, &self.edges_by_weight);
        }
    }

    impl SpanningTreeSampler for RegionAwareSampler {
        /// Draws a random spanning tree of a graph by sampling random edge weights
        /// and finding the minimum spanning tree (using Kruskal's algorithm).
        /// Reweights edges based on the region-aware settings in the buffer
        /// configuration: edges that span two units (e.g. two counties)
        /// are downweighted by the unit's weight, such that (assuming positive weights)
        /// the minimum spanning tree is more likely to contain edges _between_ units.
        ///
        /// Returns nothing; The MST buffer `buf` is updated in place.
        ///
        /// # Arguments
        /// * `graph` - The graph to form a spanning tree from.
        /// * `buf` - The buffer to insert the spanning tree into.
        /// * `rng` - A random number generator (used to generate random edge weights).
        fn random_spanning_tree(
            &mut self,
            graph: &Graph,
            buf: &mut SpanningTreeBuffer,
            rng: &mut SmallRng,
        ) {
            let identity: Vec<usize> = (0..graph.pops.len()).collect();
            self.sample_with_attr_source(graph, graph, &identity, buf, rng);
        }

        fn random_spanning_tree_with_parent(
            &mut self,
            graph: &Graph,
            parent: &Graph,
            raw_nodes: &[usize],
            buf: &mut SpanningTreeBuffer,
            rng: &mut SmallRng,
        ) {
            self.sample_with_attr_source(graph, parent, raw_nodes, buf, rng);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffers::SpanningTreeBuffer;
    use rand::SeedableRng;
    use std::collections::HashMap;

    /// A triangle on nodes 0, 1, 2 with a `barrier` edge attribute parallel to
    /// `edges` = [(0,1), (0,2), (1,2)].
    fn triangle(barrier: [f64; 3]) -> Graph {
        let mut edge_attr = HashMap::new();
        edge_attr.insert("barrier".to_string(), barrier.to_vec());
        Graph {
            edges: vec![Edge(0, 1), Edge(0, 2), Edge(1, 2)],
            pops: vec![1, 1, 1],
            neighbors: vec![vec![1, 2], vec![0, 2], vec![0, 1]],
            edges_start: vec![0, 2, 3],
            total_pop: 3,
            attr: HashMap::new(),
            edge_attr,
            int_attr: HashMap::new(),
        }
    }

    fn tree_has_edge(buf: &SpanningTreeBuffer, u: usize, v: usize) -> bool {
        buf.st[u].contains(&v)
    }

    #[test]
    fn edge_weight_key_forces_high_weight_edge_out_of_tree() {
        // Edge (1,2) carries an overwhelming weight, so the minimum spanning
        // tree must never include it (1 and 2 connect via node 0 instead).
        let graph = triangle([0.0, 0.0, 1e6]);
        let mut sampler =
            RegionAwareSampler::new(graph.pops.len(), vec![], vec!["barrier".to_string()]);
        let mut buf = SpanningTreeBuffer::new(graph.pops.len());
        for seed in 0..64u64 {
            let mut rng = SmallRng::seed_from_u64(seed);
            sampler.random_spanning_tree(&graph, &mut buf, &mut rng);
            assert!(
                !tree_has_edge(&buf, 1, 2),
                "edge (1,2) should be excluded (seed {})",
                seed
            );
            let n_edges: usize = buf.st.iter().map(|a| a.len()).sum::<usize>() / 2;
            assert_eq!(n_edges, 2, "expected a spanning tree of 2 edges");
        }
    }

    #[test]
    fn without_edge_weight_key_every_edge_can_appear() {
        // With no edge weight key the three edges are ordered by random base
        // weights alone, so edge (1,2) should appear in at least one sample.
        // Guards against the key silently always applying.
        let graph = triangle([0.0, 0.0, 0.0]);
        let mut sampler = RegionAwareSampler::new(graph.pops.len(), vec![], vec![]);
        let mut buf = SpanningTreeBuffer::new(graph.pops.len());
        let mut seen = false;
        for seed in 0..64u64 {
            let mut rng = SmallRng::seed_from_u64(seed);
            sampler.random_spanning_tree(&graph, &mut buf, &mut rng);
            if tree_has_edge(&buf, 1, 2) {
                seen = true;
                break;
            }
        }
        assert!(
            seen,
            "edge (1,2) should sometimes appear without an edge weight key"
        );
    }

    #[test]
    fn edge_weight_key_maps_subgraph_edge_to_parent_index() {
        // Parent: nodes 0,1,2,3 with edges (0,1),(1,2),(1,3),(2,3); barrier on
        // parent edge (2,3). The subgraph induced by parent nodes {1,2,3} is a
        // triangle (raw_nodes = [1,2,3]); its local edge (1,2) maps to parent
        // edge (2,3) and must be excluded. Exercises the subgraph->parent edge
        // index mapping used by `random_spanning_tree_with_parent`.
        let mut edge_attr = HashMap::new();
        edge_attr.insert("barrier".to_string(), vec![0.0, 0.0, 0.0, 1e6]);
        let parent = Graph {
            edges: vec![Edge(0, 1), Edge(1, 2), Edge(1, 3), Edge(2, 3)],
            pops: vec![1, 1, 1, 1],
            neighbors: vec![vec![1], vec![0, 2, 3], vec![1, 3], vec![1, 2]],
            edges_start: vec![0, 1, 3, 4],
            total_pop: 4,
            attr: HashMap::new(),
            edge_attr,
            int_attr: HashMap::new(),
        };
        let subgraph = triangle([0.0, 0.0, 0.0]); // subgraph carries no edge attrs in production
        let raw_nodes = vec![1usize, 2, 3];
        let mut sampler = RegionAwareSampler::new(3, vec![], vec!["barrier".to_string()]);
        let mut buf = SpanningTreeBuffer::new(3);
        for seed in 0..64u64 {
            let mut rng = SmallRng::seed_from_u64(seed);
            sampler.random_spanning_tree_with_parent(
                &subgraph, &parent, &raw_nodes, &mut buf, &mut rng,
            );
            assert!(
                !tree_has_edge(&buf, 1, 2),
                "subgraph edge (1,2) -> parent (2,3) should be excluded (seed {})",
                seed
            );
        }
    }
}
