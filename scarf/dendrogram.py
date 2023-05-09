import numpy as np
from typing import List, Dict
import networkx as nx
from .utils import logger, tqdmbar

__all__ = ["BalancedCut", "CoalesceTree", "make_digraph"]


def make_digraph(d: np.ndarray, clust_info=None) -> nx.DiGraph:
    """Convert dendrogram into directed graph."""

    g = nx.DiGraph()
    n = d.shape[0] + 1  # Dendrogram contains one less sample
    if clust_info is not None:
        if len(clust_info) != d.shape[0] + 1:
            raise ValueError(
                "ERROR: cluster information doesn't match number of leaves in dendrogram"
            )
    else:
        clust_info = np.ones(d.shape[0] + 1) * -1
    for i in tqdmbar(d, desc="Constructing graph from dendrogram"):
        v = i[2]  # Distance between clusters
        i = i.astype(int)
        g.add_node(n, nleaves=i[3], dist=v)
        if i[0] <= d.shape[0]:
            g.add_node(i[0], nleaves=0, dist=v, cluster=clust_info[i[0]])
        if i[1] <= d.shape[0]:
            g.add_node(i[1], nleaves=0, dist=v, cluster=clust_info[i[1]])
        g.add_edge(n, i[0])
        g.add_edge(n, i[1])
        n += 1
    if g.number_of_edges() != d.shape[0] * 2:
        logger.warning(
            "Number of edges in directed graph not twice the dendrogram shape"
        )
    return g


def CoalesceTree(graph: nx.DiGraph, clusters: np.ndarray) -> nx.DiGraph:
    def calc_steps_to_top(g: nx.DiGraph, c: np.ndarray):
        import pandas as pd

        s = {}
        for i in range(len(c)):
            s[i] = 0
            q = [i]
            while len(q) > 0:
                for j in g.predecessors(q.pop(0)):
                    s[i] += 1
                    q.append(j)
        return pd.Series(s).sort_values()

    def iter_predecessors(g: nx.DiGraph, v):
        q = [v]
        while len(q) > 0:
            for i in g.predecessors(q.pop(0)):
                yield i
                q.append(i)

    def aggregate_leaves(g: nx.DiGraph, v):
        q = [v]
        l = []
        while len(q) > 0:
            for i in g.successors(q.pop(0)):
                if g.nodes[i]["nleaves"] == 0:
                    l.append(i)
                else:
                    q.append(i)
        return l

    def get_holding_nodes(g: nx.DiGraph, c):
        hn = {}
        s = calc_steps_to_top(g, c)
        for i in tqdmbar(set(c), desc="Identifying the top node for cluster"):
            l = set(np.where(c == i)[0])
            nl = len(l)
            for j in iter_predecessors(g, s.reindex(l).idxmin()):
                if g.nodes[j]["nleaves"] >= nl:
                    l2 = aggregate_leaves(g, j)
                    if len(l.intersection(l2)) == nl:
                        hn[j] = i
                        break
        return hn

    def aggregate_predecessors(g: nx.DiGraph, v):
        p = []
        for i in iter_predecessors(g, v):
            p.append(i)
        return p

    def make_subgraph(g: nx.DiGraph, vs):
        sn = list(vs.keys())
        for i in vs:
            sn.extend(aggregate_predecessors(g, i))
        sn = list(set(sn))
        sg = nx.DiGraph(nx.subgraph(g, sn))
        for i in vs:
            sg.nodes[i]["partition_id"] = vs[i]
        return sg

    return make_subgraph(graph, get_holding_nodes(graph, clusters))


class BalancedCut:
    def __init__(
        self,
        dendrogram: np.ndarray,
        max_size: int,
        min_size: int,
        max_distance_fc: float,
    ):
        self.nCells = dendrogram.shape[0] + 1
        self.graph = make_digraph(dendrogram)
        self.maxSize = max_size
        self.minSize = min_size
        self.maxDistFc = max_distance_fc
        self.branchpoints = self._get_branchpoints()

    def _successors(self, start: int, min_leaves: int) -> List[int]:
        """Get tree downstream of a node."""
        q = [start]
        d = []
        while len(q) > 0:
            i = q.pop(0)
            if self.graph.nodes[i]["nleaves"] > min_leaves:
                d.append(i)
                q.extend(list(self.graph.successors(i)))
        return d[1:]

    def _get_mean_dist(self, start_node: int) -> float:
        """Get mean distances in downstream tree of a node."""
        s_nodes = self._successors(start_node, -1)
        return np.array([self.graph.nodes[x]["dist"] for x in s_nodes]).mean()

    def _are_subtrees_mergeable(self, s1: int, s2: int) -> bool:
        n1, n2 = self.graph.nodes[s1]["nleaves"], self.graph.nodes[s2]["nleaves"]
        if n1 > self.minSize and n2 > self.minSize:
            d1, d2 = self.graph.nodes[s1]["dist"], self.graph.nodes[s2]["dist"]
            if d1 / d2 > self.maxDistFc or d2 / d1 > self.maxDistFc:
                logger.trace(f"Will not merge {s1} and {s2} because of high distance")
                return False
            else:
                md1, md2 = self._get_mean_dist(s1), self._get_mean_dist(s2)
                if md1 / md2 > self.maxDistFc or md2 / md1 > self.maxDistFc:
                    logger.trace(
                        f"Will not merge {s1} and {s2} because of high distance of successors"
                    )
                    return False
        return True

    def _get_branchpoints(self) -> Dict[int, List[int]]:
        """Aggregate leaves bottom up until target size is reached."""
        n_leaves = int((self.graph.number_of_nodes() + 1) / 2)
        leaves = {x: None for x in range(n_leaves)}
        bps = {}
        pbar = tqdmbar(total=len(leaves), desc="Identifying nodes to split")
        while len(leaves) > 0:
            leaf, _ = leaves.popitem()
            pbar.update(1)
            logger.trace(f"FRESH STEP: Leaf {leaf} plucked as base leaf")
            cur = leaf
            while True:
                temp = next(self.graph.predecessors(cur))
                if temp in bps:
                    logger.trace(f"Will not climb to {temp} as already a branchpoint")
                    break
                if self.graph.nodes[temp]["nleaves"] > self.maxSize:
                    logger.trace(
                        f"Will not climb to {temp} because too many leaves exist"
                    )
                    break
                s1, s2 = list(self.graph.successors(temp))
                if self._are_subtrees_mergeable(s1, s2) is False:
                    break
                cur = temp
            logger.trace(f"Aggregating from branch {cur} for leaf {leaf}")
            bps[cur] = [leaf]
            s = [cur]
            while len(s) > 0:
                i = s.pop()
                if i in leaves:
                    bps[cur].append(i)
                    leaves.pop(i)
                    logger.trace(f"Leaf {i} plucked in aggregation step")
                    pbar.update(1)
                elif i in bps and i != cur:
                    logger.trace(f"Skipping branch {i} because its already taken")
                elif self.graph.nodes[i]["nleaves"] >= self.maxSize and i != cur:
                    logger.trace(f"Skipping branch {i} to prevent greedy behaviour")
                else:
                    s.extend(list(self.graph.successors(i)))
        pbar.close()
        return bps

    def _valid_names_in_branchpoints(self) -> None:
        leaves = []
        for i in self.branchpoints:
            leaves.extend(self.branchpoints[i])
        n_leaves = len(leaves)
        if n_leaves != self.nCells:
            raise ValueError(
                "ERROR: Not all leaves present in branchpoints. This bug must be reported"
            )
        minl = min(leaves)
        if minl != 0:
            raise ValueError(f"ERROR: minimum leaf label is {minl} rather than 0")
        maxl = max(leaves)
        if n_leaves != maxl + 1:
            raise ValueError(
                f"ERROR: maximum leaf label is {maxl} while total estimated leaves are {n_leaves}"
            )
        return None

    def get_clusters(self) -> np.ndarray:
        """Make cluster label array from `get_branchpoints` output."""
        self._valid_names_in_branchpoints()
        c = np.zeros(self.nCells).astype(int)
        for n, i in enumerate(self.branchpoints, start=1):
            c[self.branchpoints[i]] = n
        if (c == 0).sum() > 0:
            logger.warning(f"{(c == 0).sum()} samples were not assigned a cluster")
            c[c == 0] = -1
        return c

    # def test(self):
    # n = 30
    # np.random.seed(154)
    # randz = ward(gamma.rvs(0.3, size=n * 4).reshape((n, 4)))
    #
    # graph = z_to_g(randz)
    # branchpoints = get_branchpoints(graph, max_leaf=10, min_leaves=2, fc=1.5)
    # clusters = bps_to_clusts(branchpoints)
    # dend = dendrogram(randz)
    # clusters[dend['leaves']]
    #
    # res = [9, 5, 5, 1, 1, 1, 1, 7, 7, 7, 3, 3, 3, 3, 3, 4, 4, 4, 2, 2, 2, 8,
    #        8, 8, 6, 6, 6, 6, 6, 6]
