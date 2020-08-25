import pandas as pd
from typing import List, Tuple, Dict
import numpy as np

__all__ = ['pcst']


def get_seed_nodes(clusts: pd.Series, frac: float,
                   min_nodes: int, rand_num: int) -> Dict[int, None]:
    seeds = []
    for i in clusts.unique():
        c = clusts[clusts == i]
        if len(c) > min_nodes:
            s = c.sample(frac=frac, random_state=rand_num).index
            if len(s) < min_nodes:
                s = c.sample(n=min_nodes, random_state=rand_num).index
        else:
            s = c.index
        seeds.extend(s)
    return {x: None for x in seeds}


def pcst(graph, clusters: pd.Series, seed_frac: float, min_nodes: int,
         rewards: Tuple[float, float], pruning_method: str,
         rand_state: int) -> Tuple[List, List]:
    from scipy.sparse.csgraph import connected_components
    import pcst_fast

    ss, se = [], []
    _, l = connected_components(graph)
    seeds = get_seed_nodes(clusters, seed_frac, min_nodes, rand_state)
    print(f"INFO: {len(seeds)} seed nodes selected", flush=True)
    for i in set(l):
        idx = np.where(l == i)[0]
        g = graph[idx].T[idx].tocoo()
        c = (1 + g.data.min()) - g.data
        r = [rewards[0] if x in seeds else rewards[1] for x in idx]
        e = np.vstack([g.row, g.col]).T
        x, y = pcst_fast.pcst_fast(e, r, c, -1, 1, pruning_method, 0)
        ss.extend(idx[x])
        se.extend([[idx[x[0]], idx[x[1]]] for x in e[y]])
    cover = set(ss).intersection(list(seeds.keys()))
    if len(cover) != len(seeds):
        print(f"WARNING: Not all seed nodes in downsampled nodes. Try increasing the reward for seeds", flush=True)
    seed_ratio = len(ss) / len(seeds)
    if seed_ratio > 2 and rewards[1] > 0:
        print(f"WARNING: High seed ratio detected. Try decreasing the non-seed reward", flush=True)
    print(f"INFO: {len(ss)} steiner nodes found. {len(cover)} of which are present in seed list.", flush=True)
    down_ratio = 100 * len(ss)/graph.shape[0]
    down_ratio = "%.2f" % down_ratio
    print(f"INFO: Downsampled %: {down_ratio}%", flush=True)
    seed_ratio = "%.3f" % seed_ratio
    print(f"INFO: Seed ratio: {seed_ratio}", flush=True)
    return ss, se
