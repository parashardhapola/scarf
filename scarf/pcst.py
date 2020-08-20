import pandas as pd
from typing import List, Tuple, Dict
import numpy as np

__all__ = ['pcst']


def get_seed_nodes(clusts: pd.Series, frac: float,
                   min_nodes: int, rand_num: int) -> Dict[int, None]:
    seeds = {}
    for i in clusts.unique():
        c = clusts[clusts == i]
        if len(c) > min_nodes:
            s = c.sample(frac=frac, random_state=rand_num).index
            if len(s) < min_nodes:
                s = c.sample(n=min_nodes, random_state=rand_num).index
        else:
            s = c.index
        seeds.update({x: None for x in s})
    return seeds


def pcst(graph, clusters: pd.Series, seed_frac: float, min_nodes: int,
         rewards: Tuple[float, float], pruning_method: str,
         rand_state: int) -> Tuple[List, List]:
    from scipy.sparse.csgraph import connected_components
    import pcst_fast

    ss, se = [], []
    _, l = connected_components(graph)
    seeds = get_seed_nodes(clusters, seed_frac, min_nodes, rand_state)
    for i in set(l):
        idx = np.where(l == i)[0]
        g = graph.tocsr()[idx].T[idx].tocoo()
        c = (1 + g.data.min()) - g.data
        r = [rewards[0] if x in seeds else rewards[1] for x in idx]
        e = np.vstack([g.row, g.col]).T
        x, y = pcst_fast.pcst_fast(e, r, c, -1, 1, pruning_method, 0)
        ss.extend(idx[x])
        se.extend([[idx[x[0]], idx[x[1]]] for x in e[y]])
    cover = set(ss).intersection(list(seeds.keys()))
    print(f"INFO: {len(ss)} found. {len(cover)} seed nodes present in the tree.")
    return ss, se
