from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import MiniBatchKMeans
import hnswlib
from tqdm import tqdm
import numpy as np
from scipy import sparse
from gensim.models import LsiModel
from . import threadpool_limits

__all__ = ['AnnStream']


def clean_kmeans_kwargs(kw):
    for i in ['n_clusters' 'random_state', 'batch_size']:
        if i in kw:
            print(f"INFO: Ignoring {i} kmeans_kwargs")
            del kw[i]


def fix_knn_query(indices: np.ndarray, distances: np.ndarray, ref_idx: np.ndarray):
    fixed_ind, fixed_dist = indices.copy()[:, 1:], distances.copy()[:, 1:]
    # Identify positions where first index is not a self loop
    mis_idx = ~(indices[:, 0].reshape(1, -1)[0] == ref_idx)
    n_mis = mis_idx.sum()
    if n_mis > 0:
        for n, i, j, k in zip(np.where(mis_idx)[0], ref_idx[mis_idx], indices[mis_idx], distances[mis_idx]):
            p = np.where(j == i)[0]
            if len(p) > 0:
                # p is the position of self loop. We exclude this position
                p = p[0]
                j = np.array(list(j[:p]) + list(j[p+1:]))
                k = np.array(list(k[:p]) + list(k[p+1:]))
            else:
                # No self found at all. Poor recall? simply remove the last k neighbour
                j = j[:-1]
                k = k[:-1]
            fixed_ind[n] = j
            fixed_dist[n] = k
    return fixed_ind, fixed_dist


def vec_to_bow(x):
    return [[(j, k) for j, k in zip(i.indices, i.data)] for i in sparse.csr_matrix(x)]


class AnnStream:
    def __init__(self, data, k: int, n_cluster: int, reduction_method: str,
                 dims: int, loadings: np.ndarray,
                 ann_metric: str, ann_efc: int, ann_ef: int, ann_m: int, nthreads: int,
                 rand_state: int, mu: np.ndarray, sigma: np.ndarray, **kmeans_kwargs):
        self.data = data
        self.k = k
        if self.k >= self.data.shape[0]:
            self.k = self.data.shape[0]-1
        self.nClusters = max(n_cluster, 2)
        self.dims = dims
        self.loadings = loadings
        if self.dims > self.data.shape[0]:
            self.dims = self.data.shape[0]
        if self.dims is None and self.loadings is None:
            raise ValueError("ERROR: Provide either value for atleast one: 'dims' or 'loadings'")
        self.annMetric = ann_metric
        self.annEfc = ann_efc
        self.annEf = ann_ef
        if self.annEf is None:
            self.annEf = self.k * 2
        self.annM = ann_m
        if self.annM is None:
            self.annM = int(self.dims * 1.5)
        self.nthreads = nthreads
        self.randState = rand_state
        self.batchSize = self._handle_batch_size()
        self.kmeansKwargs = kmeans_kwargs
        clean_kmeans_kwargs(self.kmeansKwargs)
        self.mu = mu
        self.sigma = sigma
        self.method = reduction_method
        self.nCells, self.nFeats = self.data.shape
        self.annIdx = self._init_ann()
        self.clusterLabels: np.ndarray = np.repeat(-1, self.nCells)
        self.kmeans = self._init_kmeans()

        self.reducer = None

    def _handle_batch_size(self):
        batch_size = self.data.chunksize[0]  # Assuming all chunks are same size
        if self.dims >= batch_size:
            self.dims = batch_size-1  # -1 because we will do PCA +1
            print(f"INFO: Number of PCA components reduced to batch size of {batch_size}")
        if self.nClusters > batch_size:
            self.nClusters = batch_size
            print(f"INFO: Cluster number reduced to batch size of {batch_size}")
        return batch_size

    def _init_ann(self):
        idx = hnswlib.Index(space=self.annMetric, dim=self.dims)
        idx.init_index(max_elements=self.nCells, ef_construction=self.annEfc,
                       M=self.annM, random_seed=self.randState)
        idx.set_ef(self.annEf)
        idx.set_num_threads(1)
        return idx

    def _init_kmeans(self):
        return MiniBatchKMeans(
            n_clusters=self.nClusters, random_state=self.randState,
            batch_size=self.batchSize, **self.kmeansKwargs)

    def iter_blocks(self, msg: str = ''):
        for i in tqdm(self.data.blocks, desc=msg, total=self.data.numblocks[0]):
            yield i.compute()

    def transform_z(self, a: np.ndarray):
        return (a - self.mu) / self.sigma

    def transform_pca(self, a: np.ndarray):
        ret_val = a.dot(self.loadings)
        return ret_val

    def transform_lsi(self, a: np.ndarray):
        ret_val = a.dot(self.loadings)
        return ret_val

    def transform_ann(self, a: np.ndarray, k: int = None, self_indices: np.ndarray = None):
        if k is None:
            k = self.k
        # Adding +1 to k because first neighbour will be the query itself
        if self_indices is None:
            i, d = self.annIdx.knn_query(a, k=k)
            return i, d
        else:
            i, d = self.annIdx.knn_query(a, k=k+1)
            return fix_knn_query(i, d, self_indices)

    def estimate_partitions(self):
        temp = []
        for i in self.iter_blocks(msg='Estimating seed partitions'):
            temp.extend(self.kmeans.predict(self.reducer(i)))
        self.clusterLabels = np.array(temp)

    def _fit_pca(self):
        # We fit 1 extra PC dim than specified and then ignore the last PC.
        self._pca = IncrementalPCA(n_components=self.dims + 1, batch_size=self.batchSize)
        for i in self.iter_blocks(msg='Fitting PCA'):
            self._pca.partial_fit(self.transform_z(i), check_input=False)
        self.loadings = self._pca.components_[:-1, :].T

    def _fit_lsi(self):
        self._lsiModel = LsiModel(vec_to_bow(self.data.blocks[0].compute()), num_topics=self.dims,
                                  chunksize=self.data.chunksize[0])
        for n, i in enumerate(self.iter_blocks(msg="Fitting LSI model")):
            if n == 0:
                continue
            self._lsiModel.add_documents(vec_to_bow(i))
        self.loadings = self._lsiModel.get_topics().T

    def fit(self):
        with threadpool_limits(limits=self.nthreads):
            if self.method == 'pca':
                self.reducer = lambda x: self.transform_pca(self.transform_z(x))
            elif self.method == 'lsi':
                self.reducer = self.transform_lsi
            else:
                raise ValueError("ERROR: Unknown reduction method")
            if self.loadings is None:
                if self.method == 'pca':
                    self._fit_pca()
                elif self.method == 'lsi':
                    self._fit_lsi()
            for i in self.iter_blocks(msg='Fitting ANN'):
                a = self.reducer(i)
                self.annIdx.add_items(a)
                self.kmeans.partial_fit(a)
            self.estimate_partitions()

    def refit_kmeans(self, n_clusters: int, **kwargs):
        self.nClusters = n_clusters
        self.kmeansKwargs = kwargs
        clean_kmeans_kwargs(self.kmeansKwargs)
        self.kmeans = self._init_kmeans()
        with threadpool_limits(limits=self.nthreads):
            for i in self.iter_blocks(msg='Fitting kmeans'):
                self.kmeans.partial_fit(self.reducer(i))
            self.estimate_partitions()
