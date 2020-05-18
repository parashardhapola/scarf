from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import MiniBatchKMeans
import hnswlib
from tqdm import tqdm
import numpy as np
from scipy import sparse
from gensim.models import LsiModel

__all__ = ['AnnStream']


def clean_kmeans_kwargs(kw):
    for i in ['n_clusters' 'random_state', 'batch_size']:
        if i in kw:
            print(f"INFO: Ignoring {i} kmeans_kwargs")
            del kw[i]


def vec_to_bow(x):
    return [[(j, k) for j, k in zip(i.indices, i.data)] for i in sparse.csr_matrix(x)]


class AnnStream:
    def __init__(self, data, k: int, n_cluster: int, reduction_method: str,
                 dims: int, loadings: np.ndarray,
                 ann_metric: str, ann_efc: int, ann_ef: int, ann_nthreads: int,
                 rand_state: int, mu: np.ndarray, sigma: np.ndarray, **kmeans_kwargs):
        # TODO: consider gensim for LSA: https://radimrehurek.com/gensim/models/lsimodel.html
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
        self.annNthreads = ann_nthreads
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
                       M=self.dims, random_seed=self.randState)
        idx.set_ef(self.annEf)
        idx.set_num_threads(self.annNthreads)
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
        return a.dot(self.loadings)

    def transform_lsi(self, a: np.ndarray):
        return a.dot(self.loadings)

    def transform_ann(self, a: np.ndarray, k: int = None):
        if k is None:
            k = self.k
        # Adding +1 to k because first neighbour will be the query itself
        i, d = self.annIdx.knn_query(a, k=k+1)
        return i[:, 1:], d[:, 1:]   # Slicing to remove self-loop

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
        for i in self.iter_blocks(msg='Fitting kmeans'):
            self.kmeans.partial_fit(self.reducer(i))
        self.estimate_partitions()
