from typing import Optional
import numpy as np
import pandas as pd
import dask.array as da
from numpy.linalg import LinAlgError
from threadpoolctl import threadpool_limits
from .utils import controlled_compute, logger, tqdmbar
from .harmony import run_harmony


__all__ = ["AnnStream", "instantiate_knn_index", "fix_knn_query"]


def instantiate_knn_index(
    space, dim, max_elements, ef_construction, M, random_seed, ef, num_threads
):
    import hnswlib

    ann_idx = hnswlib.Index(space=space, dim=dim)
    ann_idx.init_index(
        max_elements=max_elements,
        ef_construction=ef_construction,
        M=M,
        random_seed=random_seed,
    )
    ann_idx.set_ef(ef)
    ann_idx.set_num_threads(num_threads)
    return ann_idx


def fix_knn_query(indices: np.ndarray, distances: np.ndarray, ref_idx: np.ndarray):
    fixed_ind, fixed_dist = indices.copy()[:, 1:], distances.copy()[:, 1:]
    # Identify positions where first index is not a self loop
    mis_idx = indices[:, 0].reshape(1, -1)[0] != ref_idx
    n_mis = mis_idx.sum()
    if n_mis > 0:
        for n, i, j, k in zip(
            np.where(mis_idx)[0], ref_idx[mis_idx], indices[mis_idx], distances[mis_idx]
        ):
            p = np.where(j == i)[0]
            if len(p) > 0:
                # p is the position of self loop. We exclude this position
                p = p[0]
                j = np.array(list(j[:p]) + list(j[p + 1 :]))
                k = np.array(list(k[:p]) + list(k[p + 1 :]))
            else:
                # No self found at all. Poor recall? simply remove the last k neighbour
                j = j[:-1]
                k = k[:-1]
            fixed_ind[n] = j
            fixed_dist[n] = k
    return fixed_ind, fixed_dist, n_mis


class AnnStream:
    def __init__(
        self,
        data,
        k: int,
        n_cluster: int,
        reduction_method: str,
        dims: int,
        loadings: np.ndarray,
        use_for_pca: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        ann_metric: str,
        ann_efc: int,
        ann_ef: int,
        ann_m: int,
        nthreads: int,
        ann_parallel: bool,
        rand_state: int,
        do_kmeans_fit: bool,
        disable_scaling: bool,
        ann_idx,
        lsi_skip_first: bool,
        lsi_params: dict,
        harmonize: bool,
        harmonized_data: Optional[da.Array] = None,
        batches: Optional[pd.DataFrame] = None,
    ):
        self.data = data
        self.k = k
        if self.k >= self.data.shape[0]:
            self.k = self.data.shape[0] - 1
        self.nClusters = max(n_cluster, 2)
        self.dims = dims
        self.loadings = loadings
        if self.dims is None and self.loadings is None:
            raise ValueError(
                "ERROR: Provide either value for atleast one: 'dims' or 'loadings'"
            )
        self.annMetric = ann_metric
        self.annEfc = ann_efc
        self.annEf = ann_ef
        self.annM = ann_m
        self.nthreads = nthreads
        if ann_parallel:
            self.annThreads = self.nthreads
        else:
            self.annThreads = 1
        self.randState = rand_state
        self.batchSize = self._handle_batch_size()
        self.method = reduction_method
        self.nCells, self.nFeats = self.data.shape
        self.clusterLabels: np.ndarray = np.repeat(-1, self.nCells)
        self.harmonize = harmonize
        self.harmonizedData = harmonized_data
        self.batches = batches
        disable_reduction = False
        if self.dims < 1:
            disable_reduction = True
        with threadpool_limits(limits=self.nthreads):
            if self.method == "pca":
                self.mu, self.sigma = mu, sigma
                if self.loadings is None or len(self.loadings) == 0:
                    if len(use_for_pca) != self.nCells:
                        raise ValueError(
                            "ERROR: `use_for_pca` does not have sample length as nCells"
                        )
                    if disable_reduction is False:
                        self._fit_pca(disable_scaling, use_for_pca)
                else:
                    # Even though the dims might have been already adjusted according to loadings before calling
                    # AnnStream, it could still be overwritten by _handle_batch_size. Hence need to hard set it here.
                    self.dims = self.loadings.shape[1]
                    # it is okay for dimensions to be larger than batch size here because we will not fit the PCA
                if disable_scaling:
                    if disable_reduction:
                        self.reducer = lambda x: x
                    else:
                        self.reducer = lambda x: x.dot(self.loadings)
                else:
                    if disable_reduction:
                        self.reducer = lambda x: self.transform_z(x)
                    else:
                        self.reducer = lambda x: self.transform_z(x).dot(self.loadings)
            elif self.method == "lsi":
                if self.loadings is None or len(self.loadings) == 0:
                    if disable_reduction is False:
                        self._fit_lsi(lsi_skip_first, lsi_params)
                else:
                    # First dimension of LSI captures depth
                    if lsi_skip_first:
                        self.loadings = self.loadings[:, 1:]
                    self.dims = self.loadings.shape[1]
                if disable_reduction:
                    self.reducer = lambda x: x
                else:
                    self.reducer = lambda x: x.dot(self.loadings)
            elif self.method == "custom":
                if self.loadings is None or len(self.loadings) == 0:
                    logger.warning(
                        "No loadings provided for manual dimension reduction"
                    )
                else:
                    self.dims = self.loadings.shape[1]
                if disable_reduction:
                    self.reducer = lambda x: x
                else:
                    self.reducer = lambda x: x.dot(self.loadings)
            else:
                raise ValueError(f"ERROR: Unknown reduction method: {self.method}")
            if ann_idx is None:
                self.annIdx = self._fit_ann()
            else:
                self.annIdx = ann_idx
                self.annIdx.set_ef(self.annEf)
                self.annIdx.set_num_threads(self.annThreads)
            self.kmeans = self._fit_kmeans(do_kmeans_fit)

    def _handle_batch_size(self):
        if self.dims > self.data.shape[0]:
            self.dims = self.data.shape[0]
        batch_size = self.data.chunksize[0]  # Assuming all chunks are same size
        if self.dims >= batch_size:
            self.dims = batch_size - 1  # -1 because we will do PCA +1
            logger.info(
                f"Number of PCA/LSI components reduced to batch size of {batch_size}"
            )
        if self.nClusters > batch_size:
            self.nClusters = batch_size
            logger.info(f"Cluster number reduced to batch size of {batch_size}")
        return batch_size

    def iter_blocks(self, msg: str = "") -> np.ndarray:
        for i in tqdmbar(self.data.blocks, desc=msg, total=self.data.numblocks[0]):
            yield controlled_compute(i, self.nthreads)

    def transform_z(self, a: np.ndarray) -> np.ndarray:
        return (a - self.mu) / self.sigma

    def transform_ann(
        self, a: np.ndarray, k: int = None, self_indices: np.ndarray = None
    ) -> tuple:
        if k is None:
            k = self.k
        # Adding +1 to k because first neighbour will be the query itself
        if self_indices is None:
            i, d = self.annIdx.knn_query(a, k=k)
            return i, d
        else:
            i, d = self.annIdx.knn_query(a, k=k + 1)
            return fix_knn_query(i, d, self_indices)

    def _fit_pca(self, disable_scaling, use_for_pca) -> None:
        from sklearn.decomposition import IncrementalPCA

        # We fit 1 extra PC dim than specified and then ignore the last PC.
        self._pca = IncrementalPCA(
            n_components=self.dims + 1, batch_size=self.batchSize
        )
        do_sample_subset = False if use_for_pca.sum() == self.nCells else True
        s, e = 0, 0
        # We store the first block of values here. if such a case arises that we are left with less dims+1 cells to fit
        # then those cells can be added to end_reservoir for fitting. if there are no such cells then end reservoir is
        # just by itself after fitting rest of the cells. If may be the case that the first batch itself has less than
        # dims+1 cells. in that we keep adding cells to carry_over pile until it is big enough.
        end_reservoir = []
        # carry_over store cells that can yet not be added to end_reservoir ot be used for fitting pca directly.
        carry_over = []
        for i in self.iter_blocks(msg="Fitting PCA"):
            if do_sample_subset:
                e = s + i.shape[0]
                i = i[use_for_pca[s:e]]
                s = e
            if disable_scaling is False:
                i = self.transform_z(i)
            if len(carry_over) > 0:
                i = np.vstack((carry_over, i))
                carry_over = []
            if len(i) < (self.dims + 1):
                carry_over = i
                continue
            if len(end_reservoir) == 0:
                end_reservoir = i
                continue
            try:
                self._pca.partial_fit(i, check_input=False)
            except LinAlgError:
                # Add retry counter to make memory consumption doesn't escalate
                carry_over = i
        if len(carry_over) > 0:
            i = np.vstack((end_reservoir, carry_over))
        else:
            i = end_reservoir
        try:
            self._pca.partial_fit(i, check_input=False)
        except LinAlgError:
            logger.warning(
                "{i.shape[0]} samples were not used in PCA fitting due to LinAlgError",
                flush=True,
            )
        self.loadings = self._pca.components_[:-1, :].T

    def _fit_lsi(self, lsi_skip_first, lsi_params) -> None:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            from gensim.models import LsiModel
            from gensim.matutils import Dense2Corpus

        for i in ["corpus", "num_topics", "id2word", "chunksize", "dtype"]:
            if i in lsi_params:
                del lsi_params[i]
                logger.warning(
                    f"Provided parameter, {i}, for LSI model will not be used"
                )
        self._lsiModel = LsiModel(
            corpus=Dense2Corpus(
                controlled_compute(self.data.blocks[0], self.nthreads).T
            ),
            num_topics=self.dims + 1,  # +1 because first dim will be discarded
            chunksize=self.data.chunksize[0],
            id2word={x: x for x in range(self.data.shape[1])},
            **lsi_params,
        )
        for n, i in enumerate(self.iter_blocks(msg="Fitting LSI model")):
            if n == 0:
                continue
            self._lsiModel.add_documents(Dense2Corpus(i.T))
        if lsi_skip_first:
            self.loadings = self._lsiModel.get_topics().T[:, 1:]
        else:
            self.loadings = self._lsiModel.get_topics().T

    def _fit_ann(self):
        def _transform_values():
            pca_array = []
            for _i in self.iter_blocks(msg="Calculating uncorrected latent dimensions"):
                pca_array.append(self.reducer(_i))
            return np.vstack(pca_array).T

        dims = self.dims
        if dims < 1:
            dims = self.data.shape[1]

        ann_idx = instantiate_knn_index(
            self.annMetric,
            dims,
            self.nCells,
            self.annEfc,
            self.annM,
            self.randState,
            self.annEf,
            self.annThreads,
        )
        if self.harmonize:
            if self.harmonizedData is None:
                self.harmonizedData = da.from_array(
                    run_harmony(_transform_values(), self.batches).T,
                    chunks=self.data.chunksize,
                )
            for i in tqdmbar(
                self.harmonizedData.blocks,
                desc="Fitting ANN",
                total=self.harmonizedData.numblocks[0],
            ):
                ann_idx.add_items(controlled_compute(i, self.nthreads))
        else:
            for i in self.iter_blocks(msg="Fitting ANN"):
                ann_idx.add_items(self.reducer(i))
        return ann_idx

    def _fit_kmeans(self, do_ann_fit):
        from sklearn.cluster import MiniBatchKMeans

        if do_ann_fit is False:
            return None
        kmeans = MiniBatchKMeans(
            n_clusters=self.nClusters,
            random_state=self.randState,
            batch_size=self.batchSize,
            n_init=3,
        )
        temp = []
        with threadpool_limits(limits=self.nthreads):
            for i in self.iter_blocks(msg="Fitting kmeans"):
                kmeans.partial_fit(self.reducer(i))
            for i in self.iter_blocks(msg="Estimating seed partitions"):
                temp.extend(kmeans.predict(self.reducer(i)))
        self.clusterLabels = np.array(temp)
        return kmeans
