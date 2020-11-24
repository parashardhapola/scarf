import numpy as np
import pandas as pd

__all__ = ['fetch_dataset']


def fit_lowess(a, b, n_bins: int, lowess_frac: float) -> np.ndarray:
    from statsmodels.nonparametric.smoothers_lowess import lowess

    stats = pd.DataFrame({'a': a, 'b': b}).apply(np.log)
    bin_edges = np.histogram(stats.a, bins=n_bins)[1]
    bin_edges[-1] += 0.1  # For including last gene
    bin_idx = []
    for i in range(n_bins):
        idx = pd.Series((stats.a >= bin_edges[i]) & (stats.a < bin_edges[i + 1]))
        if sum(idx) > 0:
            bin_idx.append(list(idx[idx].index))
    bin_vals = []
    for idx in bin_idx:
        temp_stat = stats.reindex(idx)
        temp_gene = temp_stat.idxmin().b
        bin_vals.append(
            [temp_stat.b[temp_gene], temp_stat.a[temp_gene]])
    bin_vals = np.array(bin_vals).T
    bin_cor_fac = lowess(bin_vals[0], bin_vals[1], return_sorted=False,
                         frac=lowess_frac, it=100).T
    fixed_var = {}
    for bcf, indices in zip(bin_cor_fac, bin_idx):
        for idx in indices:
            fixed_var[idx] = np.e ** (stats.b[idx] - bcf)
    return np.array([fixed_var[x] for x in range(len(a))])


def rescale_array(a: np.ndarray, frac: float = 0.9) -> np.ndarray:
    """
    Performs edge trimming on values of the input vector and constraints them between within frac and 1-frac density of
    normal distribution created with the sample mean and std. dev. of a

    :param a: numeric vector
    :param frac: Value between 0 and 1.
    :return:
    """
    from scipy.stats import norm

    loc = (np.median(a) + np.median(a[::-1])) / 2
    dist = norm(loc, np.std(a))
    minv, maxv = dist.ppf(1 - frac), dist.ppf(frac)
    a[a < minv] = minv
    a[a > maxv] = maxv
    return a


def clean_array(x, fill_val: int = 0):
    """
    Remove nan and infinite values from
    :param x:
    :param fill_val:
    :return:
    """
    x = np.nan_to_num(x, copy=True)
    x[(x == np.Inf) | (x == -np.Inf)] = 0
    x[x == 0] = fill_val
    return x


def controlled_compute(arr, nthreads):
    from multiprocessing.pool import ThreadPool
    import dask

    with dask.config.set(schedular='threads', pool=ThreadPool(nthreads)):
        res = arr.compute()
    return res


def show_progress(arr, msg: str = None, nthreads: int = 1):
    from dask.diagnostics import ProgressBar

    if msg is not None:
        print(msg, flush=True)
    pbar = ProgressBar()
    pbar.register()
    res = controlled_compute(arr, nthreads)
    pbar.unregister()
    return res


def system_call(command):
    import subprocess
    import shlex

    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)
    while True:
        output = process.stdout.readline()
        if process.poll() is not None:
            break
        if output:
            print(output.strip())
    process.poll()
    return None


def handle_download(url, out_fn):
    import sys

    if sys.platform == 'win32':
        cmd = 'powershell -command "& { iwr %s -OutFile %s }"' % (url, out_fn)
    elif sys.platform in ['posix', 'linux']:
        cmd = f"wget -O {out_fn} {url}"
    else:
        raise ValueError(f"This operating system is not supported in this function. "
                         f"Please download the file manually from this URL:\n {url}\n "
                         f"Please save as: {out_fn}")
    print("INFO: Download started...", flush=True)
    system_call(cmd)
    print(f"INFO: Download finished! File saved here: {out_fn}", flush=True)


def fetch_dataset(dataset_id: str, save_path: str = None) -> None:
    """
    Downloads datasets from online repositories and saves them in as-is format

    Args:
        dataset_id: Name of the dataset
        save_path: Save location without name of the file

    Returns:

    """

    import os

    datasets = {
        'tenx_10k_pbmc_citeseq': [
            {'name': 'data.h5',
             'url': 'http://cf.10xgenomics.com/samples/cell-exp/3.0.0/pbmc_10k_protein_v3'
                     '/pbmc_10k_protein_v3_filtered_feature_bc_matrix.h5'}
        ],
        'kang_ctrl_pbmc_rnaseq': [
            {'name': 'matrix.mtx.gz',
             'url': 'https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM2560nnn/GSM2560248/suppl/'
                    'GSM2560248_2.1.mtx.gz'},
            {'name': 'features.tsv.gz',
             'url': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE96nnn/GSE96583/suppl/'
                    'GSE96583_batch2.genes.tsv.gz'},
            {'name': 'barcodes.tsv.gz',
             'url': 'https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM2560nnn/GSM2560248/suppl/'
                    'GSM2560248_barcodes.tsv.gz'},
        ],
        'kang_stim_pbmc_rnaseq': [
            {'name': 'matrix.mtx.gz',
             'url': 'https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM2560nnn/GSM2560249/suppl/'
                    'GSM2560249_2.2.mtx.gz'},
            {'name': 'features.tsv.gz',
             'url': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE96nnn/GSE96583/suppl/'
                    'GSE96583_batch2.genes.tsv.gz'},
            {'name': 'barcodes.tsv.gz',
             'url': 'https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM2560nnn/GSM2560249/suppl/'
                    'GSM2560249_barcodes.tsv.gz'},
        ],
    }
    if dataset_id not in datasets:
        dataset_list = '\n'.join(list(datasets.keys()))
        raise ValueError(f"ERROR: Dataset not found. Please choose one from the following:\n{dataset_list}\n")
    save_dir = os.path.join(save_path, dataset_id)
    if os.path.isdir(save_dir) is False:
        os.makedirs(save_dir)
    for i in datasets[dataset_id]:
        sp = os.path.join(save_dir, i['name'])
        handle_download(i['url'], sp)
