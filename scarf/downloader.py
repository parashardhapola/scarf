"""
Used to download datasets included in Scarf.

Classes:
    OSFdownloader: A class for downloading datasets from OSF.

- Methods:
    - handle_download: carry out the download of a specified dataset
    - show_available_datasets: list datasets offered through Scarf
    - fetch_dataset: Downloads datasets from online repositories and saves them in as-is format
"""

import requests
import os
import tarfile
from .logging_utils import logger
from .utils import system_call
import pandas as pd
import io

__all__ = ['show_available_datasets', 'fetch_dataset']


class OSFdownloader:
    """
        A class for downloading datasets from OSF.

        Attributes:
            projectId:
            storages:
            url:
            datasets:
            sourceFile:
            sources:

        Methods:
            get_json:
            get_all_pages:
            show_datasets:
            get_dataset_file_ids:
        """
    def __init__(self, project_id):
        self.projectId = project_id
        self.storages = ['osfstorage', 'figshare']
        self.url = f"https://api.osf.io/v2/nodes/{self.projectId}/files/"
        self.datasets, self.sourceFile = self._populate_datasets()
        self.sources = self._populate_sources()

    def get_json(self, storage, endpoint, url):
        if endpoint != '':
            endpoint = endpoint + '/'
        if url is None:
            url = self.url + f"{storage}/{endpoint}"
        return requests.get(url).json()

    def get_all_pages(self, storage, endpoint=''):
        data = []
        url = None
        while True:
            r = self.get_json(storage, endpoint, url)
            data.extend(r['data'])
            url = r['links']['next']
            if url is None:
                break
        return data

    @staticmethod
    def _process_path(node):
        return node['attributes']['path'].rstrip('/').lstrip('/')

    def _populate_datasets(self):
        datasets = {}
        source_fn = ''
        for storage in self.storages:
            for i in self.get_all_pages(storage):
                path = self._process_path(i)
                if i['attributes']['name'] == 'sources':
                    source_fn = path
                    continue
                datasets[i['attributes']['name']] = (path, storage)
        return datasets, source_fn

    def _populate_sources(self):
        source_fn = self._get_files_for_node('osfstorage', self.sourceFile)['sources.csv']
        return pd.read_csv(io.StringIO(requests.get(source_fn).text)).set_index('id').to_dict()

    def show_datasets(self):
        print('\n'.join(sorted(self.datasets.keys())))

    def _get_files_for_node(self, storage, file_id):
        base_url = f"https://files.de-1.osf.io/v1/resources/{self.projectId}/providers/"
        ret_val = {}
        for i in self.get_all_pages(storage, file_id):
            path = self._process_path(i)
            ret_val[i['attributes']['name']] = base_url + f"{storage}/{path}"
        return ret_val

    def get_dataset_file_ids(self, dataset_name):
        if dataset_name not in self.datasets:
            raise KeyError(f"ERROR: {dataset_name} was not found. "
                           f"Please choose one of the following:\n{self.show_datasets()}")
        file_id, storage = self.datasets[dataset_name]
        return self._get_files_for_node(storage, file_id)


osfd = None


def handle_download(url: str, out_fn: str) -> None:
    """
    Carry out the download of a specified dataset. Invokes appropriate command for Linux and Windows OS

    Args:
        url: URL of file to be downloaded
        out_fn: Absolute path to the file where the downloaded data is to be saved

    Returns: None

    """

    import sys

    if sys.platform == 'win32':
        cmd = 'powershell -command "(New-Object System.Net.WebClient).DownloadFile(%s, %s)"' % (
            f"'{url}'", f"'{out_fn}'")
    elif sys.platform in ['posix', 'linux']:
        cmd = f"wget --no-verbose -O {out_fn} {url}"
    else:
        raise ValueError(f"This operating system is not supported in this function. "
                         f"Please download the file manually from this URL:\n {url}\n "
                         f"Please save as: {out_fn}")
    logger.info("Download started...")
    system_call(cmd)
    logger.info(f"Download finished! File saved here: {out_fn}")


def show_available_datasets() -> None:
    """
    List datasets offered through Scarf.

    Prints the list of datasets.

    Returns:
        None
    """
    global osfd
    if osfd is None:
        osfd = OSFdownloader('zeupv')
    osfd.show_datasets()


def fetch_dataset(dataset_name: str, save_path: str = '.', as_zarr: bool = False) -> None:
    """
    Downloads datasets from online repositories and saves them in as-is format.

    Args:
        dataset_name: Name of the dataset
        save_path: Save location without name of the file
        as_zarr: If True, then a Zarr format file, if available, is downloaded instead

    Returns:
        None
    """

    zarr_ext = '.zarr.tar.gz'

    def has_zarr(entry):
        for e in entry:
            if e.endswith(zarr_ext):
                return True
        return False

    def get_zarr_entry(entry):
        for e in entry:
            if e.endswith(zarr_ext):
                return e, entry[e]
        return False, False

    global osfd
    if osfd is None:
        osfd = OSFdownloader('zeupv')

    files = osfd.get_dataset_file_ids(dataset_name)

    if as_zarr:
        if has_zarr(files) is False:
            logger.error(f"Zarr file does not exist for {dataset_name}. Nothing downloaded")
            return None

    save_dir = os.path.join(save_path, dataset_name)
    if os.path.isdir(save_dir) is False:
        os.makedirs(save_dir)

    if as_zarr:
        sp, url = get_zarr_entry(files)
        sp = os.path.abspath(os.path.join(save_dir, sp))
        handle_download(url, sp)
        logger.info(f"Extracting Zarr file for {dataset_name}")
        tar = tarfile.open(sp, "r:gz")
        tar.extractall(save_dir)
        tar.close()
    else:
        for i in files:
            if i.endswith(zarr_ext):
                continue
            sp = os.path.abspath(os.path.join(save_dir, i))
            handle_download(files[i], sp)
