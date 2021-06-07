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

__all__ = ['show_available_datasets', 'fetch_dataset']


class OSFdownloader:
    """
    A class for downloading datasets from OSF.

    Attributes:
        projectId:
        url:
        datasets:

    Methods:
        get_json:
        get_all_pages:
        show_datasets:
        get_dataset_file_ids:
    """

    def __init__(self, project_id):
        """
        Args:
            project_id: the ID of a project, e. g. zeupv
        """
        self.projectId = project_id
        self.url = f"https://api.osf.io/v2/nodes/{self.projectId}/files/osfstorage/"
        self.datasets = {}
        for i in self.get_all_pages():
            self.datasets[i['attributes']['name']] = i['id']

    def get_json(self, endpoint, url):
        if endpoint != '':
            endpoint = endpoint + '/'
        if url is None:
            url = self.url + endpoint
        return requests.get(url).json()

    def get_all_pages(self, endpoint=''):
        data = []
        url = None
        while True:
            r = self.get_json(endpoint, url)
            data.extend(r['data'])
            url = r['links']['next']
            if url is None:
                break
        return data

    def show_datasets(self):
        print('\n'.join(self.datasets.keys()))

    def get_dataset_file_ids(self, dataset_name):
        durl = 'https://files.de-1.osf.io/v1/resources/zeupv/providers/osfstorage/'
        if dataset_name not in self.datasets:
            raise KeyError(f"ERROR: {dataset_name} was not found. "
                           f"Please choose one of the following:\n{self.show_datasets()}")
        dataset_id = self.datasets[dataset_name]
        ret_val = {}
        for i in self.get_all_pages(dataset_id):
            ret_val[i['attributes']['name']] = durl + i['id']
        return ret_val


osfd = None


def handle_download(url, out_fn):
    """
    Carry out the download of a specified dataset.

    Args:
        out_fn: the file name (aka path) for the downloaded file(s)
    """
    import sys

    if sys.platform == 'win32':
        cmd = 'powershell -command "& { iwr %s -OutFile %s }"' % (url, out_fn)
    elif sys.platform in ['posix', 'linux']:
        cmd = f"wget -O {out_fn} {url}"
    else:
        raise ValueError(f"This operating system is not supported in this function. "
                         f"Please download the file manually from this URL:\n {url}\n "
                         f"Please save as: {out_fn}")
    logger.info("Download started...")
    system_call(cmd)
    logger.info(f"Download finished! File saved here: {out_fn}")


def show_available_datasets():
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
        as_zarr: If True, then a Zarr format file is downloaded instead

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
        sp = os.path.join(save_dir, sp)
        handle_download(url, sp)
        tar = tarfile.open(sp, "r:gz")
        tar.extractall(save_dir)
        tar.close()
    else:
        for i in files:
            sp = os.path.join(save_dir, i)
            handle_download(files[i], sp)
