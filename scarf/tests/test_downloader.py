import shutil
import os


def test_downloader():
    from ..downloader import osfd, OSFdownloader

    if osfd is None:
        osfd = OSFdownloader('zeupv')
    # Making sure that there are at least 15 datasets in the list
    assert len(osfd.datasets) > 15


def test_show_available_datasets():
    from ..downloader import show_available_datasets

    show_available_datasets()


def test_fetch_dataset():
    from ..downloader import fetch_dataset
    fetch_dataset('tenx_5K_pbmc_rnaseq',  os.path.join('scarf', 'tests', 'datasets'))
    shutil.rmtree(os.path.join('scarf', 'tests', 'datasets', 'tenx_5K_pbmc_rnaseq'))
