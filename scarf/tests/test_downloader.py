import shutil
import os


def test_downloader():
    from ..downloader import osfd, OSFdownloader, show_available_datasets, fetch_dataset

    if osfd is None:
        osfd = OSFdownloader('zeupv')
    osfd.show_datasets()

    # Making sure that there are at least 15 datasets in the list
    assert len(osfd.datasets) > 15

    show_available_datasets()
    fetch_dataset('tenx_5K_pbmc_rnaseq',  os.path.join('scarf', 'tests', 'datasets'))
    shutil.rmtree(os.path.join('scarf', 'tests', 'datasets', 'tenx_5K_pbmc_rnaseq'))
