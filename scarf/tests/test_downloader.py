def test_downloader():
    from ..downloader import osfd, OSFdownloader

    if osfd is None:
        osfd = OSFdownloader('zeupv')
    # Making sure that there are at least 15 datasets in the list
    assert len(osfd.datasets) > 15


def test_show_available_datasets():
    from ..downloader import show_available_datasets

    show_available_datasets()


def test_fetch_dataset(bastidas_ponce_data):
    # TODO: check if correct file was downloaded.

    pass

