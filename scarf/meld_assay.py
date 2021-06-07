"""
- Methods:
    - meld_assay:
    - make_bed_from_gff: create pybedtools object for genes from a GFF file
"""
# TODO: add description in docstring

from tqdm import tqdm
from typing import List, Dict
import pandas as pd
import numpy as np
from .writers import create_zarr_count_assay
from .utils import controlled_compute
from .logging_utils import logger

__all__ = ['meld_assay', 'make_bed_from_gff']


def make_bed_from_gff(gff: str, up_offset: int = 2000,
                      valid_ids: List[str] = None, flavour: str = 'body'):
    """Create pybedtools object for genes from a GFF file. Gene coordinates are promoter extended."""
    try:
        from pybedtools import BedTool
    except ImportError:
        raise ImportError("pybedtools is not installed. Check out this link to install"
                          " https://daler.github.io/pybedtools/main.html#install-via-conda")
    out = []
    ignored_genes = 0
    unknown_ids = 0
    if valid_ids is not None:
        valid_ids = {x: None for x in valid_ids}
    with open(gff) as h:
        # Testing whether first 5 lines are comment lines
        for i in range(5):
            l = next(h)
            if l[0] != '#':
                logger.warning(f"line num {i} is not comment line", flush=True)
        for l in tqdm(h):
            c = l.split('\t')
            if c[2] != 'gene':
                continue
            a = [x.split(' ') for x in c[8].rstrip('\n').split('; ')]
            a = {x: y.strip('"') for x, y in a}
            if 'gene_id' not in a:
                unknown_ids += 1
                continue
            if valid_ids is not None and a['gene_id'] not in valid_ids:
                ignored_genes += 1
                continue
            # Fetch start and end coordinate
            s, e = int(c[3]), int(c[4])
            if flavour == 'body':
                if c[6] == '+':
                    s = s - up_offset
                    s = max(s, 0)
                elif c[6] == '-':
                    e = e + up_offset
                else:
                    raise ValueError('ERROR: Unsupported symbol for strand')
            elif flavour == 'promoter':
                if c[6] == '+':
                    e = s + up_offset
                    s = s - up_offset
                    s = max(s, 0)
                elif c[6] == '-':
                    s = e - up_offset
                    s = max(s, 0)
                    e = e + up_offset
                else:
                    raise ValueError('ERROR: Unsupported symbol for strand')
            else:
                raise ValueError('ERROR: `flavour` can either be `body` or `promoter`')
            if c[0].startswith('chr'):
                chrom = c[0]
            else:
                chrom = f'chr{c[0]}'
            if 'gene_name' in a:
                gn = a['gene_name']
            else:
                gn = a['gene_id']
            o = '\t'.join([chrom, str(s), str(e), a['gene_id'], gn, c[6]])
            out.append(o)
    logger.info(f"{len(out)} genes found in the GFF file")
    logger.info(f"{ignored_genes} genes were ignored as they were not present in the valid_ids")
    logger.info(f"{unknown_ids} genes were ignored as they did not have gene_id column")
    return BedTool('\n'.join(out), from_string=True)


def _create_bed_from_coord_ids(ids: List[str], invalid_ids: Dict[str, None]):
    """convert list of 'chr:start-end' format strings to pybedtools object"""
    try:
        # noinspection PyUnresolvedReferences
        from pybedtools import BedTool
    except ImportError:
        raise ImportError("ERROR: pybedtools is not installed")

    out = []
    for i in ids:
        if i in invalid_ids:
            continue
        j = i.split(':')
        o = [j[0], j[1].split('-')[0], j[1].split('-')[1], i]
        out.append('\t'.join(o))
    return BedTool('\n'.join(out), from_string=True)


def _get_merging_map(a, b, b_name_pos: int = 7) -> dict:
    """Intersect BED files to obtain a dict with b names as keys and
       overlapped a names for each b in values as list
    """
    joined = a.intersect(b, wao=True)
    res = {x.name: [] for x in b}
    for i in joined:
        if i[b_name_pos] != '.':
            res[i[b_name_pos]].append(i.name)
    return res


def _convert_ids_to_idx(ids: pd.Series, cross_id_map: dict) -> dict:
    """
    Convert ids to indices from scarf feats table.
    """
    ref = {v: k for k, v in ids.to_dict().items()}
    idx = {x: [] for x in cross_id_map}
    null_ids = 0
    for i in cross_id_map:
        if len(cross_id_map[i]) > 0:
            idx[i] = [ref[x] for x in cross_id_map[i]]
        else:
            null_ids += 1
    logger.info(f"{null_ids}/{len(cross_id_map)} ids did not have a cross id")
    return idx


def _create_counts_mat(assay, out_store, feat_order: list, cross_idx_map: dict, nthreads: int) -> None:
    c_pos_start = 0
    for a in tqdm(assay.rawData.blocks, total=assay.rawData.numblocks[0]):
        a = controlled_compute(a, nthreads)
        b = np.zeros((a.shape[0], len(feat_order)))
        c_pos_end = c_pos_start + a.shape[0]
        for n, i in enumerate(feat_order):
            if i in cross_idx_map and len(cross_idx_map[i]) > 0:
                b[:, n] = a[:, cross_idx_map[i]].sum(axis=1)
        out_store[c_pos_start:c_pos_end, :] = b
        c_pos_start = c_pos_end
    return None


def meld_assay(assay, reference_bed, out_name: str, nthreads: int,
               peaks_col: str = 'ids', ignore_ids: List[str] = None):
    # TODO: add docstring
    if ignore_ids is None:
        ignore_ids = {}
    else:
        ignore_ids = {x: None for x in ignore_ids}
    peaks_bed = _create_bed_from_coord_ids(assay.feats.fetch_all(peaks_col), invalid_ids=ignore_ids)
    cross_idx_map = _convert_ids_to_idx(pd.Series(assay.feats.fetch_all(peaks_col)),
                                        _get_merging_map(peaks_bed, reference_bed))
    feat_ids = [x[3] for x in reference_bed]
    feat_names = [x[4] for x in reference_bed]
    g = create_zarr_count_assay(z=assay.z['/'], assay_name=out_name, chunk_size=assay.rawData.chunksize,
                                n_cells=assay.rawData.shape[0], feat_ids=feat_ids, feat_names=feat_names)
    _create_counts_mat(assay=assay, out_store=g, feat_order=feat_ids, cross_idx_map=cross_idx_map, nthreads=nthreads)
