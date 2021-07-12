"""
Contains the code for plotting in Scarf.
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from cmocean import cm
from .logging_utils import logger

plt.rcParams['svg.fonttype'] = 'none'


# These palettes were lifted from scanpy.plotting.palettes
custom_palettes = {
    10: [
        '#1f77b4', '#ff7f0e', '#279e68', '#d62728', '#aa40fc',
        '#8c564b', '#e377c2', '#7f7f7f', '#b5bd61', '#17becf'
    ],
    20: [
        '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
        '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
        '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
        '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5'
    ],
    28: [
        "#023fa5", "#7d87b9", "#bec1d4", "#d6bcc0", "#bb7784",
        "#8e063b", "#4a6fe3", "#8595e1", "#b5bbe3", "#e6afb9",
        "#e07b91", "#d33f6a", "#11c638", "#8dd593", "#c6dec7",
        "#ead3c6", "#f0b98d", "#ef9708", "#0fcfc0", "#9cded6",
        "#d5eae7", "#f3e1eb", "#f6c4e1", "#f79cd4", '#7f7f7f',
        "#c7c7c7", "#1CE6FF", "#336600"
    ],
    102: [
        "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941",
        "#006FA6", "#A30059", "#FFDBE5", "#7A4900", "#0000A6",
        "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
        "#5A0007", "#809693", "#6A3A4C", "#1B4400", "#4FC601",
        "#3B5DFF", "#4A3B53", "#FF2F80", "#61615A", "#BA0900",
        "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA",
        "#D16100", "#DDEFFF", "#000035", "#7B4F4B", "#A1C299",
        "#300018", "#0AA6D8", "#013349", "#00846F", "#372101",
        "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2",
        "#C2FF99", "#001E09", "#00489C", "#6F0062", "#0CBD66",
        "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
        "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459",
        "#456648", "#0086ED", "#886F4C", "#34362D", "#B4A8BD",
        "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F",
        "#938A81", "#575329", "#00FECF", "#B05B6F", "#8CD0FF",
        "#3B9700", "#04F757", "#C8A1A1", "#1E6E00", "#7900D7",
        "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600",
        "#D790FF", "#9B9700", "#549E79", "#FFF69F", "#201625",
        "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
        "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98",
        "#A4E804", "#324E72"
    ]
}


def clean_axis(ax, ts=11, ga=0.4):
    """
    Cleans a given matplotlib axis.
    """
    ax.xaxis.set_tick_params(labelsize=ts)
    ax.yaxis.set_tick_params(labelsize=ts)
    for i in ['top', 'bottom', 'left', 'right']:
        ax.spines[i].set_visible(False)
    ax.grid(which='major', linestyle='--', alpha=ga)
    ax.figure.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    return True


def plot_graph_qc(g):
    # TODO: add docstring description. Is this for qc of a graph, or for plotting a qc plot of a graph?
    _, axis = plt.subplots(1, 2, figsize=(12, 4))
    ax = axis[0]
    x = np.array((g != 0).sum(axis=0))[0]
    y = pd.Series(x).value_counts().sort_index()
    ax.bar(y.index, y.values, width=0.5)
    xlim = np.percentile(x, 99.5) + 5
    ax.set_xlim((0, xlim))
    ax.set_xlabel('Node degree')
    ax.set_ylabel('Frequency')
    ax.text(xlim, y.values.max(), f"plot is clipped (max degree: {y.index.max()})",
            ha='right', fontsize=9)
    clean_axis(ax)
    ax = axis[1]
    ax.hist(g.data, bins=30)
    ax.set_xlabel('Edge weight')
    ax.set_ylabel('Frequency')
    clean_axis(ax)
    plt.tight_layout()
    plt.show()


def plot_qc(data: pd.DataFrame, color: str = 'steelblue', cmap: str = 'tab20',
            fig_size: tuple = None, label_size: float = 10.0, title_size: float = 10,
            sup_title: str = None, sup_title_size: float = 12, scatter_size: float = 1.0,
            max_points: int = 10000, show_on_single_row: bool = True):
    # TODO: add docstring description. Is this for qc of a plot, or for plotting a qc plot?
    n_plots = data.shape[1] - 1
    n_groups = data['groups'].nunique()
    if n_groups > 5 and show_on_single_row is True:
        logger.info(f"Too many groups in the plot. If you think that plot is too wide then consider turning "
                    f"`show_on_single_row` parameter to False")
    if show_on_single_row is True:
        n_rows = 1
        n_cols = n_plots
    else:
        n_rows = n_plots
        n_cols = 1
    if fig_size is None:
        fig_width = min(15, n_groups + (2 * n_cols))
        fig_height = 1 + 2.5 * n_rows
        fig_size = (fig_width, fig_height)
    fig = plt.figure(figsize=fig_size)
    grouped = data.groupby('groups')
    for i in range(n_plots):
        if data.columns[i] == 'groups':
            continue
        vals = {'g': [], 'v': []}
        for j in sorted(data['groups'].unique()):
            val = grouped.get_group(j)[data.columns[i]].values
            vals['g'].extend([j for _ in range(len(val))])
            vals['v'].extend(list(val))
        vals = pd.DataFrame(vals)
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        if n_groups == 1:
            sns.violinplot(y='v', x='g', data=vals, linewidth=1, orient='v', alpha=0.6,
                           inner=None, cut=0, color=color)
        else:
            sns.violinplot(y='v', x='g', data=vals, linewidth=1, orient='v', alpha=0.6,
                           inner=None, cut=0, palette=cmap)
        if len(vals) > max_points:
            sns.stripplot(x='g', y='v', data=vals.sample(n=max_points), jitter=0.4, ax=ax, orient='v',
                          s=scatter_size, color='k', alpha=0.4)
        else:
            sns.stripplot(x='g', y='v', data=vals, jitter=0.4, ax=ax, orient='v',
                          s=scatter_size, color='k', alpha=0.4)
        ax.set_ylabel(data.columns[i], fontsize=label_size)
        ax.set_xlabel('')
        if n_groups == 1:
            ax.set_xticklabels([])
        if data['groups'].nunique() == 1:
            ax.set_title('Median: %.1f' % (int(np.median(vals['v']))), fontsize=title_size)
        clean_axis(ax)
    fig.suptitle(sup_title, fontsize=sup_title_size)
    plt.tight_layout()
    plt.show()


def plot_mean_var(nzm: np.ndarray, fv: np.ndarray, n_cells: np.ndarray, hvg: np.ndarray,
                  ax_label_fs: float = 12, fig_size: Tuple[float, float] = (4.5, 4.0),
                  ss: Tuple[float, float] = (3, 30), cmaps: Tuple[str, str] = ('winter', 'magma_r')):
    """
    Shows a mean-variance plot.
    """
    _, ax = plt.subplots(1, 1, figsize=fig_size)
    nzm = np.log2(nzm)
    fv = np.log2(fv)
    ax.scatter(nzm[~hvg], fv[~hvg], alpha=0.6, c=n_cells[~hvg], cmap=cmaps[0], s=ss[0])
    ax.scatter(nzm[hvg], fv[hvg], alpha=0.8, c=n_cells[hvg], cmap=cmaps[1], s=ss[1], edgecolor='k', lw=0.5)
    ax.set_xlabel('Log mean non-zero expression', fontsize=ax_label_fs)
    ax.set_ylabel('Log corrected variance', fontsize=ax_label_fs)
    clean_axis(ax)
    plt.tight_layout()
    plt.show()


def plot_elbow(var_exp, figsize: Tuple[float, float] = (None, 2)):
    from kneed import KneeLocator

    x = range(len(var_exp))
    kneedle = KneeLocator(x, var_exp, S=1.0, curve="convex", direction="decreasing")
    if figsize[0] is None:
        figsize = (0.25 * len(var_exp), figsize[1])
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(x, var_exp, lw=1)
    ax.set_xticks(x)
    ax.axvline(kneedle.elbow, lw=1, c='r', label='Elbow')
    ax.set_ylabel("% Variance explained", fontsize=9)
    ax.set_xlabel("Principal components", fontsize=9)
    clean_axis(ax, ts=8)
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_heatmap(cdf, fontsize: float = 10, width_factor: float = 0.03, height_factor: float = 0.02,
                 cmap=cm.matter_r, savename: str = None, save_dpi: int = 300, figsize=None):
    """
    Shows a heatmap plot.
    """
    if figsize is None:
        figsize = (cdf.shape[1] * fontsize * width_factor, fontsize * cdf.shape[0] * height_factor)
    cgx = sns.clustermap(cdf, yticklabels=cdf.index, xticklabels=cdf.columns, method='ward',
                         figsize=figsize, cmap=cmap, rasterized=True)
    cgx.ax_heatmap.set_yticklabels(cdf.index[cgx.dendrogram_row.reordered_ind], fontsize=fontsize)
    cgx.ax_heatmap.set_xticklabels(cdf.columns[cgx.dendrogram_col.reordered_ind], fontsize=fontsize)
    cgx.ax_heatmap.figure.patch.set_alpha(0)
    cgx.ax_heatmap.patch.set_alpha(0)
    if savename:
        plt.savefig(savename, dpi=save_dpi)
    plt.show()
    return None


def _scatter_fix_type(v: pd.Series, ints_as_cats: bool) -> pd.Series:
    vt = v.dtype
    if v.nunique() == 1:
        return pd.Series(np.ones(len(v)), index=v.index).astype(np.float_)
    if vt in [np.bool_]:
        # converting first to int to handle bool
        return v.astype(np.int_).astype('category')
    if vt in [str, object] or vt.name == 'category':
        return v.astype('category')
    elif np.issubdtype(vt.type, np.integer) and ints_as_cats:
        if v.nunique() > 100:
            logger.warning("Too many categories. set force_ints_as_cats to false")
        return v.astype(np.int_).astype('category')
    else:
        return v.astype(np.float_)


def _scatter_fix_mask(v: pd.Series, mask_vals: list, mask_name: str) -> pd.Series:
    if mask_vals is None:
        mask_vals = []
    mask_vals += [np.NaN]
    iscat = False
    if v.dtype.name == 'category':
        iscat = True
        v = v.astype(object)
    # There is a bug in pandas which causes failure above 1M rows
    # v[v.isin(mask_vals)] = mask_name
    v[np.isin(v, mask_vals)] = mask_name
    if iscat:
        v = v.astype('category')
    return v


def _scatter_make_colors(v: pd.Series, cmap, color_key: Optional[dict], mask_color: str, mask_name: str):
    from matplotlib.cm import get_cmap

    na_idx = v == mask_name
    uv = v[~na_idx].unique()

    if v.dtype.name != 'category':
        if cmap is None:
            return cm.deep, None
        else:
            return get_cmap(cmap), None
    else:
        if cmap is None:
            cmap = 'custom'

    if color_key is not None:
        for i in uv:
            if i not in color_key:
                raise KeyError(f"ERROR: key {i} missing in `color_key`")
        if na_idx.sum() > 0:
            if mask_name not in color_key:
                color_key[mask_name] = mpl.colors.to_hex(mask_color)
        return None, color_key
    else:
        if cmap == 'custom':
            if len(uv) <= 10:
                pal = custom_palettes[10]
            elif len(uv) <= 20:
                pal = custom_palettes[20]
            elif len(uv) <= 30:
                pal = custom_palettes[28]
            else:
                pal = custom_palettes[102]
        else:
            pal = sns.color_palette(cmap, n_colors=len(uv)).as_hex()
        color_key = dict(zip(sorted(uv), pal))
        if na_idx.sum() > 0:
            color_key[mask_name] = mpl.colors.to_hex(mask_color)
        return None, color_key


def _scatter_cleanup(ax, sw: float, sc: str, ds: tuple) -> None:
    for i in ['bottom', 'left', 'top', 'right']:
        spine = ax.spines[i]
        if i in ds:
            spine.set_visible(True)
            spine.set_linewidth(sw)
            spine.set_edgecolor(sc)
        else:
            spine.set_visible(False)
    ax.figure.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    ax.set_aspect('auto')
    return None


def _scatter_label_axis(df, ax, fs: float, fo: float):
    x, y = df.columns[:2]
    ax.set_xlabel(x, fontsize=fs)
    ax.set_ylabel(y, fontsize=fs)
    vmin, vmax = df[x].min(), df[x].max()
    ax.set_xlim((vmin - abs(vmin * fo), vmax + abs(vmax * fo)))
    vmin, vmax = df[y].min(), df[y].max()
    ax.set_ylim((vmin - abs(vmin * fo), vmax + abs(vmax * fo)))
    ax.set_xticks([])
    ax.set_yticks([])
    return None


def _scatter_legends(df, ax, cmap, ck, ondata: bool, onside: bool, fontsize: float,
                     n_per_col: int, scale: float, ls: float, cs: float, cbs: float) -> None:
    """

    Args:
        df: dataframe
        ax: axis object
        cmap: color map
        ck: color key
        ondata: display legend over scatter plot?
        onside: display legend on side?
        fontsize: fontsize of legend text
        n_per_col: number of legends per column
        scale: scale legend marker size
        ls: line spacing
        cs: column spacing
        cbs: Cbar shrink factor

    Returns:

    """
    from matplotlib.colors import Normalize
    from matplotlib.colorbar import ColorbarBase, make_axes_gridspec

    x, y, vc = df.columns[:3]
    v = df[vc]
    cax = make_axes_gridspec(ax, location='top', shrink=cbs, aspect=25, fraction=0.1)[0]
    if v.nunique() <= 1:
        cax.set_axis_off()
        return None
    if v.dtype.name == 'category':
        ax.title.set_text(vc)
        centers = df[[x, y, vc]].groupby(vc).median().T
        for i in centers:
            if ondata:
                ax.text(centers[i][x], centers[i][y], i, fontsize=fontsize,
                        ha='center', va='center')
            if onside:
                ax.scatter([float(centers[i][x])], [float(centers[i][y])],
                           c=ck[i], label=i, alpha=1, s=0.01)
        if onside:
            n_cols = v.nunique() // n_per_col
            if v.nunique() % n_per_col > 0:
                n_cols += 1
            ax.legend(ncol=n_cols, loc=(1, 0), frameon=False, fontsize=fontsize,
                      markerscale=scale, labelspacing=ls, columnspacing=cs)
        cax.set_axis_off()
    else:
        norm = Normalize(vmin=v.min(), vmax=v.max())
        cb = ColorbarBase(cax, cmap=cmap, norm=norm, orientation='horizontal')
        cb.set_label(vc, fontsize=fontsize)
        cb.ax.xaxis.set_label_position('bottom')
        cb.ax.xaxis.set_ticks_position('top')
        cb.outline.set_visible(False)
    return None


def _make_grid(width, height, w_pad, h_pad, n_panels, n_columns):
    n_columns = np.minimum(n_panels, n_columns)
    n_rows = np.ceil(n_panels / n_columns).astype(int)
    if w_pad is None and h_pad is None:
        constrained = True
    else:
        constrained = False
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(width * n_columns, height * n_rows), squeeze=False,
                             constrained_layout=constrained)
    diff = (n_rows * n_columns) - n_panels
    while diff > 0:
        fig.delaxes(axes[n_rows - 1, n_columns - diff])
        diff -= 1
    if not constrained:
        plt.tight_layout(w_pad=w_pad, h_pad=h_pad)

    return fig, axes


def plot_scatter(dfs, in_ax=None, width: float = 6, height: float = 6,
                 default_color: str = 'steelblue', color_map=None, color_key: dict = None,
                 mask_values: list = None, mask_name: str = 'NA', mask_color: str = 'k',
                 point_size: float = 10, ax_label_size: float = 12, frame_offset: float = 0.05,
                 spine_width: float = 0.5, spine_color: str = 'k', displayed_sides: tuple = ('bottom', 'left'),
                 legend_ondata: bool = True, legend_onside: bool = True,
                 legend_size: float = 12, legends_per_col: int = 20, cbar_shrink: float = 0.6,
                 marker_scale: float = 70, lspacing: float = 0.1, cspacing: float = 1,
                 savename: str = None, dpi: int = 300, force_ints_as_cats: bool = True,
                 n_columns: int = 4, w_pad: float = 1, h_pad: float = 1, scatter_kwargs: dict = None):
    """
    Shows scatter plots. If more then one dataframe is provided it will place the scatterplots in a grid. 
    """
    from matplotlib.colors import to_hex

    def _handle_scatter_kwargs(sk):
        if sk is None:
            sk = {}
        if 'c' in sk:
            logger.warning('scatter_kwarg value `c` will be ignored')
            del sk['c']
        if 's' in sk:
            logger.warning('scatter_kwarg value `s` will be ignored')
            del sk['s']
        if 'lw' not in sk:
            sk['lw'] = 0.1
        if 'edgecolors' not in sk:
            sk['edgecolors'] = 'k'
        return sk

    if len(dfs) > 1:
        if in_ax is not None:
            logger.warning(f"'in_ax' will not be used as multiple attributes will be plotted. Using internal grid"
                           f"layout")
        fig, axs = _make_grid(width, height, w_pad, h_pad, len(dfs), n_columns)
    else:
        if in_ax is None:
            _, axs = plt.subplots(1, 1, figsize=(width, height), squeeze=False)
        else:
            axs = in_ax

    for i, df in enumerate(dfs):
        # noinspection DuplicatedCode
        dim1, dim2, vc = df.columns[:3]
        v = _scatter_fix_mask(df[vc].copy(), mask_values, mask_name)
        v = _scatter_fix_type(v, force_ints_as_cats)
        df[vc] = v
        col_map, col_key = _scatter_make_colors(v, color_map, color_key,
                                                mask_color, mask_name)
        if v.dtype.name == 'category':
            df['c'] = [col_key[x] for x in v]
        else:
            if v.nunique() == 1:
                df['c'] = [default_color for _ in v]
            else:
                v = v.copy().fillna(0)
                mmv = (v - v.min()) / (v.max() - v.min())
                df['c'] = [to_hex(col_map(x)) for x in mmv]
        if 's' not in df:
            df['s'] = [point_size for _ in df.index]
        scatter_kwargs = _handle_scatter_kwargs(sk=scatter_kwargs)
        ax = axs[int(i / n_columns), i % n_columns]
        ax.scatter(df[dim1].values, df[dim2].values, c=df['c'].values, s=df['s'].values,
                   rasterized=True, **scatter_kwargs)
        _scatter_label_axis(df, ax, ax_label_size, frame_offset)
        _scatter_cleanup(ax, spine_width, spine_color, displayed_sides)
        _scatter_legends(df, ax, col_map, col_key, legend_ondata, legend_onside,
                         legend_size, legends_per_col, marker_scale, lspacing, cspacing, cbar_shrink)

    if in_ax is None:
        if savename:
            plt.savefig(savename, dpi=dpi, bbox_inches='tight')
        plt.show()
    else:
        return axs


def shade_scatter(dfs, in_ax=None, figsize: float = 6, pixels: int = 1000, sampling: float = 0.1,
                  spread_px: int = 1, spread_threshold: float = 0.2, min_alpha: int = 10,
                  color_map=None, color_key: dict = None,
                  mask_values: list = None, mask_name: str = 'NA', mask_color: str = 'k',
                  ax_label_size: float = 12, frame_offset: float = 0.05,
                  spine_width: float = 0.5, spine_color: str = 'k', displayed_sides: tuple = ('bottom', 'left'),
                  legend_ondata: bool = True, legend_onside: bool = True,
                  legend_size: float = 12, legends_per_col: int = 20, cbar_shrink: float = 0.6,
                  marker_scale: float = 70, lspacing: float = 0.1, cspacing: float = 1,
                  savename: str = None, dpi: int = 300, force_ints_as_cats: bool = True,
                  n_columns: int = 4, w_pad: float = None, h_pad: float = None):
    """
    Shows shaded scatter plots. If more then one dataframe is provided it will place the scatterplots in a grid. 
    """
    import datashader as dsh
    from IPython.display import display
    from datashader.mpl_ext import dsshow
    import datashader.transfer_functions as tf
    from functools import partial

    if len(dfs) > 1:
        if in_ax is not None:
            logger.warning(f"'in_ax' will not be used as multiple attributes will be plotted. Using internal grid"
                           f"layout")
        fig, axs = _make_grid(figsize, figsize, w_pad, h_pad, len(dfs), n_columns)
    else:
        if in_ax is None:
            _, axs = plt.subplots(1, 1, figsize=(figsize, figsize), squeeze=False)
        else:
            axs = in_ax

    for i, df in enumerate(dfs):
        dim1, dim2, vc = df.columns[:3]
        v = _scatter_fix_mask(df[vc].copy(), mask_values, mask_name)
        v = _scatter_fix_type(v, force_ints_as_cats)
        df[vc] = v
        col_map, col_key = _scatter_make_colors(v, color_map, color_key,
                                                mask_color, mask_name)
        if v.dtype.name == 'category':
            agg = dsh.count_cat(vc)
        else:
            if v.nunique() == 1:
                agg = dsh.count(vc)
            else:
                agg = dsh.mean(vc)

        ax = axs[int(i / n_columns), i % n_columns]
        artist = dsshow(df, dsh.Point(dim1, dim2), aggregator=agg, norm='eq_hist',
                        color_key=col_key, cmap=col_map, alpha_range=(min_alpha, 255),
                        shade_hook=partial(tf.dynspread, threshold=spread_threshold, max_px=spread_px),
                        plot_height=pixels, plot_width=pixels, aspect='equal', width_scale=1, height_scale=1,
                        ax=ax)

        _scatter_label_axis(df, ax, ax_label_size, frame_offset)
        _scatter_cleanup(ax, spine_width, spine_color, displayed_sides)
        _scatter_legends(df, ax, col_map, col_key, legend_ondata, legend_onside,
                         legend_size, legends_per_col, marker_scale, lspacing, cspacing, cbar_shrink)

    if savename:
        plt.savefig(savename, dpi=dpi, bbox_inches='tight')
    if in_ax is not None:
        return axs


def _draw_pie(ax, dist, colors, xpos, ypos, size):
    # https://stackoverflow.com/questions/56337732/how-to-plot-scatter-pie-chart-using-matplotlib
    cumsum = np.cumsum(dist)
    cumsum = cumsum / cumsum[-1]
    pie = [0] + cumsum.tolist()
    for r1, r2, c in zip(pie[:-1], pie[1:], colors):
        angles = np.linspace(2 * np.pi * r1, 2 * np.pi * r2)
        x = [0] + np.cos(angles).tolist()
        y = [0] + np.sin(angles).tolist()
        xy = np.column_stack([x, y])
        ax.scatter([xpos], [ypos], marker=xy, s=size, c=c)


def plot_cluster_hierarchy(sg, clusts, color_values=None, force_ints_as_cats: bool = True,
                           width: float = 2, lvr_factor: float = 0.5, vert_gap: float = 0.2,
                           min_node_size: float = 10, node_size_multiplier: float = 1e4, node_power: float = 1,
                           root_size: float = 100, non_leaf_size: float = 10,
                           show_labels: bool = False, fontsize=10,
                           root_color: str = '#C0C0C0', non_leaf_color: str = 'k',
                           cmap: str = None, color_key: bool = None, edgecolors: str = 'k',
                           edgewidth: float = 1, alpha: float = 0.7, figsize=(5, 5), ax=None, show_fig: bool = True,
                           savename: str = None, save_dpi=300):
    """
    Shows a plot showing cluster hierarchy.

    Returns:
        If requested (with parameter `show_fig`) a matplotlib Axes object containing the plot
        (which is the modified `ax` parameter if given).
    """
    import networkx as nx
    import EoN
    import math
    from matplotlib.colors import to_hex

    if color_values is None:
        color_values = pd.Series(clusts)
        using_clust_for_colors = True
    else:
        color_values = pd.Series(color_values)
        using_clust_for_colors = False
    color_values = _scatter_fix_type(color_values, force_ints_as_cats)
    cmap, color_key = _scatter_make_colors(color_values, cmap, color_key,
                                           'k', 'longdummyvaluesofh3489hfpiqehdcbla')
    pos = EoN.hierarchy_pos(sg, width=width * math.pi, leaf_vs_root_factor=lvr_factor, vert_gap=vert_gap)
    new_pos = {u: (r * math.cos(theta), r * math.sin(theta)) for u, (theta, r) in pos.items()}

    if color_key is None:
        cluster_values = pd.DataFrame({'clusters': clusts, 'v': color_values}).groupby('clusters').mean()['v']
        mmv: pd.Series = (cluster_values - cluster_values.min()) / (cluster_values.max() - cluster_values.min())
        color_key = {k: to_hex(cmap(v)) for k, v in mmv.to_dict().items()}
    else:
        cluster_values = None

    cs = pd.Series(clusts).value_counts()
    cs = (node_size_multiplier * ((cs / cs.sum()) ** node_power)).to_dict()
    nc, ns = [], []

    for i in sg.nodes():
        if 'partition_id' in sg.nodes[i]:
            clust_id = sg.nodes[i]['partition_id']
            if cluster_values is not None or using_clust_for_colors:
                nc.append(color_key[clust_id])
                ns.append(max(cs[clust_id], min_node_size))
            else:
                nc.append('white')
                ns.append(0)
        else:
            if sg.nodes[i]['nleaves'] == len(clusts):
                nc.append(root_color)
                ns.append(root_size)
            else:
                nc.append(non_leaf_color)
                ns.append(non_leaf_size)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    nx.draw(sg, pos=new_pos, node_size=ns, node_color=nc, ax=ax, edgecolors=edgecolors, alpha=alpha,
            linewidths=edgewidth)

    if cluster_values is None and using_clust_for_colors is False:
        for i in sg.nodes():
            if 'partition_id' in sg.nodes[i]:
                clust_id = sg.nodes[i]['partition_id']
                idx = clusts == clust_id
                counts = color_values[idx].value_counts()
                _draw_pie(ax, counts.values, [color_key[x] for x in counts.index],
                          new_pos[i][0], new_pos[i][1], max(cs[clust_id], min_node_size))

    if show_labels:
        for i in sg.nodes():
            if 'partition_id' in sg.nodes[i]:
                clust_id = sg.nodes[i]['partition_id']
                ax.text(new_pos[i][0], new_pos[i][1], clust_id, fontsize=fontsize, ha='center', va='center')
    if savename:
        plt.savefig(savename, dpi=save_dpi)
    if show_fig:
        plt.show()
    else:
        return ax
