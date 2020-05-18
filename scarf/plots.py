import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Tuple
from cmocean import cm
from IPython.display import display
from holoviews.plotting import mpl as hmpl
from holoviews.operation.datashader import datashade, dynspread
import holoviews as hv
import datashader as dsh

__all__ = ['plot_qc', 'plot_mean_var', 'plot_graph_qc', 'plot_scatter', 'shade_scatter']

plt.style.use('fivethirtyeight')


def clean_axis(ax, ts=11, ga=0.4):
    ax.xaxis.set_tick_params(labelsize=ts)
    ax.yaxis.set_tick_params(labelsize=ts)
    for i in ['top', 'bottom', 'left', 'right']:
        ax.spines[i].set_visible(False)
    ax.grid(which='major', linestyle='--', alpha=ga)
    ax.figure.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    return True


def plot_graph_qc(g):
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


def plot_qc(data: pd.DataFrame, color: str = 'steelblue',
            fig_size: tuple = None, label_size: float = 10.0, title_size: float = 8.0,
            scatter_size: float = 1.0, n_rows: int = 1, max_points: int = 10000):
    n_plots = data.shape[1]
    n_rows = max(n_rows, 1)
    n_cols = max(n_plots//n_rows, 1)
    if fig_size is None:
        fig_size = (1+3*n_cols, 1+2.5*n_rows)
    fig = plt.figure(figsize=fig_size)
    for i in range(n_plots):
        val = data[data.columns[i]].values
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        sns.violinplot(val, ax=ax, linewidth=1, orient='v', alpha=0.6,
                       inner=None, cut=0, color=color)
        val_dots = val
        if len(val_dots) > max_points:
            val_dots = data[data.columns[i]].sample(n=max_points).values
        sns.stripplot(val_dots, jitter=0.4, ax=ax, orient='v',
                      s=scatter_size, color='k', alpha=0.4)
        ax.set_ylabel(data.columns[i], fontsize=label_size)
        ax.set_title('Min: %.1f, Max: %.1f, Median: %.1f' % (
                    val.min(), val.max(), int(np.median(val))), fontsize=title_size)
        clean_axis(ax)
    plt.tight_layout()
    plt.show()


def plot_mean_var(nzm: np.ndarray, fv: np.ndarray, n_cells: np.ndarray, hvg: np.ndarray,
                  ax_label_fs: float = 12, fig_size: Tuple[float, float] = (4.5, 4.0),
                  ss: Tuple[float, float] = (3, 30), cmaps: Tuple[str, str] = ('winter', 'magma_r')):
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


def scatter_make_cmap(df, cmap=None, color_key=None):
    v = df.columns[2]
    if type(df[v].dtype) == pd.CategoricalDtype:
        if color_key is None:
            if cmap is None:
                cmap = 'tab20'
            pal = sns.color_palette(cmap, n_colors=df[v].nunique()).as_hex()
            color_key = {i: j for i, j in zip(sorted(df[v].unique()), pal)}
        else:
            if type(color_key) is not dict:
                raise TypeError("ERROR: colorkey need to be of dict type. With keys as categories in the `v` column.")
            if len(set(list(color_key.keys())).union(df[v].unique())) != df[v].nunique():
                raise ValueError("ERROR: The keys in colorkey dict should contain all the categories in the v "
                                 "column of data")
        cmap = None
    else:
        if cmap is None:
            cmap = cm.deep
        color_key = None
    return cmap, color_key


def scatter_ax_labels(ax, df, fontsize: float = 12, frame_offset: float = 0.05):
    x = df.columns[0]
    y = df.columns[1]
    ax.set_xlabel(x, fontsize=fontsize)
    ax.set_ylabel(y, fontsize=fontsize)
    vmin, vmax = df[x].min(), df[x].max()
    ax.set_xlim((vmin - abs(vmin * frame_offset), vmax + abs(vmax * frame_offset)))
    vmin, vmax = df[y].min(), df[y].max()
    ax.set_ylim((vmin - abs(vmin * frame_offset), vmax + abs(vmax * frame_offset)))
    ax.set_xticks([])
    ax.set_yticks([])
    return None


def scatter_ax_legends(fig, ax, df: pd.DataFrame, color_key: dict, cmap,
                       show_ondata: bool = True, show_onside: bool = True,
                       fontsize: float = 12, n_per_col: int = 20, dummy_size: float = 0.01,
                       marker_scale: float = 70, lspacing: float = 0.1, cspacing: float = 1) -> None:
    x, y, v = df.columns
    if color_key is not None:
        centers = df.groupby(v).median().T
        for i in centers:
            if show_ondata:
                ax.text(centers[i][x], centers[i][y], i, fontsize=fontsize)
            if show_onside:
                ax.scatter([float(centers[i][x])], [float(centers[i][y])], c=color_key[i],
                           label=i, alpha=1, s=dummy_size)
        if show_onside:
            ncols = max(1, int(len(color_key)/n_per_col))
            ax.legend(ncol=ncols, loc=(1, 0), frameon=False, fontsize=fontsize,
                      markerscale=marker_scale, labelspacing=lspacing, columnspacing=cspacing)
    else:
        if df[v].nunique() < 2:
            pass
        elif fig is not None:
            cbaxes = fig.add_axes([0.2, 1, 0.6, 0.05])
            norm = mpl.colors.Normalize(vmin=0, vmax=1)
            cb = mpl.colorbar.ColorbarBase(cbaxes, cmap=cmap, norm=norm,
                                           orientation='horizontal')
            cb.set_label('Relative values', fontsize=8)
            cb.ax.xaxis.set_label_position('top')
        else:
            print("WARNING: Not plotting the colorbar because fig object was not passed")
    return None


def scatter_ax_cleanup(ax, spine_width: float = 0.5, spine_color: str = 'k',
                       displayed_sides: tuple = ('bottom', 'left')) -> None:
    for i in ['bottom', 'left', 'top', 'right']:
        spine = ax.spines[i]
        if i in displayed_sides:
            spine.set_visible(True)
            spine.set_linewidth(spine_width)
            spine.set_edgecolor(spine_color)
        else:
            spine.set_visible(False)
    ax.figure.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    ax.set_aspect('auto')
    return None


def plot_scatter(df, in_ax=None, fig=None, width: float = 6, height: float = 6,
                 cmap=None, color_key: dict = None,
                 s: float = 10, lw: float = 0.1, edge_color='k',
                 labels_kwargs: dict = None, legends_kwargs: dict = None, **kwargs):
    x, y, v = df.columns
    if in_ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(width, height))
    else:
        ax = in_ax
    cmap, color_key = scatter_make_cmap(df, cmap, color_key)
    if color_key is None:
        if df[v].nunique() == 1:
            kwargs['c'] = 'steelblue'
        else:
            df[v] = (df[v] - df[v].min()) / (df[v].max() - df[v].min())
            kwargs['c'] = df[v]
    else:
        kwargs['c'] = [color_key[x] for x in df[v].values]
    ax.scatter(df[x].values, df[y].values, s=s, lw=lw, edgecolor=edge_color, cmap=cmap, **kwargs)
    if labels_kwargs is None:
        labels_kwargs = {}
    scatter_ax_labels(ax, df, **labels_kwargs)
    if legends_kwargs is None:
        legends_kwargs = {}
    scatter_ax_legends(fig, ax, df, color_key, cmap, **legends_kwargs)
    scatter_ax_cleanup(ax)
    if in_ax is None:
        plt.show()
    else:
        return ax


def shade_scatter(df, fig_size: float = 7, width_px: int = 1000, height_px: int = 1000,
                  x_sampling: float = 0.5, y_sampling: float = 0.5,
                  spread_px: int = 1, min_alpha: int = 10, cmap=None, color_key: dict = None,
                  labels_kwargs: dict = None, legends_kwargs: dict = None):
    x, y, v = df.columns
    points = hv.Points(df, kdims=[x, y], vdims=v)
    cmap, color_key = scatter_make_cmap(df, cmap, color_key)
    if color_key is None:
        if df[v].nunique() == 1:
            agg = dsh.count(v)
        else:
            df[v] = (df[v] - df[v].min()) / (df[v].max() - df[v].min())
            agg = dsh.mean(v)
    else:
        agg = dsh.count_cat(v)
    shader = datashade(points, aggregator=agg, cmap=cmap, color_key=color_key,
                       height=height_px, width=width_px,
                       x_sampling=x_sampling, y_sampling=y_sampling, min_alpha=min_alpha)
    shader = dynspread(shader, max_px=spread_px)
    renderer = hmpl.MPLRenderer.instance()
    fig = renderer.get_plot(shader.opts(fig_inches=(fig_size, fig_size))).state
    ax = fig.gca()
    if labels_kwargs is None:
        labels_kwargs = {}
    scatter_ax_labels(ax, df, **labels_kwargs)
    if legends_kwargs is None:
        legends_kwargs = {}
    scatter_ax_legends(fig, ax, df, color_key, cmap, **legends_kwargs)
    scatter_ax_cleanup(ax)
    display(fig)
