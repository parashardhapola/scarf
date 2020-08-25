import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Tuple
from cmocean import cm
import cmocean


plt.style.use('fivethirtyeight')
plt.rcParams['svg.fonttype'] = 'none'


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


def plot_heatmap(cdf, fontsize: float = 10, width_factor: float = 0.03, height_factor: float = 0.02,
                 cmap=cmocean.cm.matter_r, figsize=None):
    if figsize is None:
        figsize = (cdf.shape[1]*fontsize*width_factor, fontsize*cdf.shape[0]*height_factor)
    cgx = sns.clustermap(cdf, yticklabels=cdf.index, xticklabels=cdf.columns, method='ward',
                         figsize=figsize, cmap=cmap)
    cgx.ax_heatmap.set_yticklabels(cdf.index[cgx.dendrogram_row.reordered_ind], fontsize=fontsize)
    cgx.ax_heatmap.set_xticklabels(cdf.columns[cgx.dendrogram_col.reordered_ind], fontsize=fontsize)
    cgx.ax_heatmap.figure.patch.set_alpha(0)
    cgx.ax_heatmap.patch.set_alpha(0)
    plt.show()
    return None


def plot_cluster_hierarchy(sg, clusts, width: float, lvr_factor: float,
                           min_node_size: float, node_size_expand_factor: float, cmap):
    import networkx as nx
    import EoN
    import math

    cmap = sns.color_palette(cmap, n_colors=len(set(clusts))).as_hex()
    cs = pd.Series(clusts).value_counts().to_dict()
    nc = []
    ns = []
    for i in sg.nodes():
        v = sg.nodes[i]['partition_id']
        if v != -1:
            nc.append(cmap[v - 1])
            ns.append(cs[v] * node_size_expand_factor + min_node_size)
        else:
            nc.append('#000000')
            ns.append(min_node_size)
    pos = EoN.hierarchy_pos(sg, width=width * math.pi, leaf_vs_root_factor=lvr_factor)
    new_pos = {u: (r * math.cos(theta), r * math.sin(theta)) for u, (theta, r) in pos.items()}
    nx.draw(sg, pos=new_pos, node_size=ns, node_color=nc)
    for i in sg.nodes():
        v = sg.nodes[i]['partition_id']
        if v != -1:
            plt.text(new_pos[i][0], new_pos[i][1], v)
    plt.show()
    return None


def plot_scatter(df, in_ax=None, fig=None, width: float = 6, height: float = 6,
                 default_color: str = 'steelblue', missing_color: str = 'k', colormap=None,
                 point_size: float = 10, ax_label_size: float = 12, frame_offset: float = 0.05,
                 spine_width: float = 0.5, spine_color: str = 'k', displayed_sides: tuple = ('bottom', 'left'),
                 legend_ondata: bool = True, legend_onside: bool = True,
                 legend_size: float = 12, legends_per_col: int = 20,
                 marker_scale: float = 70, lspacing: float = 0.1, cspacing: float = 1,
                 savename: str = None, scatter_kwargs: dict = None):

    def _vals_to_colors(v: pd.Series, na_c: str, cmap):
        if v.dtype.type == np.int_ or v.dtype.type == str:
            filler_val = '####&&****!@@#!#@$'
            if v.dtype.type == np.int_:
                filler_val = v.max() + 10
            fv = v.fillna(filler_val)
            if cmap is None:
                cmap = 'tab20'
            uni_vals = [x for x in fv.unique() if x != filler_val]
            pal = sns.color_palette(cmap, n_colors=len(uni_vals)).as_hex()
            pal = dict(zip(sorted(uni_vals), pal))
            pal[filler_val] = mpl.colors.to_hex(na_c)
            c = [pal[x] for x in fv]
        elif v.dtype.type == np.float_:
            v = v.fillna(0)
            if cmap is None:
                cmap = cm.deep
            pal = plt.get_cmap(cmap)
            c = [mpl.colors.to_hex(pal(x)) for x in v / v.max()]
        else:
            raise ValueError('Unknown dtype')
        return pd.Series(c, index=v.index)

    def _handle_scatter_df(c: str, na_c: str, s: float, cmap):
        d = df.convert_dtypes()
        if 'c' not in d:
            if 'vc' not in d:
                d['c'] = [c for x in d.index]
            else:
                d['c'] = _vals_to_colors(d['vc'].copy(), na_c, cmap)
        if 's' in d:
            dt = d['s'].dtype.type
            if dt != np.float_ and dt != np.int_:
                raise TypeError("ERROR: column 's' in dataframe should either be np.float_ or np.int_ type")
        else:
            d['s'] = [s for _ in d.index]
        return d

    def _handle_scatter_kwargs(sk):
        if sk is None:
            sk = {}
        if 'c' in sk:
            print('WARNING: scatter_kwarg value `c` will be ignored')
            del sk['c']
        if 's' in sk:
            print('WARNING: scatter_kwarg value `s` will be ignored')
            del sk['s']
        if 'lw' not in sk:
            sk['lw'] = 0.1
        if 'edgecolors' not in sk:
            sk['edgecolors'] = 'k'
        return sk

    def _label_axis(fs: float, fo: float):
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

    def _cleanup(sw: float, sc: str, ds: tuple) -> None:
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

    def _legends(cmap, ondata: bool, onside: bool, fontsize: float,
                 n_per_col: int, marker_scale: float, lspacing: float, cspacing: float) -> None:
        x, y = df.columns[:2]
        if 'vc' not in df:
            return None
        v = df['vc']
        if v.dtype.type == np.int_ or v.dtype.type == str:
            centers = df[[x, y, 'vc']].groupby('vc').median().T
            for i in centers:
                if ondata:
                    ax.text(centers[i][x], centers[i][y], i, fontsize=fontsize)
                if onside:
                    ax.scatter([float(centers[i][x])], [float(centers[i][y])],
                               c=df['c'][v == i].values[0],
                               label=i, alpha=1, s=0.01)
            if onside:
                n_cols = max(1, int(v.nunique() / n_per_col))
                ax.legend(ncol=n_cols, loc=(1, 0), frameon=False, fontsize=fontsize,
                          markerscale=marker_scale, labelspacing=lspacing, columnspacing=cspacing)
        else:
            if v.nunique() <= 1:
                pass
            elif fig is not None:
                cbaxes = fig.add_axes([0.2, 1, 0.6, 0.05])
                norm = mpl.colors.Normalize(vmin=0, vmax=1)
                if cmap is None:
                    cmap = cm.deep
                cb = mpl.colorbar.ColorbarBase(cbaxes, cmap=cmap, norm=norm,
                                               orientation='horizontal')
                cb.set_label('Relative values', fontsize=fontsize)
                cb.ax.xaxis.set_label_position('top')
            else:
                print("WARNING: Not plotting the colorbar because fig object was not passed")
        return None

    dim1, dim2 = df.columns[:2]
    if in_ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(width, height))
    else:
        ax = in_ax
    df = _handle_scatter_df(c=default_color, na_c=missing_color, s=point_size, cmap=colormap)
    scatter_kwargs = _handle_scatter_kwargs(sk=scatter_kwargs)
    ax.scatter(df[dim1].values, df[dim2].values, c=df['c'].values, s=df['s'].values,
               rasterized=True, **scatter_kwargs)
    _label_axis(ax_label_size, frame_offset)
    _cleanup(spine_width, spine_color, displayed_sides)
    _legends(colormap, legend_ondata, legend_onside,
             legend_size, legends_per_col, marker_scale, lspacing, cspacing)
    if in_ax is None:
        if savename:
            plt.savefig(savename, dpi=300)
        plt.show()
    else:
        return ax


# def shade_scatter(df, fig_size: float = 7, width_px: int = 1000, height_px: int = 1000,
#                   x_sampling: float = 0.2, y_sampling: float = 0.2,
#                   spread_px: int = 1, min_alpha: int = 10, cmap=None, color_key: dict = None,
#                   labels_kwargs: dict = None, legends_kwargs: dict = None, savename: str = None):
#     from holoviews.plotting import mpl as hmpl
#     from holoviews.operation.datashader import datashade, dynspread
#     import holoviews as hv
#     import datashader as dsh
#     from IPython.display import display
#
#     x, y, v = df.columns
#     points = hv.Points(df, kdims=[x, y], vdims=v)
#     cmap, color_key = scatter_make_cmap(df, cmap, color_key)
#     if color_key is None:
#         if df[v].nunique() == 1:
#             agg = dsh.count(v)
#         else:
#             df[v] = (df[v] - df[v].min()) / (df[v].max() - df[v].min())
#             agg = dsh.mean(v)
#     else:
#         agg = dsh.count_cat(v)
#     shader = datashade(points, aggregator=agg, cmap=cmap, color_key=color_key,
#                        height=height_px, width=width_px,
#                        x_sampling=x_sampling, y_sampling=y_sampling, min_alpha=min_alpha)
#     shader = dynspread(shader, max_px=spread_px)
#     renderer = hmpl.MPLRenderer.instance()
#     fig = renderer.get_plot(shader.opts(fig_inches=(fig_size, fig_size))).state
#     ax = fig.gca()
#     if labels_kwargs is None:
#         labels_kwargs = {}
#     scatter_ax_labels(ax, df, **labels_kwargs)
#     if legends_kwargs is None:
#         legends_kwargs = {}
#     scatter_ax_legends(fig, ax, df, color_key, cmap, **legends_kwargs)
#     scatter_ax_cleanup(ax)
#     display(fig)
