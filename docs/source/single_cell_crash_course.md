---
description: Contextual background for single-cell RNA-seq analysis
---


# Single Cell Crash Course

This course attempts to give you gives you the rough contextual background behind single-cell RNA-seq analysis before you apply yourself.

A quick note on some general vocabulary used here : a *barcode* is one row of the matrix and usually means one droplet and (usually) one cell. A *feature* is one column: a gene, peak, or protein tag. *Active* means selected for analysis. A *result* means a named, stored output with a record of what producedit. See {doc}`reference/glossary` for the full definitions.



## Section 1 — Bulk RNA-seq of a Tissue is a mixtures; Single-cell keeps the distribution of each cell 

Bulk RNA-seq typically functions by grinding up an entire tissue, and reporting the average expression of genes in the sample.
The caveat is if one rare cell type turns a gene on completely while everything else stays silent, the average barely moves
and you miss it in Bulk RNA-seq. Single-cell RNA-seq attempts to solve this by isolating cells into droplets, then tagging each cell's
molecules with a barcode, sequence everything, and counts molecules per cell per gene. For example, say you have 2
T cells that have 12 copies of `CD3D` each, and1 B cell at 0; In bulk, this would result to an average of 8 'CD3D' counts; Instead, single-cell
reports `[12, 12, 0]`, the same average but with specificty now. 

The 5K PBMC dataset in {ref}`Quick start <quickstart>` holds monocytes, B cells, and several T-cell
states no that can't be identified through Bulk RNA-seq. 

In Scarf, everything downstream starts from count matrices of cells and genes, imported with a `*Reader` plus `*ToZarr` writer (see {doc}`tutorials/import_and_export`);
Once you open a store for scarf, the entire analysis can be performed, giving you one item to inspect

```python
ds = scarf.DataStore("scarf_datasets/tenx_5K_pbmc_rnaseq/data.zarr")  # from quickstart
print(ds.RNA.rawData.shape)  # (cells, genes): rows are barcodes, columns are genes
```

However it is important to consider that most single-cell distributions are sparse: many zeros of mixed origin, 
true silence plus missed capture, with a few high values, so ensure that you keep this in mind while analyzing.


## Section 2 — From droplets to a count matrix, and what each count generally means

During the wet lab process of single cell transcriptomics, cells are often isolated into droplets with barcoded beads, broken open so mRNA binds the bead, and then
reverse-transcribed and amplified with the barcode attached (Done via rt-qPCR). 
Following these steps, the droplets are then sequenced and counted per gene-barcode combination. 
Unique molecular identifiers (UMIs) tag individual molecules so PCR duplicates collapse to one count: without them, a molecule copied a thousand times would
outshout a thousand distinct molecules copied once each, and amplification noise would pose
as biology. 

Take droplet `T1` for example, which yields the read `T1-CD3D` 12 total times. In our datastore, this becomes row `T1` reading `CD3D:
12` (Gene CD3D has 12 counts). At scale this is thousands of rows by ~20K genes, over 90 percent of these rows are zeros:

```python
reader = scarf.CrH5Reader("counts.h5")  # inspect first: reader.nCells, reader.nFeatures
scarf.CrToZarr(reader, zarr_loc="data.zarr").dump()
ds = scarf.DataStore("data.zarr")  # counts now stream from disk, never fully loaded
```

It is important to note that a count is a sampled molecules, not absolute truth. Twelve counts means 12 repeated captures.
On the other hand, zero CAN BE ambiguous: silence, or a few molecules that missed capture. 
Counts also scale with depth, meaning that sequencing with twice as much depth can increase every count with zero *actual* biological change
Good practice can be considered to plot a histogram of total counts per barcode (cell ). 

```python
ds.cells.to_pandas_dataframe(
    columns=["RNA_nCounts", "RNA_nFeatures", "RNA_percentMito"]
).describe().loc[["min", "50%", "max"]]
```


## Section 3 — Technical variation can appear as true biology unless proved otherwise

Take the following thought experiment for example: 
Say you have 2 perfectly identical cells, and you sequence them with the same machine, same conditions, yet your final product has 2 different count vectors.

Capture efficiency, sequencing depth, ambient soup from burst cells, doublets sharing a barcode, batch shifts across day
or lane, and flickering sparsity all move and influence counts without touching the actual biology, and each can
impersonate an entirely different cell type during analysis. 
Regardless, with advancements in sequencing technology, these large differences occur less often and its important not to fearmonger over them, but rather be aware of what may be driving deviation in your data.

Consider how easily technical noise mimics real biology. Double T1's sequencing depth relative to T2, and its entire expression profile doubles with absolutely nothing changed in the cell. Drop ambient LYZ at a mere 1 to 2 counts into the mix, and even silent cells seem to whisper it. Real, uncorrected batches will separate slightly on a plot, and burst red blood cells will haunt clean lymphocytes with ghost counts of HBB.

To rule our issues in your analysis, always quantify depth, detected genes, and mitochondrial share per cell. You can do this by coloring embedding by sequencing depth to get a visual depicition of the data.

```python
ds.plots.embedding(layout=analysis_run["umap"], color_by="RNA_nCounts")
ds.plots.embedding(layout=analysis_run["umap"], color_by="RNA_clusters")  # the claim, then the check
```

A difference is biological when it aligns with known markers, replicates, and diverse batches. It is technical when it tracks strictly with depth or processing artifact. 

For instance, *MKI67* reads 0 in 99.4% of this dataset's 5,025 cells and a handful of counts in just 28, nearly two-thirds of them *CD3D*+ T cells: sparse, marker-consistent, a cycling state worth pursuing rather than noise.

Conversely, monocytes here total nearly double the rest, a median near 14,800 counts against roughly 7,500, on the back of *LYZ* averaging 50 with cells past 900 plus *S100A9* and *S100A8*: normalize before admiring any monocyte "upregulation." Splits that replicate across datasets and donors are biology. Splits that live exclusively in one lane are suspects.
A cluster whose markers are entirely ribosomal, *RPS27* sits in 96.8% of cells at a mean near 68, and derived from a single sample has already told you what it actually is.
Read the same three facts through the store instead of taking them asserted. The feature
table carries per-gene detection counts, and one gene column at a time streams out of the
count array:

```python
import numpy as np

feats = ds.RNA.feats.to_pandas_dataframe(columns=["names", "nCells"]).set_index("names")
print(feats.loc[["MKI67", "RPS27"]])  # 28 vs 4864 cells: sparsity vs ubiquity, one table

order = list(ds.RNA.feats.fetch_all("names"))
def gene(name):
    i = order.index(name)
    return np.asarray(ds.RNA.rawData[:, i:i + 1].compute()).ravel()  # slice, not scalar

mki, cd3 = gene("MKI67"), gene("CD3D")
print("MKI67+:", int((mki > 0).sum()), "of which CD3D+:", int(((mki > 0) & (cd3 > 0)).sum()))

nc = ds.cells.fetch_all("RNA_nCounts")
mono = gene("LYZ") >= np.percentile(gene("LYZ"), 90)
print("monocyte median vs rest:", np.median(nc[mono]), np.median(nc[~mono]))
```
Make this ordering an analytical reflex: depth and batch colorings first, cluster interpretation second, every single time. A grouping checked against confounders late is a grouping already falsely believed.


## Section 4 — Quality control is often the most significant part of your analysis

Filtering out cells during quality control (qc) is the most critical part of the analysis.
All of the work done during transcriptomics analysis depends on QC, as it aids in distinguishing between technical artificats versus true biological significant (see above).
Generally, cells with near-zero counts go, damaged cells with few genes and high mitochondrial share go, and doublets wait because they are a different problem than low quality.
Modern filtering marks rows rather than deleting them (which other softwares do!), thus thresholds stay inspectable and reversible, and cutoffs come from the tails of your own distributions, never another tissue's numbers. This allows you to try multiple different thresholds to see what fits your data the best.

Generally, work done with PBMCs  filters roughly 1000–15000 counts, 500–4000 genes, mito under 15 percent, then shows violins before and after, and those numbers are PBMC-specific. Filtering returns an immutable selection artifact and leaves live metadata untouched. This is a general guideline and potential starting point, remember, do what fits your data the best.


```python
qc_sel = ds.filter_cells(attrs=["RNA_nCounts", "RNA_nFeatures", "RNA_percentMito"],
                         lows=[1000, 500, 0], highs=[15000, 4000, 15])
# qc_sel is a cell_selection artifact: inspectable, reusable, and never a deletion.
```

Doublets instead can be addressed in 2 ways, either by removing them before the downstream analysis occurs, or marking doublets, keeping them in your final embeddings, and during visualization, then potentially remove them. Both paths share one first step that removes nothing: scoring runs after clustering, against the clustered graph, and returns an artifact of per-cell scores:

```python
doublet_run = ds.pipeline.run(..., doublets=True)  # abbreviated; see quality_control tutorial
doublets = doublet_run["doublets"]  # score artifact, not a deletion
scores = np.asarray(doublet_run.cells.fetch("doublet_score"))
pd.Series(scores, name="doublet_score").plot(kind="hist", bins=40)  # read the tail first
```

Path 2 keeps them in and looks first. Color the embedding by score and read where the
doublets live: bridges and fringes lighting up together means doublets form their own
crowd, while scattered sparks across clusters means random collisions:

```python
ds.plots.embedding(run=doublet_run, color_by="doublet_score", sort_values=True)
```

Path 1 removes by threshold once the histogram and the embedding agree. The teaching cutoff
below takes the upper 5 percent; set yours from your own tail shape, then compose the bound
with the score artifact's stored selection:

```python
cut = float(pd.Series(scores).quantile(0.95))  # teaching cutoff, not a constant
clean = ds.select_cells(doublets, high=cut, keep_bounds=True)  # downstream uses clean
```


## Section 5 — Normalization compares cells; feature selection chooses the lens.

Cells sequenced at different depths cannot have their gene expression counts compared. Scaling each profile to a common
size factor and log-transforming tames the skew so a unit of difference reads roughly as
fold-change. `Mono1` totals 50 against `B1`'s 27, and raw `GAPDH` of 10 versus 7 suggests a
difference that scaling dissolves, while `CD14` stays Mono1-specific because that
difference was real. Normalized values are more comparible values:

```python
sel = ds.snapshot_cell_selection("I")
norm = ds.run_normalization(sel, analysis_run["highly_variable_features"])
```

Most genes are uninformative: silent or constant everywhere, or pure noise. A few hundred
to a few thousand vary in ways that track identity, and highly variable gene selection models the
mean-variance trend to keep genes varying beyond expectation for their abundance. Each
default exclusion earns its place: mitochondrial patterns track damage and leakage,
ribosomal ones track depth and translational bustle, cell-cycle genes would cluster phases
instead of types, HLA patterns track donor rather than tissue biology, histones ride
technical correlated programs, and sex-linked genes split donors instead of cell states.
Across our list: `CD3D`, `MS4A1`, `CD14`, and `LYZ` stay, flat `GAPDH` goes,
damage-tracking `MT-CO1` is ruled out:

```python
hvg = ds.select_hvgs(sel, top_n=500)
ds.inspect_artifact(hvg).parameters  # what exactly defines this gene set
```

The selected set is a lens: change it and the atlas can change, which is expected rather
than a bug. A selection 80 percent ribosomal builds a depth graph in biology costume, and
normalized values over mito giants let one program set every cell's scale. Demand both
controls before moving on: housekeeping flat, lineage separation kept, known markers
surviving selection across abundances. Carry out: pass both controls or revisit the lens.


## Section 6 — PCA compresses, the graph connects, and UMAP draws the neighborhoods.

Genes move in correlated programs, so the data's real dimensionality sits far below the
gene count. PCA finds dominant axes of ranked weighted combinations; the first dozen
usually carry identity while later axes fade into noise, and the elbow makes the cutoff
visible. Our six cells compress to myeloid-versus-lymphoid then T-versus-B, with `Doub`
landing between and already hinting at mixture. Name each kept axis in words before using
it:

```python
pca = ds.run_pca(norm, dims=15, show_elbow_plot=True)
ds.inspect_artifact(pca).parameters  # name each kept axis before using it
```

In PCA space each cell's k nearest neighbors wire into a weighted graph, and this graph,
not the matrix, is what embeddings, clusters, trajectories, and imputation all consume. At
k=2, `T1` wires to `T2` and `Doub` on shared `CD3D` while `Doub`'s split loyalty shows as
edges to both lineages. Stages stay separate and persisted so alternatives branch cleanly:

```python
index = ds.build_ann_index(pca)
neighbors = ds.query_neighbors(index, k=11)
graph = ds.build_connectivity_map(neighbors)  # everything downstream consumes this
```

Certify three numbers before continuing: every active cell covered, isolates near zero,
and degree uncorrelated with depth. Tiny k shatters rare types into isolates that look
like discovery; huge k absorbs real boundaries into mush; a depth-redrawn graph is the
quietest failure because it looks healthy while measuring library size. Read the elbow the
same disciplined way: keep axes you can name in words, cut where variance flattens into
noise, and distrust any axis whose top genes tell no story.

UMAP then places cells so graph neighbors stay close, preserving local evidence while
sacrificing global geometry: adjacency reads fine, but island distances and empty space are
not measurements, and seeds reshape the drawing without touching the graph. Our cast draws
a T pair, a doublet bridging toward `B1`, and a distant `Mono1`: the bridge is honest, the
gap widths decoration. Coordinates store beside their graph:

```python
init = ds.build_embedding_initialization(pca)
umap = ds.run_umap(graph, init)  # coordinates only; the graph holds the biology
```

No biological sentence should depend on the seed, and tuning parameters until the picture
confirms the hypothesis is how this plot lies for you. The layout must agree with graph QC
and the marker heatmap first. Carry out: certify the graph, then read neighbors, never
distances.


## Section 7 — Populations are defined, named, and cleaned, in that order.

Community detection cuts the graph where edges run sparse, so dense pockets become
clusters. Resolution sets granularity, and hierarchical Paris offers a second view of the
same graph with its own cut. Nature holds no optimal partition, only partitions fit for
questions. Low resolution splits our cast into `{T1, T2, Doub, B1}` versus `{Mono1}`;
higher resolution separates T from B while `Doub` wobbles between runs, marking it as
boundary rather than population. Both methods run on one graph with every partition kept:

```python
leiden = ds.run_leiden_clustering(graph, resolution=0.5)
paris = ds.run_paris_clustering(graph)  # hierarchical second view, same graph
```

Stable blocks across methods and resolutions are populations; flickering boundaries are
hypotheses. Turn the resolution dial deliberately: sweep low to high, watch which splits
persist and which shimmer, and cross-tabulate Leiden against the Paris cut so agreement
reads as a table rather than an impression. Extra high-resolution clusters are not new
types until proven. Carry out: sweep, crosstab, then name.


Clusters become cell types by ranking genes and matching known positives and negatives:
`CD14` plus `LYZ` without `CD3D` reads monocyte, `MS4A1` without `CD3D` reads B. Mixed
signatures earn explicitly uncertain labels, because an honest unknown beats a confident
mislabel that poisons everything downstream. `Doub` ranking `CD3D` with `MS4A1` and no
clean negatives is mixed or putative doublet, never a new type. Fetch both sides of the
evidence from the stored table:

```python
labels = analysis_run.cells.fetch("clusters")
top = ds.get_markers(marker=analysis_run["markers"], group_id=labels[0],
                     min_score=0.1, min_frac_exp=0.1)  # positives AND negatives both matter
```

Doublets blend two programs, sit between populations with mixed markers and elevated
counts, and split their neighborhoods across identities. Scores simulate artificial
doublets and measure resemblance, but no score is a verdict: simulation, counts, markers,
and graph position must converge, since genuine transitional states can look similar:

```python
scores = ds.run_doublet_detection(analysis_run["clusters"],
                                  analysis_run["connectivity_map"])
# High score plus mixed markers plus bridge position: remove. Any one alone: inspect.
```

Doublets are a removal category, so report how many left and from where. Run the order of
operations as inspect, then subcluster the suspicious, then remove: a bridge carrying a
unique marker found nowhere else earns subclustering before any decision, while a "type"
that vanishes entirely on removal never was one.

## Section 8 — Ask differences of populations, counts of samples, and proof of replicates.

With an atlas built, three questions share one dataset and must never be confused:
expression shifts within a fixed population, abundance shifts of types across samples, and
definition shifts where the boundary itself moves. Within T cells, `MKI67` rising while
`CD3D` holds is a within-population shift, distinct from recruiting more T cells or
redrawing the T boundary. Groupwise tests rank candidates with correction across the gene
family, and with thousands of cells trivial shifts go significant, so effect size and
overlap lead while p-values follow. Each test variant persists for exact retrieval with
brackets read from the table, never recomputed:

```python
from scarf.plotting import CellField
res = ds.run_statistical_testing("ISG15", grouping=CellField("sample_id"), test="welch")
# Effect first: mean_1, mean_2, mean_difference in res.tables["ISG15"]. p second.
```

Declare one-sided alternatives, panels, and correction families before running, and declare
the design once for testing and plotting together so mismatches refuse to draw:

```python
from scarf.plotting import StudyDesign
design = StudyDesign(sample_by="donor_id", condition_by="disease", pair_by="pair_index")
# Same design object drives the test and the brackets; a mismatch warns and skips.
```

Composition is a sample-level question in cell-level costume: tally per sample from live
metadata, never from a pooled table, and show stacked bars per sample with points visible.
Types compete for 100 percent, so a rise in one is a fall somewhere else until proven
otherwise: always ask which type paid for the increase. A B-cell rise driven by one
treated donor is reported as exactly that:

```python
tally = ds.cells.to_pandas_dataframe(columns=["sample_id"])
tally["cluster"] = analysis_run.cells.fetch("clusters")
tally.groupby(["sample_id", "cluster"]).size()  # per-sample, never pooled
```

Condition claims with proper replicates aggregate per biological unit within type into
bulk-like profiles for bulk machinery, deliberately trading resolution for valid
inference. Six cells from two donors aggregate to an honestly weak N=2 instead of four
cells pretending at strength, and Scarf ships no replicate-aware model of its own, which
is an intentional boundary:

```python
bulk = ds.make_bulk(groups=analysis_run["clusters"], aggr_type="sum")
# bulk rows are replicates now: export to DESeq2/edgeR, not a cell-level test.
```

Resist the pseudo-replicate shortcut: randomly splitting one donor's cells into groups
produces descriptive resamples of the same cells, not independent biological replicates,
and testing them as replicates is pseudoreplication wearing a lab coat. Aggregate within
type, never across the whole mixture, or the rare signal of interest dissolves into the
average it was meant to escape. Carry out: name the N, the denominator, and the family
before running anything.


## Section 9 — Programs, states, signals, and time orderings.

Genes act in teams, and scoring a set per cell turns noisy flickers into one activity
number: the team can read confidently up while each player stays borderline. Score T
activation and `T1` with `T2` run high even where members read 0, because the rank pattern
holds, while `B1` stays low despite sharing `GAPDH`. Cite the set definition, the overlap
with measured genes, and member spot checks, never the score alone:

```python
act = ds.run_aucell(net, sel, features=universe)  # net: gene-set table, universe: all_features ref
# Then spot-check members: the team claim needs at least some players visible.
```

A program sharing 90 percent of its genes with cell cycle is measuring cell cycle, and a
five-gene set with four unmeasured genes measures nothing. Know which scorer you ran:
AUCell walks down each cell's ranked genes and measures how fast set members accumulate,
so it asks about rank recovery and tolerates dropout; WAGGR takes a weighted average over
the set, so it asks about magnitude and leans on detected values. Different questions,
different sensitivities, same duty to check members. See
{doc}`tutorials/gene_set_scoring` for the full call. Carry out: name the scorer, the
overlap, and one surviving member before believing any program.


States are graded overlays on discrete identity, scored as numbers per cell rather than
forced into new clusters: a proliferating T cell stays a T cell with a high cycle score.
`T2`'s `MKI67` against `T1`'s zero is same identity, different state, so score both and
cluster neither apart; see {doc}`tutorials/cell_cycle`. Splitting G2M off as a novel type,
or regressing cycle out when cycle is the biology, are the symmetric errors.

Cell-cell communication matches ligands in candidate senders against receptors in receivers
from prior databases. Every arrow is opportunity, not proof: dissociated co-expression
cannot show contact, direction, or causation. `Mono1`'s ligand against T-cell receptors
proposes Mono-to-T talk without saying they ever neighbored. Scarf stops at export by
design:

```python
adata = ds.to_anndata()  # active cells with labels; CCC runs externally from here
```

Write "consistent with signaling" with the missing evidence named in the same sentence,
and keep pairs whose receptor appears in 2 percent of receivers out of the main figure
unless a rule puts them there.

Trajectories order cells along a continuum where biology moves them along a path: the graph
stretches into a trail and pseudotime walks it from a declared start. Sources and sinks
supervise the orientation; no method discovers termini from nothing, and reversing the
declared source reverses the "trajectory" while the data never changes. Real pancreas work
orders progenitors toward alpha, beta, and delta fates from supplied termini with validity
keys; the full arc is {doc}`tutorials/pseudotime`, {doc}`tutorials/expression_dynamics`,
{doc}`tutorials/fate_mapping`, and {doc}`tutorials/trajectory_validation`. Demand all four
validity readings before believing: graph components connected along the claimed path,
expected markers trending monotonically, modules coherent along the axis, and fate
probabilities valid rather than leaking everywhere. Every trajectory sentence carries its
supervision, and pseudotime tracking total counts is a depth gradient until two
graph-level checks say otherwise. Carry out: components, trends, modules, validity keys,
or no trajectory sentence.

## Section 10 — Batches confound, N decides, and ten familiar failures cover the rest.

Batches shift measurements globally, and correction pulls them together while trying to
preserve type structure. Under-correction leaves batch clusters, over-correction erases
real biology, and correcting a variable confounded with condition deletes the finding while
claiming to preserve it. Our cast sequenced across two days at 20 percent offset needs day
correction to realign; with all T cells on day one and all B on day two, no method
separates type from day and the honest answer is a new experiment. Correct, then diagnose
mixing gained against structure kept, never one score alone:

```python
fixed = ds.run_harmony(analysis_run["pca"], batch_columns=["batch"])
# Harmony corrects reduced coordinates between PCA and neighbor search, where the graph is built.
# Compare uncorrected vs corrected graphs on mixing AND structure kept. Never one score.
```

Harmony edits coordinates, not counts, and it edits them blind: with no batch term for a
confounded design there is nothing innocent to remove, so check the uncorrected baseline
first and keep it in the figure. Rankings should survive new data, new seeds, and one
held-out diagnostic nobody tuned against, or the benchmark measured tuning effort instead
of methods. Carry out: baseline kept, both diagnostics quoted, seeds fixed.


The unit of inference follows the question: cells answer annotation and ranking, subjects
and runs answer whether treatment works. Thousands of cells as independent observations
manufacture significance, and neither Welch nor rank tests escape because the dependence
lives in the design, not the formula. The RA workflow averages within donors and tests 18
matched pairs, refusing 1,386 cells as replicates. Every claim carries its unit, and every
figure that matters shows both Ns:

```python
res = ds.run_statistical_testing("ISG15", grouping=CellField("sample_id"),
                                 test="wilcoxon", sample_by="donor_id",
                                 pair_by="pair_index")  # N is donors now, not cells
```

Findings that need cell N to survive are descriptive and should say so. Labels earn trust
from positives present, negatives absent, reference and method agreement, and replicate
support; build references once with diagnostics beside every transferred label:

```python
ref = ds.build_mapping_reference(neighbors)  # fixed reference; queries stay comparable
```

Compare methods on fixed data with fixed seeds, one change at a time, judged by several
independent diagnostics: `metric_ilisi` for mixing against `metric_graph_connectivity` for
structure kept, letting the pair argue instead of either ruling. Computation proposes and
independent data disposes, up the ladder from held-out donors through orthogonal assays to
perturbation, with every sentence carrying its rung. Reproducibility is provenance plus
environment plus data identity, written where the next reader trips over it, and lineage
reports turn any headline result into a diagram from counts to claim:

```python
rep = ds.lineage(analysis_run["markers"])
print(rep.to_markdown())  # counts to claim, published beside the figure it supports
```

Nearly every disaster is one of ten familiar confusions: soup called cells, doublets
called types, depth called biology, batch called condition, resolution called discovery,
means called distributions, cells called subjects, correlation called circuit, layout
distances called measurements, and parameters called defaults-that-must-be-right. Tape the
two honesty colorings, total counts and batch, to every embedding forever, add one
crosstab per clustering decision, and prosecute each presentable figure under all ten
headings. Finish by posing one narrow question on the 5K PBMC data with its population,
comparison, unit of inference, and kill criterion stated before any code runs, in three
figures maximum, and ship the strongest objection you could not dismiss alongside the
claim.

---

## Where to go next

Finished the course? The tool tutorials assume exactly what you now know. Start with
{ref}`Quick start <quickstart>`, then the complete {doc}`tutorials/scrna_seq` workflow.
Coming from another ecosystem, read {doc}`scanpy_and_seurat` first. For method-choice depth,
work through {doc}`tutorials/quality_control`, {doc}`tutorials/feature_selection`,
{doc}`tutorials/graph_construction`, and {doc}`tutorials/clustering` in order, then
{doc}`tutorials/data_organization` for the storage model that makes reverting safe.
