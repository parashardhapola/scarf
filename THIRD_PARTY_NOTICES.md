# Third-party notices

Scarf is licensed under the BSD 3-Clause License in the `LICENSE` file. Parts of Scarf derive from
the third-party software listed below, and Scarf's packages include one third-party program,
`sgtsne`. Each keeps its own license. The license texts of the adapted software are at the end of
this file, and the GPL-3.0 text that covers `sgtsne` is in `licenses/GPL-3.0.txt`. A later section
lists GPL-licensed packages that Scarf can use but does not install by default.

## Bundled program

### SG-t-SNE-Π

- Source: <https://github.com/fcdimitr/sgtsnepi>
- Authors: Nikos Pitsianis, Dimitris Floros, Alexandros-Stavros Iliopoulos and Xiaobai Sun
- License: GPL-3.0-only. The full text is in `licenses/GPL-3.0.txt`.
- Included as: `bin/sgtsne`, an executable built from release v1.0.1. The source code of that
  release is at <https://github.com/fcdimitr/sgtsnepi/tree/v1.0.1>.
- Used by: `DataStore.run_tsne` and `DataStore.run_unified_tsne`, which run `sgtsne` as a separate
  program and exchange data with it through files. No Scarf code links to it.
- The executable loads FFTW, METIS, the TBB memory allocator and the Cilk Plus runtime from the
  system. Scarf does not include these libraries.

## Software

### Scanpy

- Source: <https://github.com/scverse/scanpy>
- Copyright (c) 2025 scverse®; Copyright (c) 2017 F. Alexander Wolf, P. Angerer, Theis Lab
- License: BSD-3-Clause
- Used in: `scarf/feat_utils.py` (gene-set scoring, adapted from `score_genes`) and the categorical
  palettes in `scarf/plots.py`. The 10- and 20-color palettes are Tableau 10 and 20 as distributed
  with matplotlib (the 10-color set with Scanpy's colorblind substitutions), the 28-color palette
  is from Zeileis, Hornik, and Murrell (2009), Computational Statistics & Data Analysis,
  doi:10.1016/j.csda.2008.11.033, and the 102-color palette is from Tatarize's "Color Distribution
  Methodology". All four are as compiled in Scanpy.

### umap-learn

- Source: <https://github.com/lmcinnes/umap>
- Copyright (c) 2017, Leland McInnes
- License: BSD-3-Clause
- Used in: `scarf/umap.py` (a modified copy of umap-learn's `simplicial_set_embedding`, with its
  densMAP terms and fuzzy set union).

### Seurat

- Source: <https://github.com/satijalab/seurat>
- Copyright (c) 2021 Seurat authors
- License: MIT
- Used in: `scarf/feat_utils.py` (HTO demultiplexing, adapted from `HTODemux`).

### EoN (Epidemics on Networks)

- Source: <https://github.com/springer-math/Mathematics-of-Epidemics-on-Networks>
- Copyright (c) 2016 Joel C Miller
- License: MIT
- Used in: `scarf/plots.py` (`hierarchy_pos`, the tree layout of `plot_cluster_tree`).

### HoloViews

- Source: <https://github.com/holoviz/holoviews>
- Copyright (c) 2005, holoviz (https://holoviz.org)
- License: BSD-3-Clause
- Used in: `scarf/datastore/datastore.py` (the descriptions of the shading parameters of
  `DataStore.plot_layout`, from HoloViews' datashader operations).

## Optional packages that Scarf does not install by default

Scarf contains no code from these GPL-licensed packages. The default installation leaves them out,
and the extras named below install them:

- [harmonypy](https://github.com/slowkow/harmonypy) (GPL-3.0-or-later), with the `harmony` extra.
  It computes Harmony batch correction (`DataStore.make_graph(harmonize=True)`) and LISI scores
  (`DataStore.metric_lisi`).
- [leidenalg](https://github.com/vtraag/leidenalg) (GPL-3.0-or-later) and
  [igraph](https://github.com/igraph/python-igraph) (GPL-2.0-or-later), with the `leiden` extra.
  They compute Leiden clustering (`DataStore.run_leiden_clustering`).

## License texts

### Scanpy

```text
BSD 3-Clause License

Copyright (c) 2025 scverse®
Copyright (c) 2017 F. Alexander Wolf, P. Angerer, Theis Lab
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### umap-learn

```text
BSD 3-Clause License

Copyright (c) 2017, Leland McInnes
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### Seurat

```text
MIT License

Copyright (c) 2021 Seurat authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### EoN

```text
Copyright (c) 2016   Joel C Miller

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS OR SPRINGER BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### HoloViews

```text
Copyright (c) 2005, holoviz (https://holoviz.org)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

 * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the
   distribution.

 * Neither the name of HoloViz nor the names of its contributors
   may be used to endorse or promote products derived from this
   software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
