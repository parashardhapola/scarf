---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.6.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import scarf
scarf.__version__
```

```python
scarf.show_available_datasets()
```

```python
scarf.fetch_dataset('tenx_10k_pbmc_atacseq', save_path='./scarf_data')
```

```python
ls scarf_data
```
