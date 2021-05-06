rm -rf build dist scarf_toolkit.egg-info
python -m build
twine upload --verbose dist/*

