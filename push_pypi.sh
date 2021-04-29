file="build"
if [ -f "$file" ] ; then
    rm -r "$file"
fi
file="dist"
if [ -f "$file" ] ; then
    rm -r "$file"
fi
file="scarf_toolkit.egg-info"
if [ -f "$file" ] ; then
    rm -r "$file"
fi

python -m build
twine upload --verbose dist/*
