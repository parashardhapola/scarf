python setup.py sdist bdist_wheel
cd dist
twine upload ./*
