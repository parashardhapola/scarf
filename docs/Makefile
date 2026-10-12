# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= uv run --locked --group docs --extra extra sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
SPHINXPROJ    = Scarf

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help execute Makefile

execute:
	MPLBACKEND=module://matplotlib_inline.backend_inline $(SPHINXBUILD) -E -a -b html \
		-D nb_execution_mode=force -D nb_execution_in_temp=1 \
		-D nb_execution_timeout=1800 -D nb_execution_allow_errors=0 \
		-D nb_execution_raise_on_error=1 \
		"$(SOURCEDIR)" "$(BUILDDIR)/execute/html"
	@for notebook in "$(SOURCEDIR)"/vignettes/*.ipynb; do \
		cp "$(BUILDDIR)/execute/jupyter_execute/vignettes/$${notebook##*/}" "$$notebook" || exit; \
	done

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
