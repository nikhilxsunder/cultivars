# Makefile for the cultivars Sphinx documentation.
#
#   make html              build into build/html; warnings are reported, not fatal
#   make strict            same build with -W: any warning fails (what CI runs with strict=true)
#   make preview           build into build/preview/<VERSION>/ with a localhost switcher
#   make serve             preview, then serve build/preview on http://localhost:$(PORT)/
#   make clean             remove the build tree and sphinx-gallery's generated sources
#   make <sphinx target>   any other sphinx-build -M target (latexpdf, linkcheck, doctest, ...)
#
# Variables can be overridden on the command line, e.g.
#   make html SPHINXOPTS="-j auto -q"
#   make serve VERSION=stable PORT=8080

SPHINXOPTS    ?= -j auto --keep-going
SPHINXBUILD   ?= sphinx-build
PYTHON        ?= python
SOURCEDIR      = source
BUILDDIR       = build

# Local preview: mirrors the live layout so the version switcher resolves.
#   $(PREVIEWDIR)/switcher.json   written by switcher.py for localhost
#   $(PREVIEWDIR)/$(VERSION)/     this build
PREVIEWDIR    ?= $(BUILDDIR)/preview
VERSION       ?= dev
PORT          ?= 8000
SITE_URL      := http://localhost:$(PORT)/

.PHONY: help html strict preview serve clean Makefile

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Warnings as errors, with --keep-going so every warning is listed before the
# build is declared failed rather than only the first.
strict:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" -W --keep-going $(SPHINXOPTS) $(O)

preview:
	@rm -rf "$(PREVIEWDIR)/$(VERSION)"
	@mkdir -p "$(PREVIEWDIR)"
	DOCS_SITE_URL="$(SITE_URL)" DOCS_VERSION="$(VERSION)" DOCS_GIT_REF="$(VERSION)" \
	    $(SPHINXBUILD) -b html $(SPHINXOPTS) $(O) "$(SOURCEDIR)" "$(PREVIEWDIR)/$(VERSION)"
	$(PYTHON) switcher.py write --base-url "$(SITE_URL)" --output "$(PREVIEWDIR)/switcher.json"
	@echo
	@echo "preview assembled: $(PREVIEWDIR)/$(VERSION)  (run 'make serve' to open it)"

serve: preview
	@echo "serving http://localhost:$(PORT)/$(VERSION)/  (Ctrl-C to stop)"
	cd "$(PREVIEWDIR)" && $(PYTHON) -m http.server $(PORT)

# sphinx-gallery and autosummary write generated sources next to the real
# ones; a stale set is the usual cause of "it works on CI but not here".
clean:
	@rm -rf "$(BUILDDIR)"
	@rm -rf "$(SOURCEDIR)/auto_examples" "$(SOURCEDIR)/gen_modules"
	@rm -rf "$(SOURCEDIR)"/*/generated

# Catch-all: route any other target to sphinx-build -M.
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
