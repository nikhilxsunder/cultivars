# Accessibility

cultivars is a Python library for time series econometrics. Its accessibility surface is smaller than a web application's: a text and notebook output layer, a documentation site, and a gallery of example figures. This file states what each of those does today, what has not been checked, and how to report a barrier.

This is a statement of current behavior, not a conformance claim. cultivars has not been through a formal audit against WCAG or any other standard, and the project runs no automated accessibility testing in CI.

## Table of Contents

- [Scope](#scope)
- [Library Output](#library-output)
- [Documentation Site](#documentation-site)
- [Example Gallery and Figures](#example-gallery-and-figures)
- [Known Gaps](#known-gaps)
- [Reporting a Barrier](#reporting-a-barrier)
- [Contributing Accessibly](#contributing-accessibly)

## Scope

| Surface                 | Where it lives                                              | Audited                                                     |
| ----------------------- | ----------------------------------------------------------- | ----------------------------------------------------------- |
| Result and test reports | `SummaryTable` and the `summary()` method on fitted results | Reviewed by source reading; no assistive-technology testing |
| Exceptions and warnings | `cultivars.exceptions`                                      | Reviewed by source reading                                  |
| Documentation site      | `docs/source`, built with Sphinx and `pydata-sphinx-theme`  | Partial; see below                                          |
| Example gallery         | `examples/plot_*.py`, rendered by `sphinx-gallery`          | Not audited                                                 |

The library itself draws no figures. `fan_chart` and similar functions return NumPy arrays and leave plotting to the caller, so the accessibility of any chart you build from cultivars output is in your hands. See [Plotting Your Own Results](#plotting-your-own-results) for suggestions.

## Library Output

### Reports are text first

Every fitted result's `summary()` returns a `SummaryTable`, a record that holds the structure of the report (title, metadata pairs, one coefficient table, closing notes) separately from its rendering. One record renders four ways, so you can pick the form that works with your tools:

| Renderer        | Output                   | Notes                                                                                                                              |
| --------------- | ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| `to_text()`     | Fixed-width plain text   | Only ASCII rule characters (`=` and `-`). No color, no ANSI escape codes, no box-drawing characters.                               |
| `_repr_html_()` | HTML `<div>` for Jupyter | Real `<table>`, `<thead>`, `<th>` and `<tbody>` elements. No stylesheet and no JavaScript; all styling is inline.                  |
| `to_pandas()`   | `DataFrame`              | Parameter names become the index; title, metadata and notes go to `DataFrame.attrs`, so nothing is lost. Needs the `pandas` extra. |
| `to_polars()`   | `DataFrame`              | Needs the `polars` extra.                                                                                                          |

Meaning is never carried by color alone. Significance and model assumptions appear as numbers (statistics, p-values) and as sentences in the notes block.

### If you use a screen reader or braille display

Fixed-width text tables rely on column alignment, which linear readers do not convey well. The structured routes are usually easier to navigate:

1. Call `to_pandas()` and read the frame row by row, or export it to CSV.
2. In a notebook, use the HTML repr, which exposes table headers to assistive technology.
3. Read the notes block (`SummaryTable.notes`), which holds the assumptions in plain sentences.

### Errors and warnings

Failures raise specific exception classes (`SpecificationError`, `DimensionError`, `NumericalError`, among others) with messages that state the offending value and the expected form. Messages are plain text with no color or markup. Some known exceptions to this are listed under [Known Gaps](#known-gaps).

## Documentation Site

The site is built with Sphinx and `pydata-sphinx-theme`, so it inherits that theme's baseline: semantic landmarks, a skip link, keyboard-operable navigation and responsive layout. We have not re-verified those theme properties against our own build.

What the project sets itself:

- **Color contrast.** The accent palette in `docs/source/_static/custom.css` was chosen for contrast. Computed against its background:

  | Pair                                      | Ratio  |
  | ----------------------------------------- | ------ |
  | Light-theme primary `#2e7d32` on white    | 5.13:1 |
  | Dark-theme primary `#66bb6a` on `#14181e` | 7.53:1 |
  | Footer link `#81c784` on `#262a2f`        | 7.17:1 |
  | Footer muted text `#a8adb3` on `#262a2f`  | 6.39:1 |

  All of these clear 4.5:1, the WCAG AA threshold for body text. The footer uses fixed dark bands in both themes, and its link colors were picked to hold contrast on both.

- **Light and dark themes.** A theme switcher is in the header, and code blocks use the `friendly` (light) and `monokai` (dark) Pygments styles.
- **Storage-preferences drawer.** The cookie and storage control carries `aria-label`, `aria-modal` and `aria-labelledby`, moves focus into the drawer on open, returns focus on close, closes on Escape, and defines `:focus-visible` styles. Its transitions are disabled under `prefers-reduced-motion`.
- **Math.** Equations render through MathJax (`sphinx.ext.mathjax`). The same math is present as LaTeX in the source docstrings, which are readable in a terminal with `help()`.
- **Docstrings as documentation.** The API reference is generated from Google-style docstrings, so the content is available as plain text inside Python (`help(obj)`, an IDE hover, or a language server) without the website.
- **Copyable examples.** Code blocks have a copy button, and docstring examples are doctests that run in CI, so what you copy is what was verified.

## Example Gallery and Figures

Gallery figures are produced by Matplotlib through `sphinx-gallery`.

- Figures are placed beside prose that states what they show. The result is also available as numbers in the example's code, since the plotted quantities come from printed or returned arrays.
- Some figures separate series by line style as well as color (for example, dashed lines for the proxy-identified response in example 04 and for the narrative bounds in example 03). **This is not done consistently across all ten examples**, and no figure has been checked for color-vision-deficiency legibility. Treat that as unverified.
- Alt text on gallery thumbnails on the index page is empty, so they are treated as decorative and the adjacent link text carries the meaning. Alt text on full figures comes from `sphinx-gallery` and depends on whether the example sets figure or axes titles.

### Plotting Your Own Results

Because cultivars returns arrays, you control how they are drawn. For forecast bands from `fan_chart`:

- Distinguish bands by shading level and by labeled edges, not hue alone.
- Label levels directly (for example "68%" and "90%") instead of relying on a legend color.
- Keep the quantile levels you passed (`quantiles=`) in the caption, since the array rows follow that order.
- Write the figure's conclusion in the surrounding text.

## Known Gaps

These are accurate as of this writing and are the first candidates for improvement.

| #   | Gap                                                                                                                                                                                                  | Effect                                                                                                                         |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| 1   | The HTML repr hard-codes label and note color `#555`.                                                                                                                                                | 7.46:1 on a white notebook, but 2.39:1 on a dark one. Dark-mode Jupyter users will see low-contrast labels and notes.          |
| 2   | The HTML repr tables have no `<caption>` and no `scope` on `<th>`.                                                                                                                                   | Table purpose and header association depend on the screen reader's inference.                                                  |
| 3   | No assistive-technology testing has been done on any surface.                                                                                                                                        | Everything above comes from reading source and computing contrast, not from using a screen reader, magnifier or voice control. |
| 4   | No automated checks in CI (axe, pa11y, Lighthouse or similar).                                                                                                                                       | Regressions in the docs site would not be caught.                                                                              |
| 5   | `sphinx-tippy` hover previews are enabled for in-article links.                                                                                                                                      | Not verified as keyboard-reachable or screen-reader friendly.                                                                  |
| 6   | Gallery figure legibility for color-vision deficiency is unchecked, and line-style redundancy is inconsistent.                                                                                       | Some figures may rely on hue.                                                                                                  |
| 7   | A few error messages are truncated or misleading (for example the message ending at `_substrates.py:1132`, and some raw NumPy `LinAlgError`s that escape without the package's own exception class). | Harder to act on, for every user, and worse with a screen reader because the cut-off sentence is easy to miss.                 |
| 8   | Some docstrings cite paths and names that do not exist.                                                                                                                                              | Wrong navigation targets in documentation text.                                                                                |

Gaps 7 and 8 are tracked in the maintainers' internal API and docstring findings and are slated for the pre-beta cleanup.

## Reporting a Barrier

If something in cultivars or its documentation is hard or impossible to use with your tools, please tell us. Include:

1. What you were trying to do and which page, function or output was involved.
2. The tool or setup you use (screen reader and browser, notebook front end, terminal, magnifier, and so on).
3. What happened and what you expected.

Ways to reach us:

- Open an issue using the [bug report template](https://github.com/nikhilxsunder/cultivars/issues/new/choose) and add "accessibility" to the title.
- Email the maintainer at nsunder724@gmail.com if you would rather not file publicly.

We will acknowledge a report, say whether we can reproduce it, and say what we plan to do. This is a small, maintainer-led open-source project, so we cannot promise a timeline, but accessibility reports are treated as bugs, not feature requests.

## Contributing Accessibly

For changes that touch output or documentation:

- **New renderers or report fields.** Keep meaning in text, not in color or position alone. Add the same information to `to_text()` and to the structured routes.
- **HTML output.** Use semantic elements (`<table>`, `<th>`, `<caption>`), avoid hard-coded colors that assume a light background, and emit no JavaScript.
- **Documentation.** Give every figure alt text that states its conclusion. Use heading levels in order, write descriptive link text (not "here"), and keep tables to real tabular data.
- **Gallery examples.** Set a figure or axes title, pair color with line style or markers, and state the plotted result in the prose.
- **Error messages.** Name the offending value and the expected form in a complete sentence, and raise a package exception class, not a raw library one.
- **Docs theme changes.** Check contrast of any new color pair against 4.5:1 for text and 3:1 for large text and interface components, in both light and dark themes.

Pull requests that close any item under [Known Gaps](#known-gaps) are welcome.
