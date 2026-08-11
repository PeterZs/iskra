"""Sphinx extension: hover tooltips for Tensor shape / dtype symbols."""

import re

from docutils import nodes
from sphinx.domains.python import PyObject, PyTypedField
from sphinx.ext.napoleon.docstring import GoogleDocstring
from sphinx.locale import _

# Make-believe Tensor[DType, [...]] symbols used in iskra docstrings.
_TOOLTIPS = {
    "S": "Number of subfaces in the mesh. "
    "Usually S is documented together with some F as point of reference. "
    "E.g., if F is the number of triangles, S could be the number of "
    "edges or vertices in the mesh.",
    "Ds": "Shape of the data payload.",
    "F": "Number of faces",
    "Fs": "Shape of a face index.",
    "FS": "Number of subfaces in a face.",
    "FSs": "Shape of (nested) subfaces in a face. E.g., a tetrahedron's "
    "side-triangles' corners (`[F, FSs]` = `[Tets, FSs]`) has `FSs=[4, 3]`.",
    "FV": "Number of vertices in a face.",
    "SV": "Number of vertices in a subface.",
    "BS": "Number of boundary subfaces.",
    "V": "Number of vertices",
    "E": "Number of edges",
    "Edges": "Number of edges",
    "Tris": "Number of triangles",
    "Tets": "Number of tetrahedra",
    "Bs": "Batch shape (multiple dimensions).",
    "Bs1": "Batch shape, arg 1 (multiple dimensions)",
    "Bs2": "Batch shape, arg 2 (multiple dimensions)",
    "Bs3": "Broadcast batch shape.",
    "B": "Batch size.",
    "B1": "Batch size, arg 1",
    "B2": "Batch size, arg 2",
    "Dim": "Ambient dimension",
    "SDim": "Simplex corner count",
    "DType": "Element dtype",
    "Float": "Floating-point dtype",
    "Complex": "Complex number dtype",
    "Int": "Integer dtype",
    "Int64": "64-bit integer dtype",
    "Bool": "Boolean dtype",
    "N": "Length / rows",
    "M": "Length / columns",
    "H": "Number of handles",
    "nnz": "Nonzero entries",
}

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _shape_role(name, rawtext, text, lineno, inliner, options=None, content=None):
    """Render :shape:`Fs` as monospace text with per-symbol <abbr> tooltips."""
    children = []
    i = 0
    while i < len(text):
        m = _IDENT.match(text, i)
        if m:
            symbol = m.group(0)
            tip = _TOOLTIPS.get(symbol)
            children.append(
                nodes.abbreviation(symbol, symbol, explanation=tip)
                if tip
                else nodes.Text(symbol)
            )
            i = m.end()
        else:
            children.append(nodes.Text(text[i]))
            i += 1
    return [nodes.inline("", "", *children, classes=["shape"])], []


def _markup_type(annotation: str) -> str:
    """Rewrite a type annotation with :class:`~torch.Tensor` and :abbr: tooltips."""
    # Shield qualified torch.Tensor so the bare "Tensor" token isn't rewritten twice.
    sentinel = "\ufffc"
    annotation = re.sub(r"\btorch\.Tensor\b", sentinel, annotation)

    parts = []
    i = 0
    while i < len(annotation):
        m = _IDENT.match(annotation, i)
        if m:
            symbol = m.group(0)
            tip = _TOOLTIPS.get(symbol)
            if tip:
                parts.append(f":abbr:`{symbol} ({tip})`")
            elif symbol == "Tensor":
                parts.append(":class:`~torch.Tensor`")
            elif symbol == "SparseTensor":
                parts.append(":class:`~iskra.sparse.SparseTensor`")
            else:
                parts.append(symbol)
            i = m.end()
        else:
            ch = annotation[i]
            # Escape brackets so docutils doesn't treat them as interpreted-text.
            parts.append(f"\\{ch}" if ch in "[]" else ch)
            i += 1
    return "".join(parts).replace(sentinel, ":class:`~torch.Tensor`")


def _markup_docstring(app, what, name, obj, options, lines):
    """Rewrite autodoc lines: type fields get tooltips; shape backticks become :shape:."""
    type_line = re.compile(r"(:(?:type\s+\S+|rtype(?:\s+\S+)?):\s*)(.*)")
    backticks = re.compile(r"(?<!:)`{1,2}([^`]+)`{1,2}")

    app.env.note_dependency(__file__)

    def shape_backticks(match):
        body = match.group(1)
        if any(tok in _TOOLTIPS for tok in _IDENT.findall(body)):
            return f":shape:`{body}`"
        return match.group(0)

    for i, line in enumerate(lines):
        if m := type_line.match(line):
            lines[i] = m.group(1) + _markup_type(m.group(2))
        else:
            lines[i] = backticks.sub(shape_backticks, line)


def _parse_returns_as_typed_fields(self, section: str) -> list[str]:
    """Napoleon params_style Returns → :returns:/:rtype: (same machinery as Args)."""
    fields = []
    for i, (name, type_, desc) in enumerate(self._consume_fields()):
        # Unique ZWSP names keep multi-returns paired; shape-tooltips.js strips them.
        anon = "\u200b" * (i + 1)
        if not type_ and name.startswith("(") and name.endswith(")"):
            type_, name = name[1:-1], anon
        elif not type_:
            type_, name = name, anon
        elif not name:
            name = anon
        fields.append((name, type_, desc))
    return self._format_docutils_params(fields, field_role="returns", type_role="rtype")


def setup(app):
    # Stock params_style formats Returns as bold inline types; emit real fields
    # so Sphinx TypedField + our type markup can handle them like Parameters.
    GoogleDocstring._parse_custom_params_style_section = _parse_returns_as_typed_fields
    PyObject.doc_field_types = [
        f
        for f in PyObject.doc_field_types
        if f.name not in ("returnvalue", "returntype")
    ]
    PyObject.doc_field_types.insert(
        0,
        PyTypedField(
            "returnvalues",
            label=_("Returns"),
            names=("returns", "return"),
            typenames=("rtype",),
            typerolename="class",
            can_collapse=True,
        ),
    )

    app.add_role("shape", _shape_role)
    app.connect("autodoc-process-docstring", _markup_docstring, priority=600)
    app.add_css_file("https://unpkg.com/tippy.js@6/animations/shift-away-subtle.css")
    app.add_js_file("https://unpkg.com/@popperjs/core@2/dist/umd/popper.min.js")
    app.add_js_file("https://unpkg.com/tippy.js@6/dist/tippy-bundle.umd.min.js")
    app.add_js_file("shape-tooltips.js")
    return {"parallel_read_safe": True, "parallel_write_safe": True}
