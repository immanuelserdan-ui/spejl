"""Resolve editable text placements in page and Form XObject content order."""

from dataclasses import dataclass
from io import BytesIO

import pikepdf
import pymupdf

IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def compose(left, right):
    a, b, c, d, e, f = left
    A, B, C, D, E, F = right
    return (
        a * A + c * B,
        b * A + d * B,
        a * C + c * D,
        b * C + d * D,
        a * E + c * F + e,
        b * E + d * F + f,
    )


def op(name, operands=()):
    return pikepdf.ContentStreamInstruction(operands, pikepdf.Operator(name))


@dataclass
class Placement:
    instructions: list
    index: int
    ctm: tuple

    @property
    def matrix(self):
        return tuple(map(float, self.instructions[self.index].operands))

    def set_matrix(self, matrix):
        self.instructions[self.index] = op("Tm", matrix)


class TextPage:
    """One private page copy, with an independent stream per Form invocation."""

    def __init__(self, pdf, source):
        self.pdf = pdf
        pdf.pages.append(source)
        self.page = pdf.pages[0]
        self.placements = []
        self.streams = []
        page_stream = pdf.make_stream(b"")
        instructions = list(pikepdf.parse_content_stream(self.page))
        self.page.Contents = page_stream
        self._walk(
            page_stream,
            instructions,
            self.page.obj,
            self.page.obj.get("/Resources", {}),
            IDENTITY,
            0.0,
            0,
            frozenset(),
        )

    def _walk(self, stream, instructions, owner, resources, ctm, leading, mode, ancestors):
        output = []
        self.streams.append((stream, output))
        # Only the XObject name table is mutated. Fonts, clipping boxes,
        # transparency groups and all other stream resources remain intact.
        original_objects = resources.get("/XObject", {})
        objects = pikepdf.Dictionary(original_objects)
        local_resources = pikepdf.Dictionary(resources)
        local_resources.XObject = objects
        owner.Resources = local_resources
        invoked = set()
        stack = []
        line = None
        positioned = False
        shown = False
        for instruction in instructions:
            args, operator = instruction
            name = str(operator)
            if name == "q":
                stack.append((ctm, leading, mode))
            elif name == "Q" and stack:
                ctm, leading, mode = stack.pop()
            elif name == "cm":
                ctm = compose(ctm, tuple(map(float, args)))
            elif name == "Do":
                form = original_objects.get(str(args[0]))
                if form is not None and form.get("/Subtype") == "/Form":
                    if form.objgen in ancestors:
                        raise ValueError("Recursive PDF Form XObject.")
                    clone = self.pdf.make_stream(form.read_bytes())
                    for key, value in form.items():
                        if key not in ("/Length", "/Filter", "/DecodeParms"):
                            clone[key] = value
                    key = args[0]
                    if str(key) in invoked:
                        key = pikepdf.Name(f"/SpejlForm{len(self.streams)}")
                        while key in objects:
                            key = pikepdf.Name(str(key) + "_")
                    invoked.add(str(args[0]))
                    objects[key] = clone
                    form_ctm = compose(ctm, tuple(map(float, form.get("/Matrix", IDENTITY))))
                    self._walk(
                        clone,
                        pikepdf.parse_content_stream(form),
                        clone,
                        form.get("/Resources", resources),
                        form_ctm,
                        leading,
                        mode,
                        ancestors | {form.objgen},
                    )
                    instruction = op("Do", [key])
            elif name == "BT":
                line, positioned, shown = IDENTITY, True, False
            elif name == "ET":
                line = None
            elif name == "TL":
                leading = float(args[0])
            elif name == "Tr":
                mode = int(args[0])
            elif line is not None:
                if name == "Tm":
                    line, positioned = tuple(map(float, args)), True
                    continue
                if name in ("Td", "TD", "T*", "'", '"'):
                    if name in ("Td", "TD"):
                        x, y = map(float, args)
                        if name == "TD":
                            leading = -y
                            output.append(op("TL", [leading]))
                    else:
                        x, y = 0.0, -leading
                    line = compose(line, (1.0, 0.0, 0.0, 1.0, x, y))
                    positioned = True
                    if name not in ("'", '"'):
                        continue
                    if name == '"':
                        output.extend((op("Tw", [args[0]]), op("Tc", [args[1]])))
                    instruction = op("Tj", [args[-1]])
                    name = "Tj"
                if name in ("Tj", "TJ") and positioned:
                    # ET flushes the interpreter's trace at a new baseline.
                    # Text-state parameters persist across BT/ET.
                    if shown:
                        if mode >= 4:
                            raise ValueError("Cannot split a clipping text object.")
                        output.extend((op("ET"), op("BT")))
                    self.placements.append(Placement(output, len(output), ctm))
                    output.append(op("Tm", line))
                    positioned, shown = False, True
            output.append(instruction)

    def render(self, reflection=None):
        for stream, instructions in self.streams:
            if reflection is not None and stream is self.streams[0][0]:
                instructions = [op("q"), op("cm", reflection), *instructions, op("Q")]
            stream.write(pikepdf.unparse_content_stream(instructions))
        data = BytesIO()
        self.pdf.save(data)
        return pymupdf.open(stream=data.getvalue(), filetype="pdf")
