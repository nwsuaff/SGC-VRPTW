#!/usr/bin/env python3
"""Draw an editable SVG draft for Figure 1.

The figure is intentionally built from explicit SVG primitives so every panel,
route, card, arrow, and ledger item can be edited after export.
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape


W, H = 2400, 1290
OUT = Path(__file__).resolve().parents[1] / "manuscript" / "fig" / "ai_drafts"


COL = {
    "ink": "#23282C",
    "navy": "#263D4D",
    "slate": "#667680",
    "steel": "#D8DEE1",
    "steel2": "#BAC5CB",
    "paper": "#F8F7F3",
    "white": "#FFFFFF",
    "bluewash": "#EEF3F4",
    "teal": "#3F766D",
    "tealwash": "#EDF5F2",
    "clay": "#9B5F48",
    "claywash": "#F5EEE9",
    "graywash": "#F3F4F2",
    "grid": "#E5E8EA",
}

FONT = "Times New Roman, TimesNewRomanPSMT, Times New Roman_MSFontService, serif"


def attrs(**kwargs: object) -> str:
    parts = []
    for key, value in kwargs.items():
        if value is None:
            continue
        key = key.rstrip("_").replace("_", "-")
        parts.append(f'{key}="{escape(str(value), {"\"": "&quot;"})}"')
    return " ".join(parts)


class SVG:
    def __init__(self) -> None:
        self.items: list[str] = []

    def add(self, raw: str) -> None:
        self.items.append(raw)

    def rect(self, x: float, y: float, w: float, h: float, **kw: object) -> None:
        self.add(f"<rect {attrs(x=x, y=y, width=w, height=h, **kw)}/>")

    def line(self, x1: float, y1: float, x2: float, y2: float, **kw: object) -> None:
        self.add(f"<line {attrs(x1=x1, y1=y1, x2=x2, y2=y2, **kw)}/>")

    def path(self, d: str, **kw: object) -> None:
        self.add(f"<path {attrs(d=d, **kw)}/>")

    def circle(self, cx: float, cy: float, r: float, **kw: object) -> None:
        self.add(f"<circle {attrs(cx=cx, cy=cy, r=r, **kw)}/>")

    def text(
        self,
        x: float,
        y: float,
        text: str,
        size: float = 24,
        weight: str = "400",
        fill: str = COL["ink"],
        anchor: str = "start",
        style: str | None = None,
    ) -> None:
        extra = style or ""
        text_attrs = attrs(
            x=x,
            y=y,
            fill=fill,
            font_family=FONT,
            font_size=size,
            font_weight=weight,
            text_anchor=anchor,
            style=extra,
        )
        self.add(f"<text {text_attrs}>{escape(text)}</text>")

    def multiline(
        self,
        x: float,
        y: float,
        lines: list[str],
        size: float = 22,
        leading: float = 30,
        fill: str = COL["ink"],
        weight: str = "400",
    ) -> None:
        for i, line in enumerate(lines):
            self.text(x, y + i * leading, line, size=size, weight=weight, fill=fill)


def chamfered_rect(svg: SVG, x: float, y: float, w: float, h: float, cut: float = 10, **kw: object) -> None:
    d = (
        f"M{x + cut},{y} L{x + w},{y} L{x + w},{y + h - cut} "
        f"L{x + w - cut},{y + h} L{x},{y + h} L{x},{y + cut} Z"
    )
    svg.path(d, **kw)


def panel(svg: SVG, x: float, y: float, w: float, h: float, tag: str, title: str) -> None:
    chamfered_rect(svg, x, y, w, h, 10, fill=COL["white"], stroke=COL["ink"], stroke_width=1.5)
    svg.rect(x, y, w, 54, fill=COL["graywash"], stroke="none")
    svg.line(x, y + 54, x + w, y + 54, stroke=COL["ink"], stroke_width=1.1)
    svg.rect(x + 18, y + 13, 42, 36, fill=COL["navy"], stroke=COL["ink"], stroke_width=1)
    svg.text(x + 39, y + 39, tag, size=24, weight="700", fill=COL["white"], anchor="middle")
    svg.text(x + 80, y + 38, title, size=31, weight="700")


def arrow(svg: SVG, x1: float, y1: float, x2: float, y2: float, color: str = COL["ink"], dashed: bool = False) -> None:
    style = "8 8" if dashed else None
    svg.line(x1, y1, x2, y2, stroke=color, stroke_width=2.2, stroke_dasharray=style, marker_end="url(#arrow)")


def right_angle_arrow(svg: SVG, points: list[tuple[float, float]], color: str = COL["ink"], dashed: bool = False) -> None:
    d = "M" + " L".join(f"{x},{y}" for x, y in points)
    svg.path(d, fill="none", stroke=color, stroke_width=2.1, stroke_dasharray=("8 8" if dashed else None), marker_end="url(#arrow)")


def mini_icon(svg: SVG, kind: str, x: float, y: float, color: str = COL["ink"]) -> None:
    if kind == "scene":
        svg.line(x, y + 24, x + 34, y + 24, stroke=color, stroke_width=1.8)
        svg.line(x + 4, y + 24, x + 4, y + 12, stroke=color, stroke_width=3)
        svg.line(x + 14, y + 24, x + 14, y + 5, stroke=color, stroke_width=3)
        svg.line(x + 24, y + 24, x + 24, y + 16, stroke=color, stroke_width=3)
        svg.path(f"M{x + 3},{y + 17} L{x + 14},{y + 8} L{x + 25},{y + 14} L{x + 33},{y + 3}", fill="none", stroke=color, stroke_width=1.8)
    elif kind == "db":
        svg.path(f"M{x},{y + 8} C{x},{y} {x + 36},{y} {x + 36},{y + 8} C{x + 36},{y + 16} {x},{y + 16} {x},{y + 8}", fill="none", stroke=color, stroke_width=1.8)
        svg.line(x, y + 8, x, y + 32, stroke=color, stroke_width=1.8)
        svg.line(x + 36, y + 8, x + 36, y + 32, stroke=color, stroke_width=1.8)
        svg.path(f"M{x},{y + 32} C{x},{y + 40} {x + 36},{y + 40} {x + 36},{y + 32}", fill="none", stroke=color, stroke_width=1.8)
        svg.path(f"M{x},{y + 20} C{x},{y + 28} {x + 36},{y + 28} {x + 36},{y + 20}", fill="none", stroke=color, stroke_width=1.3)
    elif kind == "route":
        pts = [(x, y + 25), (x + 12, y + 7), (x + 27, y + 18), (x + 40, y + 5)]
        svg.path("M" + " L".join(f"{px},{py}" for px, py in pts), fill="none", stroke=color, stroke_width=1.8)
        for px, py in pts:
            svg.circle(px, py, 3.5, fill=COL["white"], stroke=color, stroke_width=1.5)
    elif kind == "code":
        svg.text(x, y + 30, "</>", size=33, weight="700", fill=color)
    elif kind == "check":
        svg.circle(x + 18, y + 18, 16, fill="none", stroke=color, stroke_width=2)
        svg.path(f"M{x + 9},{y + 18} L{x + 16},{y + 25} L{x + 29},{y + 10}", fill="none", stroke=color, stroke_width=2.3)
    elif kind == "x":
        svg.circle(x + 18, y + 18, 16, fill="none", stroke=color, stroke_width=2)
        svg.line(x + 10, y + 10, x + 26, y + 26, stroke=color, stroke_width=2.3)
        svg.line(x + 26, y + 10, x + 10, y + 26, stroke=color, stroke_width=2.3)
    elif kind == "gear":
        svg.circle(x + 18, y + 18, 14, fill="none", stroke=color, stroke_width=2)
        svg.circle(x + 18, y + 18, 5, fill="none", stroke=color, stroke_width=2)
        for dx, dy in [(18, 0), (18, 36), (0, 18), (36, 18)]:
            svg.line(x + 18, y + 18, x + dx, y + dy, stroke=color, stroke_width=1.8)
    elif kind == "doc":
        svg.path(f"M{x + 5},{y} L{x + 28},{y} L{x + 38},{y + 10} L{x + 38},{y + 42} L{x + 5},{y + 42} Z", fill="none", stroke=color, stroke_width=1.8)
        svg.path(f"M{x + 28},{y} L{x + 28},{y + 10} L{x + 38},{y + 10}", fill="none", stroke=color, stroke_width=1.5)
        for i in range(3):
            svg.line(x + 12, y + 18 + i * 8, x + 31, y + 18 + i * 8, stroke=color, stroke_width=1.3)


def evidence_chip(svg: SVG, x: float, y: float, w: float, h: float, title: str, lines: list[str], accent: str, icon: str) -> None:
    chamfered_rect(svg, x, y, w, h, 8, fill=COL["white"], stroke=accent, stroke_width=1.4)
    svg.rect(x, y, w, 38, fill=accent, opacity=0.12, stroke="none")
    svg.line(x, y + 38, x + w, y + 38, stroke=accent, stroke_width=1.1)
    mini_icon(svg, icon, x + 16, y + 8, accent)
    svg.text(x + 62, y + 29, title, size=21, weight="700")
    for i, line in enumerate(lines):
        svg.text(x + 26, y + 68 + i * 26, f"• {line}", size=18, fill=COL["ink"])


def draw_vrptw_panel(svg: SVG) -> None:
    panel(svg, 28, 30, 520, 640, "A", "VRPTW state")
    svg.text(60, 125, "Route network (epoch t)", size=22, weight="700")
    svg.rect(62, 152, 15, 15, fill=COL["ink"])
    svg.text(86, 166, "Depot", size=18)
    svg.circle(158, 160, 9, fill=COL["navy"], stroke=COL["navy"])
    svg.text(174, 166, "Served", size=18)
    svg.circle(258, 160, 9, fill=COL["white"], stroke=COL["ink"], stroke_width=2)
    svg.text(276, 166, "Newly released", size=18)

    nodes = {
        "0": (82, 265),
        "1": (205, 178),
        "2": (142, 238),
        "3": (185, 332),
        "4": (286, 245),
        "5": (369, 182),
        "6": (383, 303),
        "7": (474, 210),
        "8": (470, 330),
    }
    route1 = ["0", "2", "1", "4", "5", "7"]
    route2 = ["0", "3", "4", "6", "8"]
    for route, color in [(route1, COL["teal"]), (route2, COL["clay"])]:
        for a, b in zip(route, route[1:]):
            x1, y1 = nodes[a]
            x2, y2 = nodes[b]
            svg.line(x1, y1, x2, y2, stroke=color, stroke_width=3, marker_end="url(#smallArrow)")
    for a, b in [("0", "4"), ("2", "3"), ("4", "7"), ("6", "8")]:
        x1, y1 = nodes[a]
        x2, y2 = nodes[b]
        svg.line(x1, y1, x2, y2, stroke=COL["slate"], stroke_width=1.6, stroke_dasharray="7 7")
    for label, (x, y) in nodes.items():
        if label == "0":
            svg.rect(x - 10, y - 10, 20, 20, fill=COL["ink"], stroke=COL["ink"])
            svg.text(x, y + 6, "0", size=14, fill=COL["white"], anchor="middle")
        else:
            fill = COL["navy"] if label in {"1", "2", "3", "4", "5", "6"} else COL["white"]
            stroke = COL["navy"] if fill != COL["white"] else COL["ink"]
            svg.circle(x, y, 11, fill=fill, stroke=stroke, stroke_width=1.8)
            svg.text(x, y + 5, label, size=13, fill=(COL["white"] if fill != COL["white"] else COL["ink"]), anchor="middle")

    svg.text(60, 420, "Time windows", size=21, weight="700")
    svg.line(175, 430, 500, 430, stroke=COL["ink"], stroke_width=1.2, marker_end="url(#axisArrow)")
    svg.text(170, 456, "0", size=16)
    svg.text(326, 456, "t", size=18, fill=COL["clay"], weight="700")
    svg.text(490, 456, "T", size=16)
    svg.line(330, 415, 330, 562, stroke=COL["clay"], stroke_width=1.3, stroke_dasharray="6 5")
    for i, (name, x1, x2, ok) in enumerate([("i", 190, 255, True), ("j", 222, 315, True), ("k", 300, 390, True), ("...", 235, 435, False)]):
        y = 485 + i * 28
        svg.text(122, y + 5, name, size=18, style="font-style:italic")
        svg.line(175, y, 495, y, stroke=COL["steel2"], stroke_width=1, stroke_dasharray="5 5")
        svg.line(x1, y, x2, y, stroke=(COL["teal"] if ok else COL["clay"]), stroke_width=4)

    chamfered_rect(svg, 60, 555, 445, 92, 8, fill=COL["bluewash"], stroke=COL["slate"], stroke_width=1.2)
    svg.text(82, 586, "Instance summary", size=20, weight="700")
    mini_icon(svg, "doc", 82, 602, COL["slate"])
    svg.text(135, 624, "Q", size=22, weight="700")
    svg.text(182, 624, "Capacity", size=16)
    svg.circle(260, 614, 14, fill="none", stroke=COL["slate"], stroke_width=2)
    svg.line(260, 614, 260, 604, stroke=COL["slate"], stroke_width=2)
    svg.line(260, 614, 270, 620, stroke=COL["slate"], stroke_width=2)
    svg.text(292, 624, "[0, T]", size=21, weight="700")
    svg.text(352, 624, "horizon", size=16)
    svg.text(425, 624, "t", size=22, weight="700", style="font-style:italic")
    svg.text(450, 624, "epoch", size=16)


def draw_contract_panel(svg: SVG) -> None:
    panel(svg, 590, 30, 620, 640, "B", "Evidence-to-contract")
    evidence_chip(svg, 622, 118, 292, 132, "Scene Card", ["scale: n, m", "TW pressure: high", "demand pressure: med"], COL["teal"], "scene")
    evidence_chip(svg, 622, 274, 292, 132, "Retrieval Memory", ["OR-Tools patterns", "callbacks", "route extraction"], COL["navy"], "db")
    evidence_chip(svg, 622, 430, 292, 132, "Incumbent Route", ["vehicles: 12", "distance: 1254.3", "violations: 0"], COL["clay"], "route")

    chamfered_rect(svg, 974, 150, 190, 386, 7, fill=COL["white"], stroke=COL["slate"], stroke_width=1.4)
    svg.text(1068, 186, "Structured prompt", size=22, weight="700", anchor="middle")
    svg.text(1068, 214, "contract  z_t", size=22, weight="700", anchor="middle")
    for i, line in enumerate(["Objective", "Constraints", "Data schema", "Allowed APIs", "Termination", "Output schema", "Safety rules"]):
        yy = 270 + i * 38
        if i == 0:
            mini_icon(svg, "check", 998, yy - 22, COL["slate"])
        elif i == 1:
            svg.rect(1008, yy - 19, 20, 20, fill=COL["ink"])
        elif i == 2:
            for gx in range(3):
                for gy in range(3):
                    svg.rect(1004 + gx * 8, yy - 20 + gy * 8, 5, 5, fill=COL["slate"])
        else:
            svg.circle(1016, yy - 10, 10, fill="none", stroke=COL["slate"], stroke_width=1.5)
        svg.text(1040, yy, line, size=18)
    for y in [184, 340, 496]:
        right_angle_arrow(svg, [(914, y), (948, y), (948, 343), (974, 343)], color=COL["ink"])


def draw_generation_panel(svg: SVG) -> None:
    panel(svg, 1250, 30, 430, 640, "C", "Solver-code generation")
    chamfered_rect(svg, 1298, 140, 292, 178, 9, fill=COL["bluewash"], stroke=COL["navy"], stroke_width=1.5)
    svg.text(1444, 184, "LLM", size=28, weight="700", anchor="middle")
    svg.text(1444, 215, "(bounded code policy)", size=20, anchor="middle")
    svg.line(1322, 238, 1566, 238, stroke=COL["slate"], stroke_width=1.2, stroke_dasharray="6 5")
    svg.multiline(1336, 264, ["Proposes executable", "solver code,", "not routes"], size=19, leading=25)
    mini_icon(svg, "route", 1528, 164, COL["slate"])

    arrow(svg, 1444, 318, 1444, 382, COL["ink"])
    svg.text(1458, 360, "Output", size=16, fill=COL["slate"])
    chamfered_rect(svg, 1298, 388, 292, 190, 9, fill=COL["tealwash"], stroke=COL["slate"], stroke_width=1.5)
    svg.text(1444, 428, "OR-Tools candidate  c_t", size=22, weight="700", anchor="middle")
    mini_icon(svg, "code", 1334, 454, COL["ink"])
    svg.multiline(1394, 466, ["Model build", "Search control", "Callbacks", "Solution extract", "..."], size=18, leading=28)
    svg.line(1664, 110, 1664, 585, stroke=COL["navy"], stroke_width=2, stroke_dasharray="9 9")
    svg.add(
        f'<text transform="translate(1646,465) rotate(90)" fill="{COL["navy"]}" '
        f'font-family="{FONT}" font-size="26" font-weight="700" text-anchor="middle">'
        "code, not routes</text>"
    )
    arrow(svg, 1164, 343, 1246, 343, COL["ink"])
    arrow(svg, 1590, 343, 1680, 343, COL["ink"])


def draw_validation_panel(svg: SVG) -> None:
    panel(svg, 1710, 30, 662, 640, "D", "Execution and validation")
    svg.text(1760, 138, "Sandbox execution", size=22, weight="700")
    mini_icon(svg, "gear", 1725, 114, COL["navy"])
    chamfered_rect(svg, 1760, 168, 520, 76, 7, fill=COL["white"], stroke=COL["slate"], stroke_width=1.2)
    svg.text(1794, 213, "OR-Tools", size=20)
    svg.line(1935, 205, 2070, 205, stroke=COL["slate"], stroke_width=1.6, stroke_dasharray="7 6", marker_end="url(#arrow)")
    svg.text(2110, 213, "Candidate solution", size=20)
    arrow(svg, 2020, 244, 2020, 314, COL["ink"])

    svg.text(1760, 308, "Feasibility gate", size=22, weight="700")
    chamfered_rect(svg, 1760, 324, 520, 58, 7, fill=COL["white"], stroke=COL["slate"], stroke_width=1.2)
    svg.text(1792, 361, "basic checks before full validation", size=18)
    arrow(svg, 2020, 382, 2020, 424, COL["ink"])

    chamfered_rect(svg, 1754, 424, 540, 172, 8, fill=COL["white"], stroke=COL["ink"], stroke_width=1.2)
    svg.text(2024, 459, "Independent checker", size=24, weight="700", anchor="middle")
    for i, (lab, sym) in enumerate([("Capacity", "Q"), ("Coverage", "N"), ("Time windows", "T"), ("Objective", "OBJ")]):
        x = 1784 + i * 127
        chamfered_rect(svg, x, 480, 110, 88, 6, fill=COL["graywash"], stroke=COL["steel2"], stroke_width=1)
        svg.text(x + 55, 516, sym, size=28, weight="700", anchor="middle", fill=COL["navy"])
        svg.text(x + 55, 546, lab, size=16, anchor="middle")
        svg.text(x + 55, 566, "OK / X", size=15, anchor="middle")

    right_angle_arrow(svg, [(1885, 596), (1885, 626), (1818, 626), (1818, 660)], color=COL["teal"])
    right_angle_arrow(svg, [(2130, 596), (2130, 626), (2205, 626), (2205, 660)], color=COL["clay"])
    chamfered_rect(svg, 1738, 652, 196, 116, 7, fill=COL["tealwash"], stroke=COL["teal"], stroke_width=1.5)
    mini_icon(svg, "check", 1758, 681, COL["teal"])
    svg.text(1810, 686, "Accepted", size=20, weight="700", fill=COL["teal"])
    svg.multiline(1810, 715, ["Routes", "Schedule", "Metrics"], size=15, leading=20)
    chamfered_rect(svg, 2142, 652, 196, 116, 7, fill=COL["claywash"], stroke=COL["clay"], stroke_width=1.5)
    mini_icon(svg, "x", 2160, 681, COL["clay"])
    svg.text(2208, 686, "Rejected", size=20, weight="700", fill=COL["clay"])
    svg.multiline(2208, 715, ["Infeasible", "Suboptimal", "Timeout"], size=15, leading=20)


def draw_bottom(svg: SVG) -> None:
    chamfered_rect(svg, 28, 718, 1068, 126, 8, fill=COL["white"], stroke=COL["clay"], stroke_width=1.4)
    svg.text(88, 752, "Diagnostics (on rejection)", size=22, weight="700")
    mini_icon(svg, "doc", 50, 732, COL["clay"])
    for i, (title, sub) in enumerate(
        [
            ("Constraint conflict", ""),
            ("Capacity overload", ""),
            ("Time-window violation", ""),
            ("Inefficient search", ""),
            ("Other failure", ""),
        ]
    ):
        x = 132 + i * 186
        chamfered_rect(svg, x, 775, 168, 44, 6, fill=COL["white"], stroke=COL["steel2"], stroke_width=1.1)
        svg.circle(x + 20, 797, 14, fill=COL["white"], stroke=COL["navy"], stroke_width=1.8)
        svg.text(x + 42, 803, title, size=13.5, weight="700")
        if sub:
            svg.text(x + 42, 817, sub, size=13, fill=COL["slate"])
    right_angle_arrow(svg, [(1096, 781), (1280, 781), (1280, 580), (1444, 580)], color=COL["clay"])
    right_angle_arrow(svg, [(2170, 764), (2170, 812), (1444, 812), (1444, 580)], color=COL["clay"], dashed=True)
    svg.text(1438, 861, "typed diagnostics  ->  bounded repair", size=24, weight="700", fill=COL["clay"], anchor="middle")

    chamfered_rect(svg, 28, 914, 2344, 170, 8, fill=COL["white"], stroke=COL["ink"], stroke_width=1.4)
    svg.rect(28, 914, 130, 170, fill=COL["navy"], stroke="none")
    mini_icon(svg, "doc", 75, 960, COL["white"])
    svg.text(190, 965, "Evidence ledger (append-only)", size=25, weight="700")
    cols = [
        ("run_id", "R_20250601_00037"),
        ("prompt_hash", "a1f8...7c2e"),
        ("code_hash", "9b31...d4f1"),
        ("diagnostics_hash", "c06a...91ad"),
        ("status", "Accepted / Rejected"),
        ("runtime (s)", "312.4"),
        ("epoch t", "37"),
        ("seed", "42"),
    ]
    x0, colw = 300, 240
    for i, (head, val) in enumerate(cols):
        x = x0 + i * colw
        if i > 0:
            svg.line(x - 34, 980, x - 34, 1054, stroke=COL["slate"], stroke_width=1, stroke_dasharray="5 5")
        svg.text(x, 1010, head, size=19, weight="700", anchor="middle")
        svg.text(x, 1048, val, size=18, anchor="middle")

    # Legend strip.
    y = 1170
    legend = [
        (COL["bluewash"], "Information"),
        (COL["tealwash"], "Accepted / evidence"),
        (COL["claywash"], "Safety / rejection"),
        (COL["ink"], "Main workflow"),
        (COL["clay"], "Feedback loop"),
    ]
    x = 300
    for color, lab in legend[:3]:
        svg.rect(x, y - 16, 34, 24, fill=color, stroke=COL["steel2"], stroke_width=1)
        svg.text(x + 48, y + 3, lab, size=17)
        x += 310
    svg.line(x, y - 4, x + 70, y - 4, stroke=COL["ink"], stroke_width=2, marker_end="url(#arrow)")
    svg.text(x + 88, y + 3, legend[3][1], size=17)
    x += 290
    svg.line(x, y - 4, x + 70, y - 4, stroke=COL["clay"], stroke_width=2, stroke_dasharray="7 6", marker_end="url(#arrow)")
    svg.text(x + 88, y + 3, legend[4][1], size=17)


def build_svg() -> str:
    svg = SVG()
    svg.add(
        f'''<defs>
  <marker id="arrow" markerWidth="11" markerHeight="8" refX="10" refY="4" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L11,4 L0,8 Z" fill="{COL["ink"]}"/>
  </marker>
  <marker id="smallArrow" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L9,3.5 L0,7 Z" fill="{COL["ink"]}"/>
  </marker>
  <marker id="axisArrow" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L9,3.5 L0,7 Z" fill="{COL["ink"]}"/>
  </marker>
</defs>'''
    )
    svg.rect(0, 0, W, H, fill=COL["paper"])
    draw_vrptw_panel(svg)
    draw_contract_panel(svg)
    draw_generation_panel(svg)
    draw_validation_panel(svg)
    draw_bottom(svg)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n'
        + "\n".join(svg.items)
        + "\n</svg>\n"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "Figure_1_Method_Framework_manual_svg_v1.svg"
    target.write_text(build_svg(), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
