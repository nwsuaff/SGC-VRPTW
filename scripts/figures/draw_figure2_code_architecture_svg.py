#!/usr/bin/env python3
"""Draw an editable SVG draft for Figure 2.

The diagram uses explicit SVG primitives so lanes, gates, route sketches,
diagnostic states, and evidence records remain editable after export.
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape


W, H = 2400, 1540
OUT = Path(__file__).resolve().parents[2] / "evidence" / "qa" / "generated_figures" / "architecture_drafts"


COL = {
    "ink": "#252B2E",
    "navy": "#274154",
    "navy2": "#193247",
    "slate": "#657178",
    "steel": "#D8DEE0",
    "steel2": "#B9C4C8",
    "paper": "#F7F6F1",
    "white": "#FFFFFF",
    "bluewash": "#EEF3F4",
    "teal": "#3E746C",
    "teal2": "#2F625A",
    "tealwash": "#ECF4F1",
    "clay": "#9A5C45",
    "clay2": "#7F4939",
    "claywash": "#F5EEE9",
    "olive": "#727C5A",
    "olivewash": "#F0F2EA",
    "graywash": "#F1F2EF",
    "grid": "#E5E8E8",
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
        size: float = 22,
        weight: str = "400",
        fill: str = COL["ink"],
        anchor: str = "start",
        style: str | None = None,
        rotate: float | None = None,
    ) -> None:
        extra = style or ""
        transform = None
        if rotate is not None:
            transform = f"rotate({rotate} {x} {y})"
        text_attrs = attrs(
            x=x,
            y=y,
            fill=fill,
            font_family=FONT,
            font_size=size,
            font_weight=weight,
            text_anchor=anchor,
            style=extra,
            transform=transform,
        )
        self.add(f"<text {text_attrs}>{escape(text)}</text>")

    def multiline(
        self,
        x: float,
        y: float,
        lines: list[str],
        size: float = 20,
        leading: float = 27,
        fill: str = COL["ink"],
        weight: str = "400",
        anchor: str = "start",
    ) -> None:
        for i, line in enumerate(lines):
            self.text(x, y + i * leading, line, size=size, weight=weight, fill=fill, anchor=anchor)


def chamfered_rect(svg: SVG, x: float, y: float, w: float, h: float, cut: float = 10, **kw: object) -> None:
    d = (
        f"M{x + cut},{y} L{x + w},{y} L{x + w},{y + h - cut} "
        f"L{x + w - cut},{y + h} L{x},{y + h} L{x},{y + cut} Z"
    )
    svg.path(d, **kw)


def marker_for(color: str) -> str:
    mapping = {
        COL["ink"]: "url(#arrowInk)",
        COL["clay"]: "url(#arrowClay)",
        COL["clay2"]: "url(#arrowClay)",
        COL["teal"]: "url(#arrowTeal)",
        COL["teal2"]: "url(#arrowTeal)",
        COL["navy"]: "url(#arrowNavy)",
        COL["navy2"]: "url(#arrowNavy)",
        COL["slate"]: "url(#arrowSlate)",
        COL["olive"]: "url(#arrowOlive)",
    }
    return mapping.get(color, "url(#arrowInk)")


def arrow(svg: SVG, x1: float, y1: float, x2: float, y2: float, color: str = COL["ink"], dashed: bool = False) -> None:
    svg.line(
        x1,
        y1,
        x2,
        y2,
        stroke=color,
        stroke_width=2.2,
        stroke_dasharray=("8 7" if dashed else None),
        marker_end=marker_for(color),
    )


def right_angle_arrow(svg: SVG, points: list[tuple[float, float]], color: str = COL["ink"], dashed: bool = False) -> None:
    d = "M" + " L".join(f"{x},{y}" for x, y in points)
    svg.path(
        d,
        fill="none",
        stroke=color,
        stroke_width=2.2,
        stroke_dasharray=("8 7" if dashed else None),
        marker_end=marker_for(color),
    )


def draw_micro_grid(svg: SVG) -> None:
    for x in range(80, W, 120):
        svg.line(x, 92, x, 1378, stroke=COL["grid"], stroke_width=0.6, opacity=0.45)
    for y in range(130, 1370, 96):
        svg.line(38, y, W - 40, y, stroke=COL["grid"], stroke_width=0.6, opacity=0.35)


def lane_header(svg: SVG, x: float, y: float, w: float, tag: str, title: str, accent: str) -> None:
    chamfered_rect(svg, x, y, w, 66, 8, fill=accent, stroke=COL["ink"], stroke_width=1.1)
    svg.rect(x + 18, y + 15, 38, 36, fill=COL["paper"], stroke=COL["ink"], stroke_width=1)
    svg.text(x + 37, y + 41, tag, size=21, weight="700", anchor="middle")
    svg.text(x + 74, y + 42, title, size=25, weight="700", fill=COL["white"])


def lane(svg: SVG, x: float, y: float, w: float, h: float, tag: str, title: str, accent: str) -> None:
    chamfered_rect(svg, x, y, w, h, 10, fill=COL["white"], stroke=COL["steel2"], stroke_width=1.2)
    lane_header(svg, x, y, w, tag, title, accent)
    for yy in [y + 252, y + 502, y + 752]:
        svg.line(x + 24, yy, x + w - 24, yy, stroke=COL["grid"], stroke_width=1.0, stroke_dasharray="7 7")


def small_label(svg: SVG, x: float, y: float, label: str, accent: str = COL["slate"]) -> None:
    svg.text(x, y, label, size=16, fill=accent, weight="700")
    svg.line(x, y + 7, x + 120, y + 7, stroke=accent, stroke_width=1.1)


def chip(svg: SVG, x: float, y: float, w: float, h: float, title: str, body: str, accent: str, fill: str) -> None:
    chamfered_rect(svg, x, y, w, h, 7, fill=fill, stroke=accent, stroke_width=1.15)
    svg.text(x + 18, y + 31, title, size=19, weight="700", fill=accent)
    svg.text(x + 18, y + 58, body, size=16, fill=COL["ink"])


def icon_doc(svg: SVG, x: float, y: float, color: str = COL["ink"]) -> None:
    svg.path(f"M{x + 4},{y} L{x + 30},{y} L{x + 43},{y + 13} L{x + 43},{y + 50} L{x + 4},{y + 50} Z", fill="none", stroke=color, stroke_width=1.8)
    svg.path(f"M{x + 30},{y} L{x + 30},{y + 13} L{x + 43},{y + 13}", fill="none", stroke=color, stroke_width=1.5)
    for i in range(4):
        svg.line(x + 12, y + 22 + i * 8, x + 34, y + 22 + i * 8, stroke=color, stroke_width=1.2)


def icon_code(svg: SVG, x: float, y: float, color: str = COL["ink"]) -> None:
    svg.text(x, y + 40, "</>", size=40, weight="700", fill=color)


def icon_sandbox(svg: SVG, x: float, y: float, color: str = COL["ink"]) -> None:
    chamfered_rect(svg, x, y, 52, 44, 6, fill="none", stroke=color, stroke_width=1.8)
    svg.line(x + 8, y + 13, x + 44, y + 13, stroke=color, stroke_width=1.5)
    svg.circle(x + 14, y + 7, 2.4, fill=color)
    svg.circle(x + 23, y + 7, 2.4, fill=color)
    svg.circle(x + 32, y + 7, 2.4, fill=color)
    svg.path(f"M{x + 14},{y + 28} L{x + 23},{y + 20} L{x + 32},{y + 28} L{x + 41},{y + 19}", fill="none", stroke=color, stroke_width=1.8)


def icon_checker(svg: SVG, x: float, y: float, color: str = COL["ink"]) -> None:
    chamfered_rect(svg, x, y, 52, 48, 6, fill="none", stroke=color, stroke_width=1.8)
    for i in range(3):
        yy = y + 15 + i * 11
        svg.path(f"M{x + 10},{yy} L{x + 15},{yy + 5} L{x + 24},{yy - 4}", fill="none", stroke=color, stroke_width=1.8)
        svg.line(x + 30, yy, x + 43, yy, stroke=color, stroke_width=1.4)


def draw_state_lane(svg: SVG, x: float, y: float, w: float) -> None:
    small_label(svg, x + 34, y + 102, "normalized routing state", COL["navy"])
    nodes = {
        "0": (x + 87, y + 234),
        "1": (x + 158, y + 156),
        "2": (x + 226, y + 211),
        "3": (x + 315, y + 150),
        "4": (x + 381, y + 233),
        "5": (x + 302, y + 300),
        "6": (x + 184, y + 306),
    }
    route = ["0", "1", "2", "3", "4", "5", "6"]
    for a, b in zip(route, route[1:]):
        x1, y1 = nodes[a]
        x2, y2 = nodes[b]
        svg.line(x1, y1, x2, y2, stroke=COL["teal"], stroke_width=3.0, marker_end="url(#arrowTeal)")
    for a, b in [("0", "2"), ("2", "5"), ("1", "6"), ("3", "5")]:
        x1, y1 = nodes[a]
        x2, y2 = nodes[b]
        svg.line(x1, y1, x2, y2, stroke=COL["steel2"], stroke_width=1.5, stroke_dasharray="7 7")
    for label, (cx, cy) in nodes.items():
        if label == "0":
            svg.rect(cx - 12, cy - 12, 24, 24, fill=COL["ink"], stroke=COL["ink"])
            svg.text(cx, cy + 6, "0", size=14, weight="700", fill=COL["white"], anchor="middle")
        else:
            fill = COL["navy"] if label in {"1", "2", "3"} else COL["white"]
            svg.circle(cx, cy, 13, fill=fill, stroke=COL["navy"], stroke_width=1.7)
            svg.text(cx, cy + 5, label, size=14, fill=(COL["white"] if fill == COL["navy"] else COL["ink"]), anchor="middle")

    chamfered_rect(svg, x + 42, y + 382, w - 84, 158, 8, fill=COL["bluewash"], stroke=COL["navy"], stroke_width=1.15)
    svg.text(x + 70, y + 420, "VRPTW arrays", size=22, weight="700")
    for i, (k, v) in enumerate(
        [
            ("Cij", "travel time / distance matrix"),
            ("qi", "customer demand"),
            ("si", "service duration"),
            ("ai-bi", "hard service window"),
        ]
    ):
        yy = y + 452 + i * 26
        svg.text(x + 74, yy, k, size=18, weight="700", fill=COL["navy"])
        svg.text(x + 156, yy, v, size=18)

    small_label(svg, x + 34, y + 604, "conditioning descriptors", COL["teal"])
    chip(svg, x + 42, y + 634, 185, 78, "scale", "n, vehicles, horizon", COL["teal"], COL["tealwash"])
    chip(svg, x + 245, y + 634, 185, 78, "pressure", "capacity and TW", COL["teal"], COL["tealwash"])
    chip(svg, x + 42, y + 730, 185, 78, "geometry", "radial / clustered", COL["olive"], COL["olivewash"])
    chip(svg, x + 245, y + 730, 185, 78, "incumbent", "K, D, slack", COL["olive"], COL["olivewash"])

    chamfered_rect(svg, x + 42, y + 856, w - 84, 86, 8, fill=COL["white"], stroke=COL["steel2"], stroke_width=1.1)
    icon_doc(svg, x + 64, y + 873, COL["slate"])
    svg.multiline(x + 126, y + 892, ["single index convention", "shared by generator, executor, checker"], size=18, leading=25)


def draw_generation_lane(svg: SVG, x: float, y: float, w: float) -> None:
    small_label(svg, x + 34, y + 102, "prompt assembly", COL["clay"])
    labels = [
        ("instance descriptors", COL["tealwash"], COL["teal"]),
        ("solver scaffold", COL["bluewash"], COL["navy"]),
        ("retrieved OR-Tools patterns", COL["olivewash"], COL["olive"]),
        ("incumbent and repair memory", COL["claywash"], COL["clay"]),
    ]
    for i, (label, fill, accent) in enumerate(labels):
        yy = y + 134 + i * 58
        chamfered_rect(svg, x + 58, yy, 358, 42, 7, fill=fill, stroke=accent, stroke_width=1.1)
        svg.text(x + 78, yy + 28, label, size=18)
    svg.line(x + 236, y + 377, x + 236, y + 420, stroke=COL["ink"], stroke_width=2.0, marker_end="url(#arrowInk)")

    chamfered_rect(svg, x + 70, y + 425, 334, 122, 9, fill=COL["white"], stroke=COL["clay"], stroke_width=1.5)
    icon_code(svg, x + 92, y + 454, COL["clay"])
    svg.text(x + 252, y + 462, "LLM code interface", size=22, weight="700", anchor="middle", fill=COL["clay2"])
    svg.text(x + 252, y + 497, "generate / repair", size=18, anchor="middle")
    svg.text(x + 252, y + 526, "solver-facing program", size=18, anchor="middle")
    svg.line(x + 236, y + 547, x + 236, y + 594, stroke=COL["ink"], stroke_width=2.0, marker_end="url(#arrowInk)")

    chamfered_rect(svg, x + 45, y + 602, 390, 200, 9, fill=COL["bluewash"], stroke=COL["navy"], stroke_width=1.25)
    svg.text(x + 70, y + 640, "Code contract", size=22, weight="700", fill=COL["navy2"])
    rows = [
        ("entry point", "solve(instance)"),
        ("allowed API", "OR-Tools routing"),
        ("output", "routes JSON"),
        ("guardrails", "deterministic solver scope"),
    ]
    for i, (a, b) in enumerate(rows):
        yy = y + 676 + i * 30
        svg.text(x + 78, yy, a, size=17, weight="700", fill=COL["navy"])
        svg.text(x + 198, yy, b, size=17)

    chamfered_rect(svg, x + 45, y + 846, 390, 96, 9, fill=COL["graywash"], stroke=COL["steel2"], stroke_width=1.1)
    svg.text(x + 72, y + 882, "candidate program c_t", size=21, weight="700")
    svg.text(x + 72, y + 916, "build model, add dimensions, extract routes", size=17)


def draw_execution_lane(svg: SVG, x: float, y: float, w: float) -> None:
    small_label(svg, x + 34, y + 102, "pre-execution guard", COL["navy"])
    chamfered_rect(svg, x + 54, y + 135, 170, 92, 8, fill=COL["bluewash"], stroke=COL["navy"], stroke_width=1.2)
    icon_doc(svg, x + 72, y + 157, COL["navy"])
    svg.text(x + 145, y + 171, "AST", size=21, weight="700", fill=COL["navy"])
    svg.text(x + 145, y + 200, "syntax", size=17)
    chamfered_rect(svg, x + 250, y + 135, 170, 92, 8, fill=COL["bluewash"], stroke=COL["navy"], stroke_width=1.2)
    icon_code(svg, x + 270, y + 155, COL["navy"])
    svg.text(x + 347, y + 171, "solve", size=21, weight="700", fill=COL["navy"])
    svg.text(x + 347, y + 200, "entry check", size=17)
    arrow(svg, x + 224, y + 181, x + 250, y + 181, COL["navy"])

    small_label(svg, x + 34, y + 280, "guarded subprocess", COL["clay"])
    chamfered_rect(svg, x + 45, y + 315, 390, 198, 10, fill=COL["white"], stroke=COL["clay"], stroke_width=1.5)
    icon_sandbox(svg, x + 72, y + 346, COL["clay"])
    svg.text(x + 150, y + 362, "Execution sandbox", size=22, weight="700", fill=COL["clay2"])
    svg.multiline(
        x + 150,
        y + 395,
        ["subprocess isolation", "time budget B_exec", "fixed seed and local files", "stdout / stderr capture"],
        size=17,
        leading=25,
    )
    svg.line(x + 240, y + 513, x + 240, y + 565, stroke=COL["ink"], stroke_width=2.0, marker_end="url(#arrowInk)")

    small_label(svg, x + 34, y + 590, "diagnostic taxonomy", COL["clay"])
    states = [
        ("syntax", x + 46, y + 620),
        ("import", x + 170, y + 620),
        ("runtime", x + 294, y + 620),
        ("timeout", x + 46, y + 682),
        ("empty", x + 170, y + 682),
        ("infeasible", x + 294, y + 682),
    ]
    for label, xx, yy in states:
        chamfered_rect(svg, xx, yy, 104, 43, 7, fill=COL["claywash"], stroke=COL["clay"], stroke_width=1.05)
        svg.text(xx + 52, yy + 28, label, size=16, anchor="middle")

    chamfered_rect(svg, x + 48, y + 778, 164, 84, 8, fill=COL["white"], stroke=COL["clay"], stroke_width=1.2)
    svg.text(x + 130, y + 814, "traceback", size=18, weight="700", anchor="middle", fill=COL["clay2"])
    svg.text(x + 130, y + 842, "repair signal", size=16, anchor="middle")
    chamfered_rect(svg, x + 268, y + 778, 164, 84, 8, fill=COL["tealwash"], stroke=COL["teal"], stroke_width=1.2)
    svg.text(x + 350, y + 814, "route", size=18, weight="700", anchor="middle", fill=COL["teal2"])
    svg.text(x + 350, y + 842, "candidate", size=16, anchor="middle")
    arrow(svg, x + 240, y + 725, x + 130, y + 778, COL["clay"])
    arrow(svg, x + 240, y + 725, x + 350, y + 778, COL["teal"])

    chamfered_rect(svg, x + 76, y + 900, 328, 46, 8, fill=COL["graywash"], stroke=COL["steel2"], stroke_width=1.0)
    svg.text(x + 240, y + 930, "record every attempt t", size=18, weight="700", anchor="middle")


def draw_acceptance_lane(svg: SVG, x: float, y: float, w: float) -> None:
    small_label(svg, x + 34, y + 102, "route-schema conversion", COL["teal"])
    chamfered_rect(svg, x + 46, y + 134, 390, 112, 9, fill=COL["tealwash"], stroke=COL["teal"], stroke_width=1.25)
    svg.text(x + 74, y + 174, "RouteSolution schema", size=22, weight="700", fill=COL["teal2"])
    svg.text(x + 74, y + 208, "vehicle routes, arrival times, loads, distance", size=17)
    svg.line(x + 240, y + 246, x + 240, y + 300, stroke=COL["ink"], stroke_width=2.0, marker_end="url(#arrowInk)")

    small_label(svg, x + 34, y + 314, "independent checker", COL["navy"])
    chamfered_rect(svg, x + 46, y + 348, 390, 252, 10, fill=COL["white"], stroke=COL["navy"], stroke_width=1.45)
    icon_checker(svg, x + 70, y + 386, COL["navy"])
    svg.text(x + 240, y + 386, "Feasibility gate", size=24, weight="700", anchor="middle", fill=COL["navy2"])
    checks = [
        ("coverage", "all once"),
        ("capacity", "within Q"),
        ("time windows", "window OK"),
        ("depot return", "return OK"),
    ]
    for i, (a, b) in enumerate(checks):
        yy = y + 426 + i * 38
        svg.circle(x + 142, yy - 6, 7, fill=COL["teal"], stroke=COL["teal"])
        svg.text(x + 164, yy, a, size=18, weight="700")
        svg.text(x + 300, yy, b, size=17, fill=COL["slate"])

    arrow(svg, x + 240, y + 600, x + 240, y + 646, COL["ink"])
    small_label(svg, x + 34, y + 660, "accepted result row", COL["olive"])
    chamfered_rect(svg, x + 48, y + 692, 386, 122, 8, fill=COL["olivewash"], stroke=COL["olive"], stroke_width=1.2)
    metrics = [("feasible", "1"), ("K", "vehicles"), ("D", "distance"), ("T", "runtime")]
    for i, (head, val) in enumerate(metrics):
        xx = x + 72 + i * 88
        if i > 0:
            svg.line(xx - 23, y + 718, xx - 23, y + 790, stroke=COL["steel2"], stroke_width=1.0)
        svg.text(xx + 24, y + 738, head, size=17, weight="700", anchor="middle", fill=COL["olive"])
        svg.text(xx + 24, y + 773, val, size=16, anchor="middle")

    chamfered_rect(svg, x + 48, y + 856, 386, 92, 8, fill=COL["white"], stroke=COL["teal"], stroke_width=1.2)
    svg.text(x + 74, y + 890, "Acceptance rule", size=21, weight="700", fill=COL["teal2"])
    svg.text(x + 74, y + 922, "accept only if every checker condition passes", size=17)


def draw_evidence_rail(svg: SVG) -> None:
    y = 1135
    chamfered_rect(svg, 42, y, 2316, 244, 12, fill=COL["white"], stroke=COL["ink"], stroke_width=1.35)
    svg.rect(42, y, 144, 244, fill=COL["navy"], stroke="none")
    svg.text(114, y + 60, "evidence", size=26, weight="700", fill=COL["white"], anchor="middle")
    svg.text(114, y + 92, "rail", size=26, weight="700", fill=COL["white"], anchor="middle")
    svg.line(186, y + 38, 2320, y + 38, stroke=COL["steel2"], stroke_width=1.0)
    svg.text(222, y + 30, "append-only execution record for audit and reproduction", size=19, fill=COL["slate"])

    cells = [
        ("run metadata", ["seed", "budget", "backend"], COL["bluewash"], COL["navy"]),
        ("program archive", ["code hash", "template", "attempt"], COL["claywash"], COL["clay"]),
        ("subprocess log", ["stdout", "stderr", "timeout"], COL["graywash"], COL["slate"]),
        ("validation report", ["violations", "checker flag", "distance"], COL["tealwash"], COL["teal"]),
        ("result tables", ["CSV row", "aggregation", "status"], COL["olivewash"], COL["olive"]),
        ("figure sources", ["clean data", "script", "export"], COL["white"], COL["ink"]),
    ]
    x0, cw, gap = 230, 316, 34
    for i, (title, lines, fill, accent) in enumerate(cells):
        x = x0 + i * (cw + gap)
        chamfered_rect(svg, x, y + 70, cw, 132, 9, fill=fill, stroke=accent, stroke_width=1.15)
        svg.text(x + 26, y + 107, title, size=20, weight="700", fill=accent)
        for j, line in enumerate(lines):
            svg.line(x + 28, y + 130 + j * 24, x + 44, y + 130 + j * 24, stroke=accent, stroke_width=1.4)
            svg.text(x + 56, y + 136 + j * 24, line, size=17)
        if i < len(cells) - 1:
            arrow(svg, x + cw + 8, y + 136, x + cw + gap - 6, y + 136, COL["slate"])


def draw_top_and_legend(svg: SVG) -> None:
    chamfered_rect(svg, 42, 32, 2316, 62, 10, fill=COL["white"], stroke=COL["steel2"], stroke_width=1.1)
    svg.text(70, 70, "Execution architecture", size=24, weight="700", fill=COL["navy2"])
    svg.text(344, 70, "conditioning state -> solver program -> guarded execution -> checker-validated route schema", size=20, fill=COL["slate"])
    svg.line(1590, 63, 2316, 63, stroke=COL["teal"], stroke_width=2.0, stroke_dasharray="9 7")
    svg.text(1952, 52, "validation-first acceptance", size=18, weight="700", fill=COL["teal2"], anchor="middle")

    y = 1458
    legend = [
        (COL["navy"], "solver state / schema"),
        (COL["clay"], "generated code / repair"),
        (COL["teal"], "accepted route path"),
        (COL["olive"], "recorded metrics"),
    ]
    x = 520
    for color, label in legend:
        svg.rect(x, y - 18, 34, 23, fill=color, stroke=COL["ink"], stroke_width=0.6)
        svg.text(x + 48, y + 2, label, size=17)
        x += 338
    svg.line(x, y - 7, x + 68, y - 7, stroke=COL["clay"], stroke_width=2.0, stroke_dasharray="8 7", marker_end="url(#arrowClay)")
    svg.text(x + 88, y + 2, "bounded diagnostic feedback", size=17)


def draw_cross_lane_flows(svg: SVG) -> None:
    # Main forward architecture path.
    arrow(svg, 536, 590, 604, 590, COL["ink"])
    arrow(svg, 1100, 590, 1168, 590, COL["ink"])
    arrow(svg, 1664, 590, 1732, 590, COL["ink"])

    # Candidate route path from execution to route schema.
    right_angle_arrow(svg, [(1570, 900), (1698, 900), (1698, 198), (1732, 198)], color=COL["teal"])

    # Repair feedback path: diagnostics return to the code interface.
    right_angle_arrow(svg, [(1288, 948), (1288, 1110), (570, 1110), (570, 486), (676, 486)], color=COL["clay"], dashed=True)
    svg.text(1020, 1094, "bounded repair feedback", size=18, fill=COL["clay2"], weight="700", anchor="middle")
    svg.line(1180, 1080, 1268, 1080, stroke=COL["clay"], stroke_width=2, stroke_dasharray="8 7", marker_end="url(#arrowClay)")
    svg.text(1186, 1065, "diagnostics", size=16, fill=COL["clay2"])


def build_svg() -> str:
    svg = SVG()
    svg.add(
        f'''<defs>
  <marker id="arrowInk" markerWidth="11" markerHeight="8" refX="10" refY="4" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L11,4 L0,8 Z" fill="{COL["ink"]}"/>
  </marker>
  <marker id="arrowClay" markerWidth="11" markerHeight="8" refX="10" refY="4" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L11,4 L0,8 Z" fill="{COL["clay"]}"/>
  </marker>
  <marker id="arrowTeal" markerWidth="11" markerHeight="8" refX="10" refY="4" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L11,4 L0,8 Z" fill="{COL["teal"]}"/>
  </marker>
  <marker id="arrowNavy" markerWidth="11" markerHeight="8" refX="10" refY="4" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L11,4 L0,8 Z" fill="{COL["navy"]}"/>
  </marker>
  <marker id="arrowSlate" markerWidth="11" markerHeight="8" refX="10" refY="4" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L11,4 L0,8 Z" fill="{COL["slate"]}"/>
  </marker>
  <marker id="arrowOlive" markerWidth="11" markerHeight="8" refX="10" refY="4" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L11,4 L0,8 Z" fill="{COL["olive"]}"/>
  </marker>
</defs>'''
    )
    svg.rect(0, 0, W, H, fill=COL["paper"])
    draw_micro_grid(svg)
    draw_top_and_legend(svg)

    y, h, w = 122, 956, 482
    xs = [42, 606, 1170, 1734]
    lanes = [
        ("1", "State abstraction", COL["navy"]),
        ("2", "Code generation", COL["clay"]),
        ("3", "Bounded execution", COL["clay2"]),
        ("4", "Deterministic acceptance", COL["teal2"]),
    ]
    for x, (tag, title, accent) in zip(xs, lanes):
        lane(svg, x, y, w, h, tag, title, accent)

    draw_state_lane(svg, xs[0], y, w)
    draw_generation_lane(svg, xs[1], y, w)
    draw_execution_lane(svg, xs[2], y, w)
    draw_acceptance_lane(svg, xs[3], y, w)
    draw_cross_lane_flows(svg)
    draw_evidence_rail(svg)

    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n'
        + "\n".join(svg.items)
        + "\n</svg>\n"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "Figure_2_Code_Architecture_manual_svg_v1.svg"
    target.write_text(build_svg(), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
