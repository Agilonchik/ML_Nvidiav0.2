#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_taxonomy_svg_unlimited_roots.py

Builds an SVG taxonomy diagram from YAML.

Main features:
- unlimited number of root / main classes;
- roots are detected automatically by parent: null;
- no hardcoded root columns;
- dynamic canvas size;
- recursive tree layout;
- optional PNG export if cairosvg is installed.

Expected YAML structure:

classes:
  - id: 0
    name: "Background"
    parent: null
    type: "misc class"

  - id: 1
    name: "Superstructure"
    parent: null
    type: "Main class 1"

  - id: 4
    name: "Roadway"
    parent: Superstructure
    type: "Main class 4"
"""

from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import math
import textwrap
from typing import Any


# ----------------------------
# Optional dependency: PyYAML
# ----------------------------

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "PyYAML is required. Install it with:\n"
        "pip install pyyaml"
    ) from exc


# ----------------------------
# Basic helpers
# ----------------------------

def esc(text: Any) -> str:
    """Escape text for SVG."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def safe_parent(value: Any) -> str | None:
    """Normalize YAML parent field."""
    if value in (None, "", "null", "NULL", "None", "none"):
        return None
    return str(value)


def wrap_text(text: Any, width: int = 32) -> list[str]:
    return textwrap.wrap(str(text), width=width, break_long_words=False) or [""]


# ----------------------------
# YAML loading and validation
# ----------------------------

def load_yaml(input_path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(input_path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError("YAML root must be a dictionary.")

    if "classes" not in data:
        raise ValueError('YAML must contain top-level key "classes".')

    raw_classes = data["classes"]
    if not isinstance(raw_classes, list):
        raise ValueError('"classes" must be a list.')

    classes: list[dict[str, Any]] = []

    for i, item in enumerate(raw_classes):
        if not isinstance(item, dict):
            raise ValueError(f"Class item #{i} must be a dictionary.")

        if "id" not in item:
            raise ValueError(f"Class item #{i} has no required field: id")

        if "name" not in item:
            raise ValueError(f"Class item #{i} has no required field: name")

        classes.append(
            {
                "id": int(item["id"]),
                "name": str(item["name"]),
                "parent": safe_parent(item.get("parent")),
                "type": str(item.get("type", "")),
            }
        )

    classes.sort(key=lambda c: c["id"])
    validate_classes(classes)

    return classes


def validate_classes(classes: list[dict[str, Any]]) -> None:
    names = [c["name"] for c in classes]
    ids = [c["id"] for c in classes]

    duplicated_names = sorted({name for name in names if names.count(name) > 1})
    duplicated_ids = sorted({id_ for id_ in ids if ids.count(id_) > 1})

    if duplicated_names:
        raise ValueError(f"Duplicated class names found: {duplicated_names}")

    if duplicated_ids:
        raise ValueError(f"Duplicated class IDs found: {duplicated_ids}")

    by_name = {c["name"]: c for c in classes}

    # Parent may be missing if you want to show orphan nodes as separate roots.
    # We warn later in SVG, but do not fail.
    for c in classes:
        parent = c["parent"]
        if parent is not None and parent not in by_name:
            print(
                f'Warning: parent "{parent}" for class "{c["name"]}" '
                f'is not present in classes. This node will be treated as orphan root.'
            )


# ----------------------------
# Tree building
# ----------------------------

def build_forest(classes: list[dict[str, Any]]):
    by_name = {c["name"]: c for c in classes}

    children: dict[str | None, list[dict[str, Any]]] = {}
    orphan_roots: list[dict[str, Any]] = []

    for c in classes:
        parent = c["parent"]

        if parent is not None and parent not in by_name:
            orphan_roots.append(c)
        else:
            children.setdefault(parent, []).append(c)

    for key in children:
        children[key].sort(key=lambda x: x["id"])

    roots = children.get(None, []) + orphan_roots
    roots.sort(key=lambda x: x["id"])

    return by_name, children, roots


def detect_cycles(classes: list[dict[str, Any]]) -> None:
    """Detect simple parent cycles by walking upward from each node."""
    by_name = {c["name"]: c for c in classes}

    for c in classes:
        seen: set[str] = set()
        current = c

        while current["parent"] is not None:
            parent = current["parent"]

            if parent in seen:
                raise ValueError(
                    f'Cycle detected near class "{c["name"]}". '
                    f'Parent chain repeats "{parent}".'
                )

            seen.add(parent)

            if parent not in by_name:
                break

            current = by_name[parent]


# ----------------------------
# Styling
# ----------------------------

DEFAULT_PALETTE = [
    ("#1D4ED8", "#EFF6FF"),  # blue
    ("#15803D", "#F0FDF4"),  # green
    ("#7E22CE", "#FAF5FF"),  # purple
    ("#D97706", "#FFFBEB"),  # orange
    ("#BE123C", "#FFF1F2"),  # rose
    ("#0F766E", "#F0FDFA"),  # teal
    ("#4338CA", "#EEF2FF"),  # indigo
    ("#A16207", "#FEFCE8"),  # yellow/brown
    ("#0369A1", "#F0F9FF"),  # sky
    ("#047857", "#ECFDF5"),  # emerald
    ("#9333EA", "#FAF5FF"),  # violet
    ("#C2410C", "#FFF7ED"),  # deep orange
]

SPECIAL_STYLES = {
    "Background": {
        "stroke": "#6B7280",
        "fill": "#F3F4F6",
        "accent": "#4B5563",
    },
}


def stable_color_index(text: str, modulo: int) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def build_root_styles(roots: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    """
    Assign a stable color to every root class.
    No limit on root count: colors are reused cyclically if needed.
    """
    styles: dict[str, dict[str, str]] = {}

    for i, root in enumerate(roots):
        name = root["name"]

        if name in SPECIAL_STYLES:
            styles[name] = SPECIAL_STYLES[name]
            continue

        # First roots get palette order; later roots get stable hash.
        palette_index = i % len(DEFAULT_PALETTE)
        stroke, fill = DEFAULT_PALETTE[palette_index]

        styles[name] = {
            "stroke": stroke,
            "fill": fill,
            "accent": stroke,
        }

    return styles


def find_root_name(
    node: dict[str, Any],
    by_name: dict[str, dict[str, Any]],
) -> str:
    """Find top-level root for a node."""
    current = node

    while current.get("parent") is not None:
        parent_name = current["parent"]
        parent = by_name.get(parent_name)

        if parent is None:
            break

        current = parent

    return current["name"]


def style_for(
    node: dict[str, Any],
    by_name: dict[str, dict[str, Any]],
    root_styles: dict[str, dict[str, str]],
) -> dict[str, str]:
    root_name = find_root_name(node, by_name)
    return root_styles.get(
        root_name,
        {
            "stroke": "#374151",
            "fill": "#FFFFFF",
            "accent": "#374151",
        },
    )


# ----------------------------
# Recursive layout
# ----------------------------

def subtree_height(
    node: dict[str, Any],
    children: dict[str | None, list[dict[str, Any]]],
    node_h: int,
    row_gap: int,
) -> int:
    """
    Vertical height required by subtree.
    Children are placed below the node, in a vertical stack.
    """
    child_nodes = children.get(node["name"], [])

    if not child_nodes:
        return node_h

    total_children_height = 0

    for child in child_nodes:
        total_children_height += subtree_height(child, children, node_h, row_gap)

    total_children_height += row_gap * (len(child_nodes) - 1)

    return node_h + row_gap + total_children_height


def place_subtree(
    node: dict[str, Any],
    x: int,
    y: int,
    positions: dict[str, tuple[int, int, int, int]],
    children: dict[str | None, list[dict[str, Any]]],
    node_w: int,
    node_h: int,
    row_gap: int,
    indent_x: int,
    depth: int = 0,
) -> int:
    """
    Place node and all descendants.
    Returns bottom y of subtree.
    """
    width = max(360, node_w - depth * 35)
    positions[node["name"]] = (x + depth * indent_x, y, width, node_h)

    child_nodes = children.get(node["name"], [])
    if not child_nodes:
        return y + node_h

    current_y = y + node_h + row_gap

    for child in child_nodes:
        current_y = place_subtree(
            node=child,
            x=x,
            y=current_y,
            positions=positions,
            children=children,
            node_w=node_w,
            node_h=node_h,
            row_gap=row_gap,
            indent_x=indent_x,
            depth=depth + 1,
        )
        current_y += row_gap

    return current_y - row_gap


def layout_forest(
    roots: list[dict[str, Any]],
    children: dict[str | None, list[dict[str, Any]]],
    root_w: int = 560,
    node_w: int = 560,
    root_h: int = 150,
    node_h: int = 132,
    col_gap: int = 120,
    row_gap: int = 34,
    indent_x: int = 85,
    margin_x: int = 110,
    title_h: int = 200,
    max_roots_per_row: int | None = None,
):
    """
    Layout for unlimited root classes.

    If max_roots_per_row is None:
      all root trees are placed in one horizontal row and canvas becomes wide.

    If max_roots_per_row is set:
      root trees wrap into several rows.
    """
    positions: dict[str, tuple[int, int, int, int]] = {}

    if not roots:
        return positions, 1200, 800

    if max_roots_per_row is None or max_roots_per_row <= 0:
        max_roots_per_row = len(roots)

    root_tree_width = root_w + indent_x * 3

    root_rows = [
        roots[i:i + max_roots_per_row]
        for i in range(0, len(roots), max_roots_per_row)
    ]

    current_y = title_h + 40
    max_right = 0
    max_bottom = 0

    for row_roots in root_rows:
        # Height of current row = maximum subtree height in this row.
        row_subtree_heights = [
            subtree_height(root, children, node_h, row_gap)
            for root in row_roots
        ]

        row_height = max(root_h, max(row_subtree_heights))

        current_x = margin_x

        for root in row_roots:
            # Root box
            positions[root["name"]] = (current_x, current_y, root_w, root_h)

            # Children below root
            child_nodes = children.get(root["name"], [])
            child_y = current_y + root_h + row_gap + 55

            for child in child_nodes:
                bottom_y = place_subtree(
                    node=child,
                    x=current_x,
                    y=child_y,
                    positions=positions,
                    children=children,
                    node_w=node_w,
                    node_h=node_h,
                    row_gap=row_gap,
                    indent_x=indent_x,
                    depth=0,
                )
                child_y = bottom_y + row_gap

            max_right = max(max_right, current_x + root_tree_width)
            max_bottom = max(max_bottom, child_y, current_y + root_h)

            current_x += root_tree_width + col_gap

        current_y += row_height + 260

    canvas_w = max(max_right + margin_x, 1200)
    canvas_h = max(max_bottom + 240, 900)

    return positions, int(canvas_w), int(canvas_h)


# ----------------------------
# SVG drawing
# ----------------------------

def center_top(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, w, h = box
    return x + w / 2, y


def center_bottom(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, w, h = box
    return x + w / 2, y + h


def draw_edge(
    svg: list[str],
    parent_box: tuple[int, int, int, int],
    child_box: tuple[int, int, int, int],
    color: str,
) -> None:
    x1, y1 = center_bottom(parent_box)
    x2, y2 = center_top(child_box)

    mid_y = (y1 + y2) / 2

    svg.append(
        f'<path d="M{x1:.1f},{y1:.1f} '
        f'V{mid_y:.1f} H{x2:.1f} V{y2:.1f}" '
        f'fill="none" stroke="{color}" stroke-width="4"/>'
    )
    svg.append(f'<circle cx="{x2:.1f}" cy="{y2:.1f}" r="7" fill="{color}"/>')


def draw_box(
    svg: list[str],
    node: dict[str, Any],
    box: tuple[int, int, int, int],
    style: dict[str, str],
) -> None:
    x, y, w, h = box
    parent_value = "null" if node["parent"] is None else node["parent"]

    svg.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
        f'rx="22" ry="22" fill="{style["fill"]}" '
        f'stroke="{style["stroke"]}" stroke-width="4"/>'
    )

    svg.append(
        f'<circle cx="{x+52}" cy="{y+52}" r="34" '
        f'fill="#FFFFFF" stroke="{style["stroke"]}" stroke-width="4"/>'
    )

    svg.append(
        f'<text x="{x+52}" y="{y+63}" text-anchor="middle" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="30" '
        f'font-weight="800" fill="{style["accent"]}">{node["id"]}</text>'
    )

    tx = x + 105
    ty = y + 38

    name = esc(node["name"])
    parent = esc(parent_value)
    type_text = esc(wrap_text(node["type"], 32)[0])

    svg.append(
        f'<text x="{tx}" y="{ty}" '
        f'font-family="Arial, Helvetica, sans-serif" '
        f'font-size="25" font-weight="800" fill="#111827">'
        f'ID: {node["id"]}</text>'
    )

    svg.append(
        f'<text x="{tx}" y="{ty+34}" '
        f'font-family="Arial, Helvetica, sans-serif" '
        f'font-size="23" font-weight="800" fill="#111827">'
        f'Name: <tspan font-weight="500">{name}</tspan></text>'
    )

    svg.append(
        f'<text x="{tx}" y="{ty+68}" '
        f'font-family="Arial, Helvetica, sans-serif" '
        f'font-size="21" font-weight="800" fill="#111827">'
        f'Parent: <tspan font-weight="500">{parent}</tspan></text>'
    )

    svg.append(
        f'<text x="{tx}" y="{ty+100}" '
        f'font-family="Arial, Helvetica, sans-serif" '
        f'font-size="20" font-weight="800" fill="#111827">'
        f'Type: <tspan font-weight="600" fill="{style["accent"]}">{type_text}</tspan></text>'
    )


def make_svg(
    classes: list[dict[str, Any]],
    by_name: dict[str, dict[str, Any]],
    roots: list[dict[str, Any]],
    positions: dict[str, tuple[int, int, int, int]],
    canvas_w: int,
    canvas_h: int,
    title: str,
) -> str:
    root_styles = build_root_styles(roots)
    svg: list[str] = []

    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{canvas_w}" height="{canvas_h}" '
        f'viewBox="0 0 {canvas_w} {canvas_h}">'
    )

    svg.append('<rect width="100%" height="100%" fill="#FFFFFF"/>')

    # Title
    svg.append(
        f'<text x="{canvas_w/2}" y="78" text-anchor="middle" '
        f'font-family="Arial, Helvetica, sans-serif" '
        f'font-size="62" font-weight="800" fill="#0F172A">'
        f'{esc(title)}</text>'
    )

    svg.append(
        f'<text x="{canvas_w/2}" y="125" text-anchor="middle" '
        f'font-family="Arial, Helvetica, sans-serif" '
        f'font-size="28" fill="#475569">'
        f'Automatically generated from YAML | root classes: {len(roots)} | total classes: {len(classes)}</text>'
    )

    # Edges
    for node in classes:
        parent_name = node["parent"]

        if parent_name is None:
            continue

        if parent_name not in by_name:
            continue

        if parent_name not in positions or node["name"] not in positions:
            continue

        parent_node = by_name[parent_name]
        parent_style = style_for(parent_node, by_name, root_styles)

        draw_edge(
            svg=svg,
            parent_box=positions[parent_name],
            child_box=positions[node["name"]],
            color=parent_style["stroke"],
        )

    # Boxes
    for node in classes:
        if node["name"] not in positions:
            continue

        node_style = style_for(node, by_name, root_styles)
        draw_box(svg, node, positions[node["name"]], node_style)

    # Footer
    footer_y = canvas_h - 150
    svg.append(
        f'<rect x="110" y="{footer_y}" width="{canvas_w-220}" height="95" '
        f'rx="20" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="3"/>'
    )

    svg.append(
        f'<text x="140" y="{footer_y+38}" '
        f'font-family="Arial, Helvetica, sans-serif" '
        f'font-size="25" font-weight="800" fill="#111827">'
        f'Layout: unlimited root classes, dynamic canvas, recursive hierarchy.</text>'
    )

    svg.append(
        f'<text x="140" y="{footer_y+70}" '
        f'font-family="Arial, Helvetica, sans-serif" '
        f'font-size="22" fill="#475569">'
        f'Root classes are detected by parent: null. Missing parents are shown as orphan roots.</text>'
    )

    svg.append("</svg>")

    return "\n".join(svg)


# ----------------------------
# PNG export
# ----------------------------

def try_export_png(svg_path: Path, png_path: Path) -> bool:
    try:
        import cairosvg
    except ImportError:
        return False

    try:
        cairosvg.svg2png(url=str(svg_path), write_to=str(png_path))
        return True
    except Exception as exc:
        print(f"PNG export failed: {exc}")
        return False


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build SVG taxonomy diagram from YAML with unlimited root classes."
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input YAML file, for example Classes.yaml",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="taxonomy_diagram.svg",
        help="Output SVG file",
    )

    parser.add_argument(
        "--png",
        default=None,
        help="Optional output PNG file. If omitted, uses SVG name with .png extension.",
    )

    parser.add_argument(
        "--title",
        default="Bridge CV Taxonomy",
        help="Diagram title",
    )

    parser.add_argument(
        "--max-roots-per-row",
        type=int,
        default=0,
        help=(
            "Wrap root classes after this number per row. "
            "0 means no wrapping: all roots are placed in one wide row."
        ),
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_svg = Path(args.output)
    output_png = Path(args.png) if args.png else output_svg.with_suffix(".png")

    classes = load_yaml(input_path)
    detect_cycles(classes)

    by_name, children, roots = build_forest(classes)

    max_roots_per_row = args.max_roots_per_row or None

    positions, canvas_w, canvas_h = layout_forest(
        roots=roots,
        children=children,
        max_roots_per_row=max_roots_per_row,
    )

    svg_text = make_svg(
        classes=classes,
        by_name=by_name,
        roots=roots,
        positions=positions,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        title=args.title,
    )

    output_svg.write_text(svg_text, encoding="utf-8")
    png_ok = try_export_png(output_svg, output_png)

    root_names = ", ".join(root["name"] for root in roots)

    print("Done.")
    print(f"Input YAML: {input_path}")
    print(f"Output SVG: {output_svg}")
    if png_ok:
        print(f"Output PNG: {output_png}")
    else:
        print("Output PNG: not created. Install cairosvg if PNG is needed:")
        print("pip install cairosvg")
    print(f"Root classes found: {len(roots)}")
    print(f"Root names: {root_names}")
    print(f"Total classes: {len(classes)}")
    print(f"Canvas: {canvas_w} x {canvas_h}")


if __name__ == "__main__":
    main()
