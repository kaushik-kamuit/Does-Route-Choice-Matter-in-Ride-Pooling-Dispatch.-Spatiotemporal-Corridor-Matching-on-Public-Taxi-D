"""
Generate a clean static corridor map for Figure 2(a).

This version avoids HTML -> screenshot conversion. It renders OSM tiles,
overlays three illustrative route corridors in distinct colors, and places
the legend in a dedicated footer so it never overlaps the map content.

Outputs:
  - results/plots/paper_fig2a_corridor_map.png
  - results/plots/paper_fig2a_corridor_map.jpg
  - paper/figures/paper_fig2a_corridor_map.png
  - paper/figures/paper_fig2a_corridor_map.jpg
"""

from __future__ import annotations

import io
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.font_manager as fm
import requests
from PIL import Image, ImageDraw, ImageFont

import h3

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.spatial.corridor import build_corridor
from src.spatial.h3_utils import LatLng
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PAPER_FIG_DIR = ROOT / "paper" / "figures"

OSRM_BASE_URL = os.environ.get("OSRM_BASE_URL", "https://router.project-osrm.org").rstrip("/")
TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_SIZE = 256
ZOOM = 13
PADDING_PX = 90
FOOTER_H = 118


@dataclass(frozen=True, slots=True)
class RouteStyle:
    label: str
    color: tuple[int, int, int]
    waypoints: tuple[LatLng, ...] = ()


ORIGIN: LatLng = (40.8200, -73.9490)
DESTINATION: LatLng = (40.7007, -74.0125)

ROUTES = [
    RouteStyle("Route A", (186, 146, 34), ((40.7610, -73.9910), (40.7315, -73.9900))),
    RouteStyle("Route B", (168, 65, 54), ((40.7620, -73.9680), (40.7240, -73.9690))),
    RouteStyle("Route C", (108, 108, 108), ((40.7420, -74.0105),)),
]


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    family = "DejaVu Sans"
    props = fm.FontProperties(family=family, weight="bold" if bold else "regular")
    path = fm.findfont(props, fallback_to_default=True)
    return ImageFont.truetype(path, size=size)


def _latlng_to_world_px(lat: float, lng: float, zoom: int) -> tuple[float, float]:
    scale = TILE_SIZE * (2**zoom)
    x = (lng + 180.0) / 360.0 * scale
    lat_rad = math.radians(lat)
    y = (
        (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi)
        / 2.0
        * scale
    )
    return x, y


def _world_px_to_latlng(x: float, y: float, zoom: int) -> LatLng:
    scale = TILE_SIZE * (2**zoom)
    lng = x / scale * 360.0 - 180.0
    n = math.pi - 2.0 * math.pi * y / scale
    lat = math.degrees(math.atan(math.sinh(n)))
    return lat, lng


def _fetch_route(coords: list[LatLng]) -> list[LatLng]:
    path = ";".join(f"{lng},{lat}" for lat, lng in coords)
    url = (
        f"{OSRM_BASE_URL}/route/v1/driving/{path}"
        "?alternatives=false&overview=full&steps=false&geometries=geojson"
    )
    resp = requests.get(
        url,
        timeout=40,
        headers={"User-Agent": "route-aware-dispatch-figure-generator/1.0"},
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != "Ok" or not data.get("routes"):
        raise RuntimeError(f"OSRM route fetch failed: {data}")
    coords_lonlat = data["routes"][0]["geometry"]["coordinates"]
    return [(lat, lng) for lng, lat in coords_lonlat]


def _corridor_boundary_points(cells: Iterable[str]) -> list[LatLng]:
    pts: list[LatLng] = []
    for cell in cells:
        boundary = h3.cell_to_boundary(cell)
        pts.extend((lat, lng) for lat, lng in boundary)
    return pts


def _compute_canvas_bounds(route_polylines: list[list[LatLng]], corridor_cells: list[set[str]]) -> tuple[int, int, int, int]:
    xs: list[float] = []
    ys: list[float] = []
    for polyline in route_polylines:
        for lat, lng in polyline:
            x, y = _latlng_to_world_px(lat, lng, ZOOM)
            xs.append(x)
            ys.append(y)
    for pts in (_corridor_boundary_points(cells) for cells in corridor_cells):
        for lat, lng in pts:
            x, y = _latlng_to_world_px(lat, lng, ZOOM)
            xs.append(x)
            ys.append(y)
    return (
        math.floor(min(xs) - PADDING_PX),
        math.floor(min(ys) - PADDING_PX),
        math.ceil(max(xs) + PADDING_PX),
        math.ceil(max(ys) + PADDING_PX),
    )


def _fetch_tile(z: int, x: int, y: int) -> Image.Image:
    url = TILE_URL.format(z=z, x=x, y=y)
    resp = requests.get(url, timeout=30, headers={"User-Agent": "route-aware-dispatch-figure-generator/1.0"})
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGBA")


def _build_basemap(min_x: int, min_y: int, max_x: int, max_y: int) -> tuple[Image.Image, int, int]:
    tile_x0 = min_x // TILE_SIZE
    tile_y0 = min_y // TILE_SIZE
    tile_x1 = max_x // TILE_SIZE
    tile_y1 = max_y // TILE_SIZE
    width = (tile_x1 - tile_x0 + 1) * TILE_SIZE
    height = (tile_y1 - tile_y0 + 1) * TILE_SIZE
    canvas = Image.new("RGBA", (width, height))
    for tx in range(tile_x0, tile_x1 + 1):
        for ty in range(tile_y0, tile_y1 + 1):
            tile = _fetch_tile(ZOOM, tx, ty)
            canvas.paste(tile, ((tx - tile_x0) * TILE_SIZE, (ty - tile_y0) * TILE_SIZE))
    crop = canvas.crop(
        (
            min_x - tile_x0 * TILE_SIZE,
            min_y - tile_y0 * TILE_SIZE,
            max_x - tile_x0 * TILE_SIZE,
            max_y - tile_y0 * TILE_SIZE,
        )
    )
    wash = Image.new("RGBA", crop.size, (255, 255, 255, 62))
    crop = Image.alpha_composite(crop, wash)
    return crop, min_x, min_y


def _to_local_px(lat: float, lng: float, world_x0: int, world_y0: int) -> tuple[float, float]:
    x, y = _latlng_to_world_px(lat, lng, ZOOM)
    return x - world_x0, y - world_y0


def _rounded_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
    text_fill: tuple[int, int, int] = (255, 255, 255),
) -> tuple[int, int, int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0] + 18
    h = bbox[3] - bbox[1] + 10
    x, y = int(xy[0]), int(xy[1])
    rect = (x, y, x + w, y + h)
    draw.rounded_rectangle(rect, radius=10, fill=fill, outline=(255, 255, 255), width=2)
    draw.text((x + 9, y + 5), text, fill=text_fill, font=font)
    return rect


def _draw_origin_destination(draw: ImageDraw.ImageDraw, world_x0: int, world_y0: int) -> None:
    label_font = _font(15, bold=True)
    specs = [
        ("Origin", ORIGIN, (37, 37, 37), (-78, -14)),
        ("Destination", DESTINATION, (37, 37, 37), (-38, -34)),
    ]
    for label, point, fill, offset in specs:
        x, y = _to_local_px(point[0], point[1], world_x0, world_y0)
        draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=(255, 255, 255, 240), outline=(70, 70, 70), width=2)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=fill)
        _rounded_label(draw, (x + offset[0], y + offset[1]), label, label_font, fill)


def _route_midpoint(polyline: list[LatLng]) -> LatLng:
    return polyline[len(polyline) // 2]


def _draw_route_tag(
    draw: ImageDraw.ImageDraw,
    world_x0: int,
    world_y0: int,
    polyline: list[LatLng],
    style: RouteStyle,
    offset: tuple[int, int],
) -> None:
    font = _font(15, bold=True)
    lat, lng = _route_midpoint(polyline)
    x, y = _to_local_px(lat, lng, world_x0, world_y0)
    _rounded_label(draw, (x + offset[0], y + offset[1]), style.label, font, style.color)


def _draw_corridors(base: Image.Image, route_polylines: list[list[LatLng]], corridors: list[set[str]]) -> Image.Image:
    overlay = Image.new("RGBA", base.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    world_x0 = 0
    world_y0 = 0
    # We pass the local origin through base.info to avoid threading extra args around.
    world_x0 = int(base.info["world_x0"])
    world_y0 = int(base.info["world_y0"])

    for style, polyline, corridor_cells in zip(ROUTES, route_polylines, corridors):
        fill = (*style.color, 86)
        outline = (*style.color, 150)
        for cell in corridor_cells:
            boundary = h3.cell_to_boundary(cell)
            pts = [_to_local_px(lat, lng, world_x0, world_y0) for lat, lng in boundary]
            draw.polygon(pts, fill=fill, outline=outline)

        route_pts = [_to_local_px(lat, lng, world_x0, world_y0) for lat, lng in polyline]
        darker = tuple(max(c - 42, 0) for c in style.color)
        draw.line(route_pts, fill=(255, 255, 255, 220), width=12, joint="curve")
        draw.line(route_pts, fill=(*darker, 250), width=7, joint="curve")

    label_offsets = {
        "Route A": (-8, 22),
        "Route B": (8, -28),
        "Route C": (-86, -8),
    }
    for style, polyline in zip(ROUTES, route_polylines):
        _draw_route_tag(draw, world_x0, world_y0, polyline, style, label_offsets.get(style.label, (8, 8)))

    _draw_origin_destination(draw, world_x0, world_y0)
    return Image.alpha_composite(base, overlay)


def _legend_row(draw: ImageDraw.ImageDraw, x: int, y: int, style: RouteStyle, font: ImageFont.FreeTypeFont) -> None:
    draw.rounded_rectangle((x, y + 7, x + 46, y + 25), radius=5, fill=(*style.color, 50), outline=(*style.color, 210), width=2)
    draw.line((x + 4, y + 16, x + 42, y + 16), fill=style.color, width=6)
    draw.text((x + 58, y), style.label, fill=(32, 32, 32), font=font)


def _compose_final(map_img: Image.Image) -> Image.Image:
    width, height = map_img.size
    canvas = Image.new("RGB", (width, height + FOOTER_H), "white")
    canvas.paste(map_img.convert("RGB"), (0, 0))
    draw = ImageDraw.Draw(canvas)

    title_font = _font(16, bold=True)
    body_font = _font(14, bold=False)
    body_bold = _font(14, bold=True)

    draw.line((0, height, width, height), fill=(221, 221, 221), width=1)
    draw.text((28, height + 14), "Three illustrative route corridors", fill=(25, 25, 25), font=title_font)

    legend_y = height + 42
    margin = 28
    gap = 18
    col_w = (width - 2 * margin - 2 * gap) // 3
    for i, style in enumerate(ROUTES):
        x = margin + i * (col_w + gap)
        _legend_row(draw, x, legend_y, style, body_font)

    draw.ellipse((28, height + 88, 40, height + 100), fill=(35, 35, 35))
    draw.text((48, height + 84), "Origin / destination markers", fill=(50, 50, 50), font=body_font)
    draw.text((width - 275, height + 84), "H3 resolution 9, k=1 expansion", fill=(50, 50, 50), font=body_bold)
    return canvas


def _save(image: Image.Image, name: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    png_plot = PLOTS_DIR / f"{name}.png"
    jpg_plot = PLOTS_DIR / f"{name}.jpg"
    png_paper = PAPER_FIG_DIR / f"{name}.png"
    jpg_paper = PAPER_FIG_DIR / f"{name}.jpg"
    image.save(png_plot)
    image.save(jpg_plot, quality=95)
    image.save(png_paper)
    image.save(jpg_paper, quality=95)
    print(f"  [Saved] {png_plot}")
    print(f"  [Saved] {png_paper}")


def main() -> None:
    route_polylines: list[list[LatLng]] = []
    corridor_cells: list[set[str]] = []
    for style in ROUTES:
        coords = [ORIGIN, *style.waypoints, DESTINATION]
        polyline = _fetch_route(coords)
        route_polylines.append(polyline)
        corridor = build_corridor(polyline, resolution=9, buffer_rings=1, densify_step_m=80)
        corridor_cells.append(set(corridor.corridor_cells))

    min_x, min_y, max_x, max_y = _compute_canvas_bounds(route_polylines, corridor_cells)
    basemap, world_x0, world_y0 = _build_basemap(min_x, min_y, max_x, max_y)
    basemap.info["world_x0"] = world_x0
    basemap.info["world_y0"] = world_y0
    with_corridors = _draw_corridors(basemap, route_polylines, corridor_cells)
    final = _compose_final(with_corridors)
    _save(final, "paper_fig2a_corridor_map")


if __name__ == "__main__":
    main()
