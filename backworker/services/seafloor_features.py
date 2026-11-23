"""GEBCO SCUFN 国际海底地名数据读取、分类与视口范围查询。"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_POINT_RE = re.compile(
    r"POINT\s*\(\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+"
    r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)
_NUMBER_PAIR_RE = re.compile(
    r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+"
    r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
)
_DIMENSION_RE = re.compile(r"(\d+(?:\.\d+)?)\s*km", re.IGNORECASE)

_CATEGORY_DEFS: tuple[dict[str, Any], ...] = (
    {"key": "strait", "label": "海峡/水道", "color": "#5fd4ff"},
    {"key": "trench", "label": "海沟/海槽", "color": "#b18cff"},
    {"key": "seamount", "label": "海山/海丘", "color": "#ffc65c"},
    {"key": "ridge", "label": "海岭/海脊", "color": "#7fe3a4"},
    {"key": "basin", "label": "海盆/平原", "color": "#8fa8c8"},
    {"key": "canyon", "label": "海底峡谷", "color": "#ff9b6a"},
    {"key": "other", "label": "其他", "color": "#9fb4c6"},
)

CATEGORY_MAP: dict[str, str] = {
    "passage": "strait",
    "channel": "strait",
    "channels": "strait",
    "sea channel": "strait",
    "sea channels": "strait",
    "gap": "strait",
    "pass": "strait",
    "sill": "strait",
    "saddle": "strait",
    "seabight": "strait",
    "re-entrant": "strait",
    "trench": "trench",
    "trenches": "trench",
    "trough": "trench",
    "troughs": "trench",
    "deep": "trench",
    "deeps": "trench",
    "seamount": "seamount",
    "seamounts": "seamount",
    "seamount chain": "seamount",
    "seamount group": "seamount",
    "seamount province": "seamount",
    "guyot": "seamount",
    "guyots": "seamount",
    "knoll": "seamount",
    "knolls": "seamount",
    "tablemount": "seamount",
    "tablemounts": "seamount",
    "peak": "seamount",
    "peaks": "seamount",
    "cone": "seamount",
    "mound": "seamount",
    "mounds": "seamount",
    "hill": "seamount",
    "hills": "seamount",
    "hill province": "seamount",
    "bank": "seamount",
    "banks": "seamount",
    "shoal": "seamount",
    "shoals": "seamount",
    "reef": "seamount",
    "reefs": "seamount",
    "caldera": "seamount",
    "calderas": "seamount",
    "ridge": "ridge",
    "ridges": "ridge",
    "rise": "ridge",
    "rises": "ridge",
    "spur": "ridge",
    "spurs": "ridge",
    "fracture zone": "ridge",
    "fracture zones": "ridge",
    "fracture zone system": "ridge",
    "fracture zone province": "ridge",
    "escarpment": "ridge",
    "scarps": "ridge",
    "scarp": "ridge",
    "rift": "ridge",
    "salt dome province": "ridge",
    "discordance": "ridge",
    "basin": "basin",
    "basins": "basin",
    "basin and ridge province": "basin",
    "abyssal plain": "basin",
    "plain": "basin",
    "plateau": "basin",
    "terrace": "basin",
    "borderland": "basin",
    "province": "basin",
    "apron": "basin",
    "lobe": "basin",
    "fan": "basin",
    "levee": "basin",
    "canyon": "canyon",
    "canyons": "canyon",
    "canyon province": "canyon",
    "valley": "canyon",
    "valleys": "canyon",
    "sea valley": "canyon",
    "sea valleys": "canyon",
    "seachannel": "canyon",
    "seachannels": "canyon",
    "ground": "other",
    "finger": "other",
    "cap": "other",
    "bars": "other",
    "hole": "other",
    "mud volcano": "other",
    "pingos": "other",
    "continental slope": "other",
    "slope": "other",
    "promontory": "other",
    "sand ridge province": "other",
    "unknown": "other",
}

_GENERIC_ZH: dict[str, str] = {
    "seamount": "海山",
    "seamounts": "海山群",
    "seamount chain": "海山链",
    "seamount group": "海山群",
    "seamount province": "海山区",
    "guyot": "平顶海山",
    "guyots": "平顶海山",
    "knoll": "海丘",
    "knolls": "海丘群",
    "tablemount": "平顶海山",
    "peak": "海峰",
    "peaks": "海峰群",
    "cone": "火山锥",
    "mound": "海丘",
    "mounds": "海丘群",
    "hill": "山丘",
    "hills": "山丘群",
    "hill province": "山丘区",
    "bank": "浅滩",
    "banks": "浅滩群",
    "shoal": "浅滩",
    "shoals": "浅滩群",
    "reef": "礁",
    "reefs": "礁群",
    "caldera": "火山口",
    "calderas": "火山口群",
    "trench": "海沟",
    "trenches": "海沟群",
    "trough": "海槽",
    "troughs": "海槽群",
    "deep": "海渊",
    "deeps": "海渊群",
    "canyon": "海底峡谷",
    "canyons": "海底峡谷群",
    "canyon province": "峡谷区",
    "valley": "海底谷",
    "valleys": "海底谷群",
    "sea valley": "海底谷",
    "sea valleys": "海底谷群",
    "seachannel": "海底水道",
    "seachannels": "海底水道群",
    "channel": "水道",
    "channels": "水道群",
    "sea channel": "海底水道",
    "sea channels": "海底水道群",
    "passage": "通道",
    "gap": "隘口",
    "pass": "通道",
    "sill": "海槛",
    "saddle": "鞍部",
    "seabight": "海湾区",
    "re-entrant": "凹入区",
    "ridge": "海岭",
    "ridges": "海岭群",
    "rise": "海隆",
    "rises": "海隆群",
    "spur": "海山嘴",
    "spurs": "海山嘴群",
    "fracture zone": "断裂带",
    "fracture zones": "断裂带群",
    "fracture zone system": "断裂带系统",
    "fracture zone province": "断裂带区",
    "escarpment": "陡崖",
    "scarp": "陡坎",
    "scarps": "陡坎群",
    "rift": "裂谷",
    "salt dome province": "盐丘区",
    "discordance": "不整合面",
    "basin": "海盆",
    "basins": "海盆群",
    "basin and ridge province": "盆岭区",
    "abyssal plain": "深海平原",
    "plain": "平原",
    "plateau": "海底高原",
    "terrace": "阶地",
    "borderland": "边缘地",
    "province": "地貌区",
    "apron": "裙状地",
    "lobe": "叶状地",
    "fan": "海底扇",
    "levee": "天然堤",
    "ground": "浅滩",
    "finger": "指状地貌",
    "cap": "盖层",
    "bars": "沙坝",
    "hole": "海穴",
    "mud volcano": "泥火山",
    "pingos": "冰丘群",
    "continental slope": "大陆坡",
    "slope": "陆坡",
    "promontory": "海角",
    "sand ridge province": "沙脊区",
    "unknown": "未分类",
}

_CACHE_LOCK = threading.Lock()
_FEATURE_CACHE: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
_NAME_MAP_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}


class SeafloorFeatureError(RuntimeError):
    """海底地名数据不可用时抛出的业务异常。"""


def category_definitions() -> list[dict[str, Any]]:
    return [dict(item) for item in _CATEGORY_DEFS]


def category_keys() -> set[str]:
    return {item["key"] for item in _CATEGORY_DEFS}


def _category_color(key: str) -> str:
    for item in _CATEGORY_DEFS:
        if item["key"] == key:
            return item["color"]
    return "#9fb4c6"


def _normalise_term(value: Any) -> str:
    text = " ".join(str(value or "").replace("’", "'").split()).strip().lower()
    return text


def _resolve_category(term: str) -> str:
    direct = CATEGORY_MAP.get(term)
    if direct:
        return direct
    if term.endswith("ies") and CATEGORY_MAP.get(term[:-3] + "y"):
        return CATEGORY_MAP[term[:-3] + "y"]
    if term.endswith("es") and CATEGORY_MAP.get(term[:-2]):
        return CATEGORY_MAP[term[:-2]]
    if term.endswith("s") and CATEGORY_MAP.get(term[:-1]):
        return CATEGORY_MAP[term[:-1]]
    return "other"


def _generic_zh(term: str) -> str:
    return _GENERIC_ZH.get(term) or (term.title() if term else "")


def _cell_text(cell: ET.Element, shared: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        inline = cell.find(_NS + "is")
        if inline is not None:
            return "".join(node.text or "" for node in inline.iter(_NS + "t")).strip()
    value = cell.find(_NS + "v")
    if value is None or value.text is None:
        return ""
    text = value.text
    if cell_type == "s":
        try:
            return shared[int(text)].strip()
        except (ValueError, IndexError):
            return ""
    return text.strip()


def _read_sheet(path: str) -> list[dict[str, str]]:
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            shared: list[str] = []
            if "xl/sharedStrings.xml" in names:
                root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
                for item in root.findall(_NS + "si"):
                    shared.append("".join(node.text or "" for node in item.iter(_NS + "t")))
            sheets = sorted(
                name for name in names
                if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name)
            )
            if not sheets:
                raise SeafloorFeatureError("features.xlsx 中未找到工作表")
            sheet = ET.fromstring(archive.read(sheets[0]))
    except (OSError, zipfile.BadZipFile, ET.ParseError) as exc:
        raise SeafloorFeatureError(f"无法读取海底地名数据：{path}（{exc}）") from exc

    rows: list[dict[str, str]] = []
    for row in sheet.iter(_NS + "row"):
        values: dict[str, str] = {}
        for cell in row.findall(_NS + "c"):
            ref = str(cell.attrib.get("r") or "")
            match = re.match(r"[A-Z]+", ref)
            if not match:
                continue
            values[match.group()] = _cell_text(cell, shared)
        if values:
            rows.append(values)
    return rows


def _to_float(value: Any) -> float | None:
    text = str(value or "").strip().replace(",", "")
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _parse_point(value: Any) -> tuple[float, float] | None:
    text = str(value or "").strip()
    match = _POINT_RE.search(text) or _NUMBER_PAIR_RE.search(text)
    if not match:
        return None
    lon = _to_float(match.group(1))
    lat = _to_float(match.group(2))
    if lon is None or lat is None:
        return None
    if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
        return None
    return lon, lat


def _parse_dimension_km(value: Any) -> float | None:
    text = str(value or "")
    matches = [float(item) for item in _DIMENSION_RE.findall(text)]
    return max(matches) if matches else None


def _unwrap_geometry(text: str) -> str:
    text = text.strip()
    if not (text.startswith("(") and text.endswith(")")):
        return text
    depth = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[1:-1] if index == len(text) - 1 else text
    return text


def _split_top_level(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    parts.append(text[start:].strip())
    return [part for part in parts if part]


def _parse_coordinate(text: str) -> list[float] | None:
    numbers = re.findall(
        r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", _unwrap_geometry(text)
    )
    if len(numbers) < 2:
        return None
    lon = _to_float(numbers[0])
    lat = _to_float(numbers[1])
    if lon is None or lat is None:
        return None
    if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
        return None
    return [round(lon, 6), round(lat, 6)]


def _parse_coordinate_list(text: str) -> list[list[float]]:
    coordinates: list[list[float]] = []
    for part in _split_top_level(_unwrap_geometry(text)):
        coordinate = _parse_coordinate(part)
        if coordinate is not None:
            coordinates.append(coordinate)
    return coordinates


def _parse_polygon_rings(text: str) -> list[list[list[float]]]:
    rings: list[list[list[float]]] = []
    for part in _split_top_level(_unwrap_geometry(text)):
        ring = _parse_coordinate_list(part)
        if len(ring) >= 3:
            rings.append(ring)
    return rings


def _parse_wkt(text: str) -> list[dict[str, Any]]:
    match = re.match(r"^([A-Za-z]+)\s*(\(.*\))$", text.strip(), re.DOTALL)
    if not match:
        return []
    kind = match.group(1).upper()
    body = match.group(2)
    if kind == "GEOMETRYCOLLECTION":
        shapes: list[dict[str, Any]] = []
        for part in _split_top_level(_unwrap_geometry(body)):
            shapes.extend(_parse_wkt(part))
        return shapes
    if kind == "POINT":
        coordinate = _parse_coordinate(body)
        return [{"kind": "point", "coordinates": coordinate}] if coordinate else []
    if kind == "MULTIPOINT":
        points: list[dict[str, Any]] = []
        for part in _split_top_level(_unwrap_geometry(body)):
            coordinate = _parse_coordinate(part)
            if coordinate is not None:
                points.append({"kind": "point", "coordinates": coordinate})
        return points
    if kind == "LINESTRING":
        coordinates = _parse_coordinate_list(body)
        return [{"kind": "line", "coordinates": coordinates}] if len(coordinates) >= 2 else []
    if kind == "MULTILINESTRING":
        lines: list[dict[str, Any]] = []
        for part in _split_top_level(_unwrap_geometry(body)):
            coordinates = _parse_coordinate_list(part)
            if len(coordinates) >= 2:
                lines.append({"kind": "line", "coordinates": coordinates})
        return lines
    if kind == "POLYGON":
        rings = _parse_polygon_rings(body)
        return [{"kind": "polygon", "rings": rings}] if rings else []
    if kind == "MULTIPOLYGON":
        polygons: list[dict[str, Any]] = []
        for part in _split_top_level(_unwrap_geometry(body)):
            rings = _parse_polygon_rings(part)
            if rings:
                polygons.append({"kind": "polygon", "rings": rings})
        return polygons
    return []


def _parse_boundary(value: Any) -> list[dict[str, Any]]:
    text = str(value or "").strip()
    if not text or "(" not in text:
        return []
    return [
        shape for shape in _parse_wkt(text)
        if shape["kind"] in {"polygon", "line"}
    ]


def _boundary_stats(boundaries: list[dict[str, Any]]) -> tuple[float, list[float] | None]:
    min_lat = math.inf
    max_lat = -math.inf
    longitudes: list[float] = []
    for shape in boundaries:
        rings = shape.get("rings") if shape["kind"] == "polygon" else [shape.get("coordinates")]
        for ring in rings or []:
            for coordinate in ring or []:
                lon, lat = coordinate
                longitudes.append(((lon % 360.0) + 360.0) % 360.0)
                min_lat = min(min_lat, lat)
                max_lat = max(max_lat, lat)
    if not longitudes or not math.isfinite(min_lat):
        return 0.0, None
    longitudes.sort()
    west = 0.0
    width = 0.0
    if len({round(value, 9) for value in longitudes}) > 1:
        max_gap = -1.0
        gap_index = 0
        for index, lon in enumerate(longitudes):
            next_lon = longitudes[(index + 1) % len(longitudes)]
            gap = (next_lon - lon + 360.0) % 360.0
            if gap > max_gap:
                max_gap = gap
                gap_index = index
        west360 = longitudes[(gap_index + 1) % len(longitudes)]
        width = 360.0 - max_gap
        west = west360 - 360.0 if west360 > 180.0 else west360
    else:
        west360 = longitudes[0] % 360.0
        west = west360 - 360.0 if west360 > 180.0 else west360
    if width >= 360.0 - 1e-7:
        west = -180.0
        east = 180.0
    else:
        east = west + max(0.0, width)
    lat_mid = (min_lat + max_lat) / 2.0
    width_m = width * 111320.0 * max(0.05, math.cos(math.radians(lat_mid)))
    height_m = (max_lat - min_lat) * 110574.0
    bbox = [
        round(west, 6), round(min_lat, 6),
        round(east, 6), round(max_lat, 6),
    ]
    return max(width_m, height_m), bbox


def _lon_segments(west: float, east: float) -> list[tuple[float, float]]:
    if east - west >= 360.0:
        return [(-180.0, 180.0)]
    if west <= east:
        if east <= 180.0:
            return [(west, east)]
        return [(west, 180.0), (-180.0, east - 360.0)]
    return [(west, 180.0), (-180.0, east)]


def _build_feature(row: Mapping[str, str], index: int, origin_max_chars: int) -> dict[str, Any] | None:
    name = str(row.get("A") or "").strip()
    point = _parse_point(row.get("D"))
    if not name or not point:
        return None
    lon, lat = point
    generic_term = _normalise_term(row.get("B"))
    category = _resolve_category(generic_term)
    min_depth = _to_float(row.get("F"))
    max_depth = _to_float(row.get("G"))
    relief = _to_float(row.get("H"))
    dimension = str(row.get("I") or "").strip()
    dimension_km = _parse_dimension_km(dimension)
    boundaries = _parse_boundary(row.get("E"))
    boundary_extent, boundary_bbox = _boundary_stats(boundaries) if boundaries else (0.0, None)
    extent_m = max(
        relief if relief is not None else 0.0,
        (dimension_km or 0.0) * 1000.0 * 0.25,
        boundary_extent,
    )
    importance = math.log1p(max(0.0, extent_m)) + math.log1p(max(0.0, (dimension_km or 0.0)))
    origin = " ".join(str(row.get("C") or "").split())
    if origin_max_chars > 0 and len(origin) > origin_max_chars:
        origin = origin[:origin_max_chars].rstrip() + "…"
    return {
        "id": f"scufn-{index:05d}",
        "name": name,
        "generic_term": generic_term,
        "generic_term_zh": _generic_zh(generic_term),
        "category": category,
        "category_label": next(
            (item["label"] for item in _CATEGORY_DEFS if item["key"] == category),
            "其他",
        ),
        "category_color": _category_color(category),
        "longitude": round(lon, 6),
        "latitude": round(lat, 6),
        "min_depth_m": min_depth,
        "max_depth_m": max_depth,
        "total_relief_m": relief,
        "dimension": dimension,
        "extent_m": round(extent_m, 2),
        "boundary_extent_m": round(boundary_extent, 2),
        "boundary_bbox": boundary_bbox,
        "has_boundary": bool(boundaries),
        "boundaries": boundaries,
        "importance": round(importance, 4),
        "origin": origin,
        "name_zh": "",
        "name_zh_source": "",
    }


def _load_name_map(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    absolute = os.path.abspath(path)
    if not os.path.isfile(absolute):
        return {}
    stat = os.stat(absolute)
    cache_key = (absolute, stat.st_mtime_ns, stat.st_size)
    with _CACHE_LOCK:
        cached = _NAME_MAP_CACHE.get(cache_key)
    if cached is not None:
        return cached
    try:
        with open(absolute, "r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, ValueError):
        return {}
    mapping = payload.get("mapping") if isinstance(payload, Mapping) else None
    result = mapping if isinstance(mapping, Mapping) else {}
    with _CACHE_LOCK:
        _NAME_MAP_CACHE.clear()
        _NAME_MAP_CACHE[cache_key] = result
    return result


def _load_features(
    path: str,
    origin_max_chars: int,
    name_map_path: str | None = None,
) -> list[dict[str, Any]]:
    absolute = os.path.abspath(path)
    if not os.path.isfile(absolute):
        raise SeafloorFeatureError(f"海底地名数据文件不存在：{absolute}")
    stat = os.stat(absolute)
    name_map = _load_name_map(name_map_path)
    name_absolute = os.path.abspath(name_map_path) if name_map_path else ""
    name_stat = os.stat(name_absolute) if name_absolute and os.path.isfile(name_absolute) else None
    cache_key = (
        absolute, stat.st_mtime_ns, stat.st_size, origin_max_chars,
        name_absolute,
        name_stat.st_mtime_ns if name_stat else 0,
        name_stat.st_size if name_stat else 0,
    )
    with _CACHE_LOCK:
        cached = _FEATURE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    rows = _read_sheet(absolute)
    features: list[dict[str, Any]] = []
    for index, row in enumerate(rows[1:], start=1):
        feature = _build_feature(row, index, origin_max_chars)
        if feature is not None:
            entry = name_map.get(feature["id"])
            if isinstance(entry, Mapping):
                name_zh = str(entry.get("name_zh") or "").strip()
                if name_zh:
                    feature["name_zh"] = name_zh
                    feature["name_zh_source"] = "民政部国家地名信息库"
            features.append(feature)
    with _CACHE_LOCK:
        _FEATURE_CACHE.clear()
        _FEATURE_CACHE[cache_key] = features
    return features


def _in_bbox(lon: float, lat: float, west: float, south: float, east: float, north: float) -> bool:
    if lat < south or lat > north:
        return False
    if west <= east:
        return west <= lon <= east
    return lon >= west or lon <= east


def _bbox_intersects(
    bbox: list[float] | None,
    west: float, south: float, east: float, north: float,
) -> bool:
    if not bbox or len(bbox) < 4:
        return False
    b_west, b_south, b_east, b_north = bbox
    if b_north < south or b_south > north:
        return False
    for left, right in _lon_segments(west, east):
        for feature_left, feature_right in _lon_segments(b_west, b_east):
            if not (right < feature_left or left > feature_right):
                return True
    return False


def _normalise_categories(categories: Iterable[str] | None) -> set[str] | None:
    if not categories:
        return None
    selected = {str(item).strip().lower() for item in categories if str(item).strip()}
    unknown = selected - category_keys()
    if unknown:
        raise SeafloorFeatureError(f"未知的地貌类型：{', '.join(sorted(unknown))}")
    return selected or None


def query_features(
    path: str,
    *,
    west: float,
    south: float,
    east: float,
    north: float,
    categories: Iterable[str] | None = None,
    min_extent_m: float | None = None,
    limit: int = 500,
    origin_max_chars: int = 600,
    name_map_path: str | None = None,
) -> dict[str, Any]:
    selected_categories = _normalise_categories(categories)
    features = _load_features(path, origin_max_chars, name_map_path)
    limit = max(1, min(2000, int(limit)))

    matched: list[dict[str, Any]] = []
    for feature in features:
        point_inside = _in_bbox(
            float(feature["longitude"]), float(feature["latitude"]),
            west, south, east, north,
        )
        boundary_inside = _bbox_intersects(
            feature.get("boundary_bbox"), west, south, east, north,
        )
        if not point_inside and not boundary_inside:
            continue
        if selected_categories and feature["category"] not in selected_categories:
            continue
        if min_extent_m is not None and float(feature["extent_m"]) < min_extent_m:
            continue
        matched.append(feature)

    matched.sort(key=lambda item: (-float(item["importance"]), str(item["name"])))
    counts = Counter(item["category"] for item in matched)
    selected = matched[:limit]
    categories_payload = [
        {
            "key": item["key"],
            "label": item["label"],
            "color": item["color"],
            "count": int(counts.get(item["key"], 0)),
        }
        for item in _CATEGORY_DEFS
    ]
    return {
        "source": "GEBCO SCUFN",
        "source_file": os.path.basename(path),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bbox": {
            "west": west, "south": south, "east": east, "north": north,
        },
        "total": len(matched),
        "count": len(selected),
        "truncated": len(selected) < len(matched),
        "categories": categories_payload,
        "features": selected,
    }
