"""Leica time-lapse droplet analysis pipeline."""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tifffile
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from scipy import ndimage as ndi
from skimage import exposure, filters, measure, morphology, transform
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks
from xml.etree import ElementTree as ET


LOGGER = logging.getLogger(__name__)


FLOAT_PATTERN = re.compile(r"-?[0-9]+(?:[.,][0-9]+)?(?:[eE][-+]?[0-9]+)?")


def _strip_namespace(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _first_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    match = FLOAT_PATTERN.search(text)
    if not match:
        return None
    token = match.group(0).replace(",", ".")
    try:
        return float(token)
    except ValueError:
        return None


def _first_int(value: object) -> Optional[int]:
    float_val = _first_float(value)
    if float_val is None:
        return None
    try:
        return int(round(float_val))
    except (TypeError, ValueError):
        return None


def _parse_float_list(text: str) -> Optional[List[float]]:
    if not text:
        return None
    matches = FLOAT_PATTERN.findall(text)
    if not matches:
        return None
    values: List[float] = []
    for token in matches:
        token = token.replace(",", ".")
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values or None


def _parse_color_value(value: str) -> Optional[Dict[str, object]]:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    if re.match(r"^[0-9]+[ ,][0-9]+[ ,][0-9]+", text):
        parts = re.split(r"[ ,]+", text)
        rgb = []
        for part in parts[:3]:
            try:
                val = float(part)
            except ValueError:
                continue
            if val <= 1.0:
                val *= 255.0
            rgb.append(int(round(max(0.0, min(255.0, val)))))
        if len(rgb) == 3:
            return {"r": rgb[0], "g": rgb[1], "b": rgb[2]}
    if text.startswith("#") and len(text) in {7, 9}:
        try:
            r, g, b = [int(text[i : i + 2], 16) for i in (1, 3, 5)]
            return {"r": r, "g": g, "b": b}
        except ValueError:
            return None
    try:
        rgb_float = mcolors.to_rgb(text)
        rgb = [int(round(channel * 255.0)) for channel in rgb_float]
        return {"r": rgb[0], "g": rgb[1], "b": rgb[2]}
    except Exception:
        return {"name": text}


def _convert_time_to_seconds(value: object, unit: Optional[str] = None) -> Optional[float]:
    seconds = _first_float(value)
    if seconds is None:
        return None
    if unit:
        u = unit.lower()
        if u.startswith("ms"):
            return seconds / 1000.0
        if u.startswith("us") or u.startswith("µs"):
            return seconds / 1_000_000.0
        if u.startswith("min"):
            return seconds * 60.0
        if u.startswith("h"):
            return seconds * 3600.0
    # Heuristic: very small values are likely seconds already for time-lapse gaps
    return seconds


def _convert_exposure_to_ms(value: object, unit: Optional[str] = None) -> Optional[float]:
    exposure = _first_float(value)
    if exposure is None:
        return None
    if unit:
        u = unit.lower()
        if u.startswith("ms"):
            return exposure
        if u.startswith("us") or u.startswith("µs"):
            return exposure / 1000.0
        if u.startswith("s"):
            return exposure * 1000.0
    if exposure <= 5.0:
        return exposure * 1000.0
    return exposure


def _convert_length_to_um(value: object, unit_hint: Optional[str] = None) -> Optional[float]:
    length = _first_float(value)
    if length is None:
        return None
    if unit_hint:
        unit = unit_hint.lower()
        if unit in {"m", "meter", "metre"}:
            return length * 1_000_000.0
        if unit in {"mm", "millimeter", "millimetre"}:
            return length * 1000.0
        if unit in {"nm", "nanometer", "nanometre"}:
            return length / 1000.0
        if unit in {"um", "µm", "micrometer", "micrometre"}:
            return length
    if abs(length) < 1e-3:
        return length * 1_000_000.0
    if abs(length) < 10:
        return length
    return length  # assume already in micrometers


def _parse_color_from_element(attr: Dict[str, str], element: ET.Element) -> Optional[Dict[str, object]]:
    for key in ("displaycolor", "color", "colorvalue", "lut", "displaycolorvalue"):
        if key in attr:
            parsed = _parse_color_value(attr[key])
            if parsed:
                return parsed
    rgb_components = {}
    for key, value in attr.items():
        lowered = key.lower()
        if lowered in {"r", "red"}:
            rgb_components["r"] = _first_int(value)
        elif lowered in {"g", "green"}:
            rgb_components["g"] = _first_int(value)
        elif lowered in {"b", "blue"}:
            rgb_components["b"] = _first_int(value)
    if len(rgb_components) == 3:
        return {
            "r": int(rgb_components["r"] or 0),
            "g": int(rgb_components["g"] or 0),
            "b": int(rgb_components["b"] or 0),
        }
    for child in element:
        child_tag = _strip_namespace(child.tag).lower()
        child_attr = { _strip_namespace(k).lower(): v for k, v in child.attrib.items() }
        if child_tag in {"color", "displaycolor"}:
            parsed = _parse_color_from_element(child_attr, child)
            if parsed:
                return parsed
        if child_tag in {"r", "g", "b"}:
            rgb_components[child_tag] = _first_int(child.text)
    if len(rgb_components) == 3:
        return {
            "r": int(rgb_components.get("r", 0)),
            "g": int(rgb_components.get("g", 0)),
            "b": int(rgb_components.get("b", 0)),
        }
    return None


def _find_xml_files(meta_dir: Path) -> List[Path]:
    search_dirs: List[Path] = []
    if meta_dir.exists():
        if meta_dir.is_dir():
            search_dirs.append(meta_dir)
            parent = meta_dir.parent
            if parent not in search_dirs:
                search_dirs.append(parent)
        else:
            search_dirs.append(meta_dir.parent)
    else:
        search_dirs.append(meta_dir)
    xml_paths: List[Path] = []
    seen = set()
    for directory in search_dirs:
        if not directory.exists() or not directory.is_dir():
            continue
        for path in directory.glob("*.xml"):
            if path not in seen:
                seen.add(path)
                xml_paths.append(path)
        for path in directory.rglob("*.xml"):
            if path not in seen:
                seen.add(path)
                xml_paths.append(path)
    return sorted(xml_paths)


def _select_xml_pair(xml_paths: Sequence[Path]) -> Tuple[Optional[Path], Optional[Path]]:
    if not xml_paths:
        return (None, None)
    properties_candidates = [p for p in xml_paths if "properties" in p.stem.lower()]
    if properties_candidates:
        for prop in properties_candidates:
            base_stem = re.sub(r"_?properties", "", prop.stem, flags=re.IGNORECASE)
            siblings = [
                candidate
                for candidate in xml_paths
                if candidate != prop
                and re.sub(r"_?properties", "", candidate.stem, flags=re.IGNORECASE)
                == base_stem
            ]
            if siblings:
                sibling = min(siblings, key=lambda p: len(p.name))
                return (prop, sibling)
        return (properties_candidates[0], None)
    return (None, xml_paths[0])


def _fallback_px_size_from_tiff(any_tiff: Path) -> Optional[float]:
    try:
        with tifffile.TiffFile(any_tiff) as tif:
            page = tif.pages[0]
            description = page.description or ""
            if description:
                match = re.search(
                    r"PhysicalSizeX\s*=\s*\"?(?P<value>[0-9]+\.?[0-9]*)\"?\s*(?P<unit>[A-Za-zµ]+)?",
                    description,
                    re.IGNORECASE,
                )
                if match:
                    value = _first_float(match.group("value"))
                    unit = match.group("unit")
                    if value is not None:
                        return _convert_length_to_um(value, unit)
            if "PhysicalSizeX" in page.tags:
                phys_tag = page.tags["PhysicalSizeX"].value
                if isinstance(phys_tag, (tuple, list)):
                    phys_tag = phys_tag[0]
                return _convert_length_to_um(phys_tag, "um")
            res_unit = page.tags.get("ResolutionUnit")
            x_res = page.tags.get("XResolution")
            if res_unit is not None and x_res is not None:
                num, den = x_res.value
                resolution = num / den if den else float(num)
                unit_value = res_unit.value
                if unit_value == 2:  # inch
                    return 25_400.0 / resolution
                if unit_value == 3:  # centimeter
                    return 10_000.0 / resolution
    except Exception as exc:  # pragma: no cover - robustness
        LOGGER.warning("Failed to read TIFF metadata from %s: %s", any_tiff, exc)
    return None


def parse_metadata(meta_dir: Path, any_tiff: Path) -> Dict[str, object]:
    """Parse Leica metadata from LAS X XML files and TIFF fallbacks."""

    meta_dir = Path(meta_dir)
    any_tiff = Path(any_tiff)
    if meta_dir.is_dir() and meta_dir.name.lower() in {"metadata"}:
        run_root = meta_dir.parent
    elif meta_dir.is_file():
        run_root = meta_dir.parent
    else:
        run_root = meta_dir

    xml_paths = _find_xml_files(meta_dir)
    properties_path, main_xml_path = _select_xml_pair(xml_paths)
    other_xml_paths = [
        path
        for path in xml_paths
        if path not in {p for p in [properties_path, main_xml_path] if p is not None}
    ]

    LOGGER.info(
        "Found %d XML metadata files (properties=%s, main=%s)",
        len(xml_paths),
        properties_path,
        main_xml_path,
    )

    roots: List[Tuple[int, str, ET.Element]] = []

    def _load_xml(path: Optional[Path], priority: int) -> None:
        if path is None:
            return
        try:
            root = ET.parse(path).getroot()
            roots.append((priority, path.name, root))
        except Exception as exc:  # pragma: no cover - malformed XML
            LOGGER.warning("Failed to parse XML %s: %s", path, exc)

    _load_xml(main_xml_path, 0)
    _load_xml(properties_path, 1)
    for extra in other_xml_paths:
        _load_xml(extra, -1)

    channels: Dict[str, ChannelInfo] = {}
    channel_priority: Dict[str, int] = {}
    channel_order: List[str] = []
    exposures_map: Dict[str, float] = {}
    exposure_priority: Dict[str, int] = {}

    dt_s: Optional[float] = None
    dt_priority = -1
    cycle_count: Optional[int] = None
    cycle_priority = -1
    timestamps: Optional[List[float]] = None
    timestamp_priority = -1

    objective_name: Optional[str] = None
    objective_priority = -1
    objective_mag: Optional[float] = None
    objective_mag_priority = -1
    objective_na: Optional[float] = None
    objective_na_priority = -1
    immersion: Optional[str] = None
    immersion_priority = -1

    camera_model: Optional[str] = None
    camera_priority = -1
    px_sensor_values: List[float] = []
    px_sensor_priority = -1
    px_size_um: Optional[float] = None
    px_size_priority = -1

    image_size: Optional[Tuple[int, int]] = None
    image_size_priority = -1
    binning: Optional[Tuple[int, int]] = None
    binning_priority = -1
    video_mag: Optional[float] = None
    video_mag_priority = -1

    tiles: Dict[str, object] = {}
    tiles_priority = -1
    stage: Dict[str, float] = {}
    stage_priority: Dict[str, int] = {}

    def _update_channel(
        name: str,
        priority: int,
        source: str,
        *,
        color: Optional[Dict[str, object]] = None,
        index: Optional[int] = None,
        exposure: Optional[float] = None,
    ) -> None:
        key = name.strip()
        if not key:
            return
        key_upper = key.upper()
        info = channels.get(key_upper)
        if info is None:
            info = ChannelInfo(name=key)
            channels[key_upper] = info
            channel_priority[key_upper] = priority
            channel_order.append(key_upper)
            LOGGER.info("Discovered channel '%s' from %s", key, source)
        if color and (info.color is None or priority >= channel_priority[key_upper]):
            info.color = color
        if index is not None and (info.index is None or priority >= channel_priority[key_upper]):
            info.index = index
        if exposure is not None:
            info.exposure_ms = exposure
            prev_priority = exposure_priority.get(key_upper, -1)
            if priority >= prev_priority:
                exposures_map[key] = exposure
                exposure_priority[key_upper] = priority

    def _update_value(
        current_value: Optional[object],
        current_priority: int,
        new_value: Optional[object],
        new_priority: int,
        field_name: str,
        source: str,
    ) -> Tuple[Optional[object], int]:
        if new_value is None:
            return current_value, current_priority
        if current_value is None or new_priority >= current_priority:
            LOGGER.info("Metadata field '%s' set to %s from %s", field_name, new_value, source)
            return new_value, new_priority
        return current_value, current_priority

    def _update_stage_value(axis: str, value: Optional[float], priority: int, source: str) -> None:
        if value is None:
            return
        current_priority = stage_priority.get(axis, -1)
        if axis not in stage or priority >= current_priority:
            stage[axis] = value
            stage_priority[axis] = priority
            LOGGER.info("Stage %s set to %.3f um from %s", axis, value, source)

    def _update_tiles(field: str, value: object, priority: int, source: str) -> None:
        nonlocal tiles_priority
        if value is None:
            return
        if field not in tiles or priority >= tiles_priority:
            tiles[field] = value
            tiles_priority = priority
            LOGGER.info("Tile metadata '%s' set from %s", field, source)

    for priority, source, root in sorted(roots, key=lambda item: item[0]):
        if root is None:
            continue
        for element in root.iter():
            tag_lower = _strip_namespace(element.tag).lower()
            attr = {
                _strip_namespace(key).lower(): str(value)
                for key, value in element.attrib.items()
            }
            text = (element.text or "").strip()

            channel_name = None
            if "channel" in tag_lower:
                channel_name = (
                    attr.get("channelname")
                    or attr.get("name")
                    or attr.get("id")
                    or attr.get("channel")
                )
                if not channel_name:
                    for child in element:
                        child_tag = _strip_namespace(child.tag).lower()
                        if child_tag in {"name", "channelname", "label"}:
                            channel_name = (child.text or "").strip()
                            if channel_name:
                                break
            if channel_name and channel_name.strip():
                color = _parse_color_from_element(attr, element)
                index = None
                for key in ("index", "order", "channelindex", "id"):
                    if key in attr:
                        index = _first_int(attr[key])
                        if index is not None:
                            break
                exposure = None
                units_map = {
                    key[:-4]: value
                    for key, value in attr.items()
                    if key.endswith("unit")
                }
                for key, value in attr.items():
                    if key.endswith("unit"):
                        continue
                    if "exposure" in key or key in {"exptime", "exposuretime"}:
                        unit = units_map.get(key) or attr.get("unit")
                        exposure = _convert_exposure_to_ms(value, unit)
                        if exposure is not None:
                            break
                if exposure is None:
                    for child in element:
                        child_tag = _strip_namespace(child.tag).lower()
                        child_attr = {
                            _strip_namespace(k).lower(): str(v)
                            for k, v in child.attrib.items()
                        }
                        child_text = (child.text or child_attr.get("value") or "").strip()
                        if "exposure" in child_tag or any(
                            "exposure" in key for key in child_attr.keys()
                        ):
                            unit = child_attr.get("unit") or attr.get("unit")
                            exposure = _convert_exposure_to_ms(child_text, unit)
                            if exposure is not None:
                                break
                _update_channel(
                    channel_name,
                    priority,
                    source,
                    color=color,
                    index=index,
                    exposure=exposure,
                )

            channel_ref = (
                attr.get("channelname")
                or attr.get("channel")
                or attr.get("name")
                if "channel" in tag_lower
                else None
            )
            if channel_ref and ("exposure" in tag_lower or any("exposure" in key for key in attr)):
                unit = attr.get("unit")
                for key, value in attr.items():
                    if "exposure" in key or key in {"exptime", "exposuretime"}:
                        exposure_val = _convert_exposure_to_ms(value, unit)
                        if exposure_val is not None:
                            _update_channel(channel_ref, priority, source, exposure=exposure_val)

            if "cyclecount" in tag_lower or "numberoftimes" in tag_lower:
                value = _first_int(text or attr.get("value"))
                cycle_count, cycle_priority = _update_value(
                    cycle_count, cycle_priority, value, priority, "cycle_count", source
                )
            for key, value in attr.items():
                key_lower = key.lower()
                if "cyclecount" in key_lower or key_lower in {"frames", "framecount"}:
                    parsed = _first_int(value)
                    cycle_count, cycle_priority = _update_value(
                        cycle_count, cycle_priority, parsed, priority, "cycle_count", source
                    )
                if any(token in key_lower for token in ["cycletime", "timeinterval", "frametime", "timeinterval"]):
                    parsed = _convert_time_to_seconds(value, attr.get(f"{key}unit"))
                    dt_s, dt_priority = _update_value(
                        dt_s, dt_priority, parsed, priority, "dt_s", source
                    )
                if key_lower.endswith("unit") and "cycle" in key_lower:
                    unit_hint = value
                    if "cycletime" in attr:
                        parsed = _convert_time_to_seconds(attr.get("cycletime"), unit_hint)
                        dt_s, dt_priority = _update_value(
                            dt_s, dt_priority, parsed, priority, "dt_s", source
                        )
                if key_lower.endswith("unit") and any(
                    token in key_lower for token in ["timeinterval", "frametime"]
                ):
                    base_key = key_lower.replace("unit", "")
                    base_value = attr.get(base_key)
                    if base_value is not None:
                        parsed = _convert_time_to_seconds(base_value, value)
                        dt_s, dt_priority = _update_value(
                            dt_s, dt_priority, parsed, priority, "dt_s", source
                        )

            if "timestamp" in tag_lower:
                combined_parts = [text] + list(attr.values())
                unit_hint = None
                for key, value in attr.items():
                    if key.lower().endswith("unit"):
                        unit_hint = value
                        break
                for child in element:
                    combined_parts.append((child.text or "").strip())
                    combined_parts.extend(str(v) for v in child.attrib.values())
                parsed_list = _parse_float_list(" ".join(filter(None, combined_parts)))
                if parsed_list:
                    if unit_hint:
                        unit_lower = unit_hint.lower()
                        if unit_lower.startswith("ms"):
                            parsed_list = [value / 1000.0 for value in parsed_list]
                        elif unit_lower.startswith("us") or unit_lower.startswith("µs"):
                            parsed_list = [value / 1_000_000.0 for value in parsed_list]
                        elif unit_lower.startswith("min"):
                            parsed_list = [value * 60.0 for value in parsed_list]
                        elif unit_lower.startswith("h"):
                            parsed_list = [value * 3600.0 for value in parsed_list]
                    timestamps, timestamp_priority = _update_value(
                        timestamps, timestamp_priority, parsed_list, priority, "timestamps", source
                    )

            if "objective" in tag_lower or any("objective" in key for key in attr):
                candidate_name = attr.get("name") or attr.get("objective") or text
                objective_name, objective_priority = _update_value(
                    objective_name, objective_priority, candidate_name, priority, "objective_name", source
                )
                mag = None
                na = None
                immersion_val = None
                for key, value in attr.items():
                    key_lower = key.lower()
                    if key_lower in {"magnification", "mag"} or key_lower.endswith("magnification"):
                        mag = _first_float(value)
                    if "numericalaperture" in key_lower or key_lower == "na":
                        na = _first_float(value)
                    if "immersion" in key_lower:
                        immersion_val = value.strip()
                if candidate_name:
                    name_lower = candidate_name.lower()
                    if mag is None:
                        match = re.search(r"(\d+(?:\.\d+)?)\s*[x×]", candidate_name)
                        if match:
                            mag = float(match.group(1))
                    if na is None:
                        match = re.search(r"/(\d+(?:\.\d+)?)", candidate_name)
                        if match:
                            na = float(match.group(1))
                    if immersion_val is None:
                        for token in ["dry", "oil", "water", "glycerol"]:
                            if token in name_lower:
                                immersion_val = token.upper()
                                break
                objective_mag, objective_mag_priority = _update_value(
                    objective_mag, objective_mag_priority, mag, priority, "objective_mag", source
                )
                objective_na, objective_na_priority = _update_value(
                    objective_na, objective_na_priority, na, priority, "objective_na", source
                )
                if immersion_val:
                    immersion, immersion_priority = _update_value(
                        immersion, immersion_priority, immersion_val.upper(), priority, "immersion", source
                    )

            if "camera" in tag_lower:
                candidate_model = attr.get("name") or attr.get("model") or text
                camera_model, camera_priority = _update_value(
                    camera_model, camera_priority, candidate_model, priority, "camera_model", source
                )
                for key, value in attr.items():
                    key_lower = key.lower()
                    if "sensorpixelsize" in key_lower:
                        converted = _convert_length_to_um(value, attr.get("unit"))
                        if converted is not None:
                            px_sensor_values.append(converted)
                            px_sensor_priority = max(px_sensor_priority, priority)
                    if "binningx" == key_lower or key_lower.endswith("binningx"):
                        bx = _first_int(value)
                        by = None
                        if "binningy" in attr:
                            by = _first_int(attr.get("binningy"))
                        if bx is not None:
                            binning, binning_priority = _update_value(
                                binning, binning_priority, (bx, by or bx), priority, "binning", source
                            )
                    if key_lower.endswith("binningy") and "binningx" not in attr:
                        by = _first_int(value)
                        if by is not None:
                            binning, binning_priority = _update_value(
                                binning, binning_priority, (by, by), priority, "binning", source
                            )
                    if "videomag" in key_lower or "zoom" in key_lower:
                        mag_val = _first_float(value)
                        video_mag, video_mag_priority = _update_value(
                            video_mag, video_mag_priority, mag_val, priority, "video_mag", source
                        )

            if any(token in tag_lower for token in ["frame", "image", "size"]):
                width = None
                height = None
                for key, value in attr.items():
                    key_lower = key.lower()
                    if key_lower in {"width", "sizex", "imagesizex", "pixelwidth", "sizeinx"}:
                        width = _first_int(value)
                    if key_lower in {"height", "sizey", "imagesizey", "pixelheight", "sizeiny"}:
                        height = _first_int(value)
                if width and height:
                    image_size, image_size_priority = _update_value(
                        image_size, image_size_priority, (width, height), priority, "image_size", source
                    )

            if "tile" in tag_lower or "stitch" in tag_lower:
                grid_x = None
                grid_y = None
                overlap_x = None
                overlap_y = None
                stitching_enabled = None
                for key, value in attr.items():
                    key_lower = key.lower()
                    if key_lower in {"gridx", "tilecountx", "columns"} or (
                        "grid" in key_lower and key_lower.endswith("x")
                    ):
                        grid_x = _first_int(value)
                    if key_lower in {"gridy", "tilecounty", "rows"} or (
                        "grid" in key_lower and key_lower.endswith("y")
                    ):
                        grid_y = _first_int(value)
                    if "overlap" in key_lower and key_lower.endswith("x"):
                        overlap_x = _first_float(value)
                    if "overlap" in key_lower and key_lower.endswith("y"):
                        overlap_y = _first_float(value)
                    if key_lower in {"autostitch", "stitchingenabled", "applystitching"}:
                        stitching_enabled = value.lower() in {"true", "1", "yes"}
                if grid_x or grid_y:
                    _update_tiles("grid_xy", [grid_x or 1, grid_y or 1], priority, source)
                if overlap_x is not None or overlap_y is not None:
                    _update_tiles(
                        "overlap_xy_pct",
                        [overlap_x or 0.0, overlap_y or overlap_x or 0.0],
                        priority,
                        source,
                    )
                if stitching_enabled is not None:
                    _update_tiles("stitching_enabled", stitching_enabled, priority, source)

            if "stage" in tag_lower or "position" in tag_lower:
                for key, value in attr.items():
                    key_lower = key.lower()
                    if key_lower.endswith("x") and ("stage" in key_lower or "position" in key_lower):
                        _update_stage_value("x_um", _convert_length_to_um(value, attr.get("unit")), priority, source)
                    if key_lower.endswith("y") and ("stage" in key_lower or "position" in key_lower):
                        _update_stage_value("y_um", _convert_length_to_um(value, attr.get("unit")), priority, source)
                    if key_lower.endswith("z") and (
                        "stage" in key_lower or "focus" in key_lower or "position" in key_lower
                    ):
                        _update_stage_value("z_um", _convert_length_to_um(value, attr.get("unit")), priority, source)
                if "focus" in tag_lower and text:
                    _update_stage_value("z_um", _convert_length_to_um(text, attr.get("unit")), priority, source)

    if not channels:
        LOGGER.warning("No channels discovered from XML metadata; attempting filename inference")
        tif_files = sorted(run_root.rglob("*.tif")) if run_root.exists() else []
        inferred_channels = infer_channels_from_filenames(tif_files)
        for channel in inferred_channels:
            _update_channel(channel.name, -1, "filename", color=channel.color, index=channel.index)

    channel_list = [channels[key].to_dict() for key in channel_order if key in channels]

    if not channel_list:
        raise RuntimeError("No channels could be determined from metadata or filenames.")

    # Exposure map fallback to channel order
    for channel in channel_list:
        name = channel["name"]
        upper = name.upper()
        if name not in exposures_map and upper in channels and channels[upper].exposure_ms is not None:
            exposures_map[name] = channels[upper].exposure_ms  # type: ignore[index]

    for channel in channel_list:
        name = channel["name"]
        if name not in exposures_map:
            LOGGER.warning("Missing exposure for channel '%s'", name)

    if timestamps is None and cycle_count and dt_s:
        LOGGER.warning("Timestamps missing; synthesizing from dt_s * frame index")
        timestamps = [i * dt_s for i in range(cycle_count)]

    px_sensor_um = None
    if px_sensor_values:
        px_sensor_um = sum(px_sensor_values) / len(px_sensor_values)

    if px_sensor_um is not None:
        binning_x = binning[0] if binning else 1
        total_mag = None
        if video_mag:
            total_mag = video_mag
        elif objective_mag:
            total_mag = objective_mag
        if total_mag and total_mag > 0:
            px_size_calc = (px_sensor_um * (binning_x or 1)) / total_mag
            px_size_um, px_size_priority = _update_value(
                px_size_um, px_size_priority, px_size_calc, px_sensor_priority, "px_size_um", "xml"
            )
        else:
            LOGGER.warning(
                "Cannot compute sample pixel size: missing total magnification (video or objective)."
            )

    if px_size_um is None:
        px_size_um = _fallback_px_size_from_tiff(any_tiff)
        if px_size_um is not None:
            LOGGER.info("Pixel size derived from TIFF fallback: %.4f um", px_size_um)

    if px_size_um is None:
        raise RuntimeError("Pixel size (px_size_um) could not be determined from metadata or TIFF tags.")

    if image_size is None:
        try:
            with tifffile.TiffFile(any_tiff) as tif:
                page = tif.pages[0]
                shape = getattr(page, "shape", None)
                if shape and len(shape) >= 2:
                    height, width = shape[-2:]
                    image_size = (int(width), int(height))
        except Exception as exc:  # pragma: no cover - fallback safety
            LOGGER.warning("Failed to read image dimensions from TIFF %s: %s", any_tiff, exc)
    if image_size is None:
        image_size = (0, 0)

    if binning is None:
        binning = (1, 1)

    if properties_path and main_xml_path:
        assert len(channel_list) >= 1, "Expected at least one channel when XML files present"
        if dt_s is None or dt_s <= 0:
            raise AssertionError("Time interval dt_s should be positive when metadata XML is available")
        if cycle_count is None or cycle_count < 1:
            raise AssertionError("Cycle count must be >= 1 when metadata XML is available")
        if timestamps is not None and cycle_count is not None:
            assert len(timestamps) == cycle_count, "Timestamp count must match cycle count"
        assert image_size[0] > 0 and image_size[1] > 0, "Image size must be positive"

    t_total_s = None
    if cycle_count is not None and dt_s is not None:
        t_total_s = cycle_count * dt_s
    if t_total_s is None:
        t_total_s = 0.0
    if dt_s is None:
        dt_s = 0.0

    metadata = {
        "channels": channel_list,
        "exposure_ms": exposures_map,
        "t_total_s": t_total_s,
        "dt_s": dt_s,
        "timestamps_s": timestamps,
        "objective_name": objective_name,
        "objective_mag": objective_mag,
        "objective_na": objective_na,
        "immersion": immersion,
        "camera_model": camera_model,
        "px_sensor_um": px_sensor_um,
        "px_size_um": px_size_um,
        "image_size_px": list(image_size),
        "binning": list(binning),
        "tiles": tiles or None,
        "stage": stage or None,
    }

    return metadata


@dataclass
class ChannelInfo:
    name: str
    color: Optional[Dict[str, object]] = None
    exposure_ms: Optional[float] = None
    index: Optional[int] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "color": self.color,
            "exposure_ms": self.exposure_ms,
            "index": self.index,
        }
def extract_time_index(path: Path) -> int:
    name = path.name
    match = re.search(r"(?:^|[_-])t(?:ime)?0*([0-9]+)", name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0


def infer_channels_from_filenames(paths: Sequence[Path]) -> List[ChannelInfo]:
    channel_indices: Dict[str, ChannelInfo] = {}
    pattern_idx = re.compile(r"(?:^|[_-])c(?:h)?0*([0-9]+)", re.IGNORECASE)
    pattern_name = re.compile(r"(?:^|[_-])(ch)?([A-Za-z]{2,})(?:[_-]|\.|$)")
    for path in paths:
        name = path.stem
        idx_match = pattern_idx.search(name)
        idx = int(idx_match.group(1)) if idx_match else None
        candidate = None
        name_match = re.search(r"C=([A-Za-z0-9]+)", name)
        if name_match:
            candidate = name_match.group(1)
        else:
            chunks = re.split(r"[_-]", name)
            for chunk in chunks[::-1]:
                if chunk.lower().startswith("t") and chunk[1:].isdigit():
                    continue
                if chunk.lower().startswith("z") and chunk[1:].isdigit():
                    continue
                if chunk.lower().startswith("c") and chunk[1:].isdigit():
                    continue
                if chunk.lower() in {"max", "min"}:
                    continue
                if re.fullmatch(r"[A-Za-z0-9]+", chunk):
                    candidate = chunk
                    break
        if candidate:
            key = candidate.upper()
            channel_indices.setdefault(key, ChannelInfo(name=candidate, index=idx))
    return list(channel_indices.values())


def find_channel_files(run_root: Path, metadata: Dict[str, object]) -> Dict[str, List[Tuple[int, Path]]]:
    run_root = Path(run_root)
    tiff_paths = sorted(run_root.rglob("*.tif"))
    if not tiff_paths:
        raise FileNotFoundError(f"No TIFF files found in {run_root}")

    known_channels = {}
    for entry in metadata.get("channels", []):
        name = entry.get("name")
        if name:
            known_channels[name.upper()] = entry
        idx = entry.get("index")
        if idx is not None:
            known_channels[str(idx)] = entry

    channel_files: Dict[str, List[Tuple[int, Path]]] = {}

    for path in tiff_paths:
        time_index = extract_time_index(path)
        channel_key = None
        name_upper = None
        idx = None

        match_idx = re.search(r"(?:^|[_-])c(?:h)?0*([0-9]+)", path.stem, re.IGNORECASE)
        if match_idx:
            idx = int(match_idx.group(1))
            channel_entry = known_channels.get(str(idx))
            if channel_entry:
                channel_key = channel_entry["name"].upper()

        if channel_key is None:
            match_name = re.search(r"C=([A-Za-z0-9]+)", path.stem)
            if match_name:
                name_upper = match_name.group(1).upper()
            else:
                chunks = re.split(r"[_-]", path.stem)
                for chunk in reversed(chunks):
                    if chunk.lower().startswith("t") and chunk[1:].isdigit():
                        continue
                    if chunk.lower().startswith("z") and chunk[1:].isdigit():
                        continue
                    if chunk.lower().startswith("c") and chunk[1:].isdigit():
                        continue
                    if re.fullmatch(r"[A-Za-z0-9]+", chunk):
                        name_upper = chunk.upper()
                        break
            if name_upper:
                channel_entry = known_channels.get(name_upper)
                if channel_entry:
                    channel_key = channel_entry["name"].upper()
                else:
                    channel_key = name_upper

        if channel_key is None:
            # Fallback to TIFF metadata
            try:
                with tifffile.TiffFile(path) as tif:
                    desc = tif.pages[0].description or ""
            except Exception:  # pragma: no cover
                desc = ""
            match_desc = re.search(r"Channel[^\n]{0,20}?Name=\"([^\"]+)\"", desc)
            if match_desc:
                channel_key = match_desc.group(1).upper()
            elif known_channels:
                channel_key = next(iter(known_channels.keys()))
            else:
                channel_key = "CHANNEL"

        channel_files.setdefault(channel_key, []).append((time_index, path))

    for files in channel_files.values():
        files.sort(key=lambda pair: pair[0])

    return channel_files


def read_first_plane(path: Path) -> np.ndarray:
    with tifffile.TiffFile(path) as tif:
        arr = tif.asarray()
    if arr.ndim > 2:
        arr = arr[0]
    return arr.astype(np.float32)


def segment_droplet(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    if image.ndim != 2:
        raise ValueError("Droplet segmentation expects a 2D image.")

    smoothed = filters.gaussian(image, sigma=2.0)
    smoothed = exposure.rescale_intensity(smoothed)
    thresh = filters.threshold_otsu(smoothed)
    binary = smoothed > thresh

    binary = morphology.remove_small_objects(binary, 100)
    binary = morphology.remove_small_holes(binary, 200)
    binary = morphology.binary_closing(binary, morphology.disk(5))

    labeled = measure.label(binary)
    if labeled.max() == 0:
        raise RuntimeError("Droplet segmentation failed: no objects detected.")

    regions = measure.regionprops(labeled)
    largest = max(regions, key=lambda r: r.area)
    mask = labeled == largest.label
    mask = ndi.binary_fill_holes(mask)
    return mask


def fit_circle(mask: np.ndarray, image: np.ndarray) -> Tuple[Tuple[float, float], float, bool, List[str]]:
    image_scaled = exposure.rescale_intensity(image.astype(float))
    edges = canny(image_scaled, sigma=2.0)
    warnings: List[str] = []
    coords = np.column_stack(np.nonzero(edges))
    if coords.size == 0:
        warnings.append("No edges detected for circle fitting; using mask-derived circle.")
        y, x = ndi.center_of_mass(mask)
        area = mask.sum()
        radius = math.sqrt(area / math.pi)
    else:
        area_est = mask.sum()
        radius_est = max(5, int(round(math.sqrt(area_est / math.pi))))
        radii = np.arange(max(5, int(radius_est * 0.7)), int(radius_est * 1.3) + 1)
        if len(radii) == 0:
            radii = np.arange(5, 50)
        hough_res = hough_circle(edges, radii)
        accums, cx, cy, radii_found = hough_circle_peaks(hough_res, radii, total_num_peaks=1)
        if len(radii_found) == 0:
            warnings.append("Hough circle detection failed; using mask-derived circle.")
            y, x = ndi.center_of_mass(mask)
            area = mask.sum()
            radius = math.sqrt(area / math.pi)
        else:
            x = float(cx[0])
            y = float(cy[0])
            radius = float(radii_found[0])
    center = (x, y)
    height, width = mask.shape
    partial = not (
        radius <= x <= width - radius and radius <= y <= height - radius
    )
    if partial:
        warnings.append("Circle extends beyond field-of-view; volume estimate may be biased.")
    return center, radius, partial, warnings


def estimate_volumes(mask: np.ndarray, px_size_um: float, chamber_height_um: float, circle_radius_px: float) -> Tuple[float, float, float]:
    area_px = float(mask.sum())
    area_um2 = area_px * (px_size_um ** 2)
    volume_um3_slab = area_um2 * chamber_height_um
    volume_uL_slab = volume_um3_slab * 1e-9

    radius_um = circle_radius_px * px_size_um
    volume_um3_circle = (4.0 / 3.0) * math.pi * (radius_um ** 3)
    volume_uL_circle = volume_um3_circle * 1e-9

    return area_um2, volume_uL_slab, volume_uL_circle


def save_mask_preview(image: np.ndarray, mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap="gray")
    plt.contour(mask, colors="r", linewidths=1)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_circle_overlay(image: np.ndarray, center: Tuple[float, float], radius: float, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap="gray")
    circle = plt.Circle((center[0], center[1]), radius, color="cyan", fill=False, linewidth=2)
    ax.add_patch(circle)
    ax.scatter([center[0]], [center[1]], s=30, c="yellow")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def quantify_gfp(
    gfp_files: List[Tuple[int, Path]],
    mask: np.ndarray,
    volumes: Dict[str, float],
    process_all: bool,
    dt_s: float,
) -> Tuple[Dict[str, object], Optional[pd.DataFrame]]:
    results = []
    mask_bool = mask.astype(bool)
    mask_shape = mask_bool.shape

    for time_index, path in gfp_files:
        image = read_first_plane(path)
        if image.shape != mask_shape:
            LOGGER.warning(
                "Mask shape %s does not match image shape %s; resizing mask.",
                mask_shape,
                image.shape,
            )
            resized_mask = transform.resize(
                mask_bool.astype(float),
                image.shape,
                order=0,
                preserve_range=True,
                anti_aliasing=False,
            )
            mask_use = resized_mask >= 0.5
        else:
            mask_use = mask_bool
        intensity_sum = float(image[mask_use].sum())
        volume_slab = volumes.get("volume_uL_slab")
        volume_circle = volumes.get("volume_uL_circle")
        intensity_per_uL_slab = intensity_sum / volume_slab if volume_slab else float("nan")
        intensity_per_uL_circle = (
            intensity_sum / volume_circle if volume_circle else float("nan")
        )
        results.append(
            {
                "time_index": time_index,
                "t_s": time_index * dt_s,
                "sum_intensity": intensity_sum,
                "intensity_per_uL_slab": intensity_per_uL_slab,
                "intensity_per_uL_circle": intensity_per_uL_circle,
            }
        )
        if not process_all:
            break

    if not results:
        raise RuntimeError("No GFP images found for quantification.")

    summary = results[0]
    summary_data = {
        "sum_intensity": summary["sum_intensity"],
        "intensity_per_uL_slab": summary["intensity_per_uL_slab"],
        "intensity_per_uL_circle": summary["intensity_per_uL_circle"],
    }

    df = pd.DataFrame(results) if process_all else None
    return summary_data, df


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Leica droplet analysis pipeline")
    parser.add_argument("--run-root", required=True, help="Path to Leica run folder")
    parser.add_argument("--cy5", default="CY5", help="CY5 channel name or index")
    parser.add_argument("--gfp", default="GFP", help="GFP channel name or index")
    parser.add_argument(
        "--chamber-height-um",
        type=float,
        default=120.0,
        help="Chamber height in micrometers (default: 120)",
    )
    parser.add_argument(
        "--all-frames",
        action="store_true",
        help="Process all time frames for GFP quantification",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output folder (default: <RUN_ROOT>/analysis)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--dump-meta",
        action="store_true",
        help="Only parse metadata and write meta_summary.json",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    run_root = Path(args.run_root).resolve()
    if not run_root.exists():
        raise FileNotFoundError(f"Run root not found: {run_root}")

    output_dir = Path(args.out) if args.out else run_root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_dir = None
    for candidate in [run_root / "metaData", run_root / "metadata", run_root / "MetaData"]:
        if candidate.exists() and candidate.is_dir():
            metadata_dir = candidate
            break
    if metadata_dir is None:
        metadata_dir = run_root

    tiff_paths = sorted(run_root.rglob("*.tif"))
    if not tiff_paths:
        raise FileNotFoundError(f"No TIFF files found in {run_root}")

    LOGGER.info("Parsing metadata from %s", run_root)
    metadata = parse_metadata(metadata_dir, tiff_paths[0])
    meta_path = output_dir / "meta_summary.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if args.dump_meta:
        LOGGER.info("Metadata dump complete. Output saved to %s", meta_path)
        return

    LOGGER.info("Finding channel files")
    channel_files = find_channel_files(run_root, metadata)

    cy5_key = args.cy5.upper()
    gfp_key = args.gfp.upper()

    cy5_candidates = None
    for key, files in channel_files.items():
        if key == cy5_key or key.endswith(cy5_key) or cy5_key.endswith(key):
            cy5_candidates = files
            break
    if cy5_candidates is None:
        raise RuntimeError(f"Could not find CY5 channel files matching '{args.cy5}'")

    cy5_t0_path = min(cy5_candidates, key=lambda pair: pair[0])[1]
    LOGGER.info("Segmenting droplet from %s", cy5_t0_path)
    cy5_image = read_first_plane(cy5_t0_path)
    droplet_mask = segment_droplet(cy5_image)

    mask_path = output_dir / "droplet_mask_t0.png"
    save_mask_preview(cy5_image, droplet_mask, mask_path)

    center, radius_px, partial, warnings = fit_circle(droplet_mask, cy5_image)
    circle_overlay_path = output_dir / "droplet_circle_fit_overlay.png"
    save_circle_overlay(cy5_image, center, radius_px, circle_overlay_path)

    area_um2, volume_uL_slab, volume_uL_circle = estimate_volumes(
        droplet_mask,
        metadata["px_size_um"],
        args.chamber_height_um,
        radius_px,
    )

    droplet_summary = {
        "area_um2": area_um2,
        "volume_uL_slab": volume_uL_slab,
        "volume_uL_circle": volume_uL_circle,
        "mask_path": str(mask_path),
        "circle_center_px": {
            "x": float(center[0]),
            "y": float(center[1]),
        },
        "circle_radius_px": float(radius_px),
        "circle_partial": partial,
        "warnings": warnings,
    }
    if partial:
        droplet_summary.setdefault("warnings", []).append("Circle fit indicates partial droplet in FOV.")

    droplet_summary_path = output_dir / "droplet_summary.json"
    droplet_summary_path.write_text(json.dumps(droplet_summary, indent=2), encoding="utf-8")

    LOGGER.info("Quantifying GFP channel")
    gfp_candidates = None
    for key, files in channel_files.items():
        if key == gfp_key or key.endswith(gfp_key) or gfp_key.endswith(key):
            gfp_candidates = files
            break
    if gfp_candidates is None:
        raise RuntimeError(f"Could not find GFP channel files matching '{args.gfp}'")

    summary_data, df = quantify_gfp(
        gfp_candidates,
        droplet_mask,
        {
            "area_um2": area_um2,
            "volume_uL_slab": volume_uL_slab,
            "volume_uL_circle": volume_uL_circle,
        },
        args.all_frames,
        metadata["dt_s"],
    )

    if df is not None:
        csv_path = output_dir / "gfp_quant_timeseries.csv"
        df.to_csv(csv_path, index=False)
    else:
        quant_path = output_dir / "gfp_quant_t0.json"
        quant_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")

    summary_lines = [
        "Leica Droplet Analysis Summary",
        "===============================",
        f"Run root: {run_root}",
        "",
        "Metadata:",
        json.dumps(metadata, indent=2),
        "",
        "Droplet volumes:",
        json.dumps(droplet_summary, indent=2),
        "",
        "GFP quantification:",
        json.dumps(summary_data, indent=2),
    ]
    (output_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    LOGGER.info("Analysis completed. Outputs saved to %s", output_dir)


if __name__ == "__main__":  # pragma: no cover
    main()
