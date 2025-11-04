"""Quantify GFP expression levels from Leica TIFF images.

This script aggregates fluorescence measurements from multi-channel
TIFF images (for example, droplets acquired on a Leica system) and
reports the total GFP intensity as well as intensity per microlitre.
When sufficient metadata is available it also estimates the droplet
footprint and volume so that protein concentrations can be derived.
If the droplet is only partially captured in the field of view the
tool fits a circle through the detected boundary to extrapolate the
missing portion before reporting the inferred size and volume.

Usage examples
--------------

To analyse all TIFF files within a directory using the default channel
name "GFP" and assuming a droplet volume of 1 µL::

    python quantify_gfp.py path/to/images/*.tif --volume-ul 1

If the GFP channel uses a different label in the image metadata you can
override it::

    python quantify_gfp.py sample.tif --channel-name "488-Gain"

When channel annotations are missing entirely, provide an explicit
channel index (0-based)::

    python quantify_gfp.py sample.tif --channel-index 1

If the TIFF does not expose channel names, the script attempts to look up
the acquisition order from a Leica ``MetaData`` directory located next to
the images (or provided manually via ``--metadata-dir``) so that the GFP
channel can be detected automatically.

The script prints a table to stdout and can optionally export the
results as JSON or CSV files via the ``--output`` flag.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional


def _ensure_dependency(module_name: str, package_hint: Optional[str] = None) -> None:
    """Abort execution with a helpful message when a dependency is missing."""

    if importlib.util.find_spec(module_name) is None:
        package = package_hint or module_name
        raise SystemExit(
            "Missing dependency: '{module}'. Install it with 'pip install {package}' and retry."
            .format(module=module_name, package=package)
        )


_ensure_dependency("numpy", "numpy")
_ensure_dependency("tifffile", "tifffile")

import numpy as np  # type: ignore  # noqa: E402
import tifffile  # type: ignore  # noqa: E402
from xml.etree import ElementTree as ET

parse_metadata_directory = None
if importlib.util.find_spec("parse_metadata") is not None:  # pragma: no cover - optional helper
    from parse_metadata import parse_metadata_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        help="TIFF image files or directories containing TIFF files.",
    )
    parser.add_argument(
        "--channel-name",
        default="GFP",
        help=(
            "Channel name to extract based on the image metadata. "
            "Ignored when --channel-index is supplied."
        ),
    )
    parser.add_argument(
        "--channel-index",
        type=int,
        help="0-based channel index to extract when metadata is unavailable.",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        help=(
            "Optional Leica MetaData directory. When omitted the script searches "
            "for a sibling 'MetaData' folder next to each image."
        ),
    )
    parser.add_argument(
        "--volume-ul",
        type=float,
        default=1.0,
        help=(
            "Sample volume in microlitres used to normalise the intensity when a droplet "
            "estimate cannot be derived."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output file (JSON or CSV) for the aggregated metrics.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=4,
        help="Number of decimal places when printing floating-point metrics.",
    )
    return parser.parse_args()


def iter_tiff_files(paths: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            files.extend(sorted(path.glob("*.tif")))
            files.extend(sorted(path.glob("*.tiff")))
        elif path.suffix.lower() in {".tif", ".tiff"}:
            files.append(path)
        else:
            raise ValueError(f"Unsupported input path: {path}")
    unique_files = []
    seen = set()
    for file in files:
        if file not in seen:
            unique_files.append(file)
            seen.add(file)
    if not unique_files:
        raise FileNotFoundError("No TIFF files were found for the supplied paths.")
    return unique_files


def extract_channel_names(ome_xml: Optional[str]) -> List[str]:
    if not ome_xml:
        return []
    try:
        root = ET.fromstring(ome_xml)
    except ET.ParseError:
        return []
    ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
    names: List[str] = []
    for channel in root.findall(".//ome:Channel", ns):
        name = channel.get("Name") or channel.get("ID")
        if name:
            names.append(name)
    return names


def discover_metadata_directory(image_path: Path, override: Optional[Path]) -> Optional[Path]:
    if override:
        if not override.is_dir():
            raise FileNotFoundError(f"Specified metadata directory not found: {override}")
        return override.resolve()

    for parent in [image_path.parent, *image_path.parents]:
        candidate = parent / "MetaData"
        if candidate.is_dir():
            return candidate.resolve()
    return None


@lru_cache(maxsize=None)
def load_leica_channel_names(metadata_dir: Path) -> List[str]:
    channel_names: List[str] = []
    seen = set()
    xml_files = sorted(metadata_dir.rglob("*.xml"))
    for xml_path in xml_files:
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            continue
        for element in root.iter():
            tag = element.tag.split("}")[-1]
            if tag != "WideFieldChannelInfo":
                continue
            name = (
                element.attrib.get("UserDefName")
                or element.attrib.get("FluoCubeName")
                or element.attrib.get("ContrastingMethodName")
                or element.attrib.get("Channel")
            )
            if not name or name in seen:
                continue
            seen.add(name)
            channel_names.append(name)
    return channel_names


@lru_cache(maxsize=None)
def load_metadata_records(metadata_dir: Path) -> List[dict]:
    if parse_metadata_directory is None:
        return []
    try:
        return parse_metadata_directory(metadata_dir)
    except FileNotFoundError:
        return []


def parse_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_pixel_sizes_from_metadata(metadata_dir: Path) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Infer pixel dimensions (in µm) from Leica metadata files when possible."""

    records = load_metadata_records(metadata_dir)
    x_values: List[float] = []
    y_values: List[float] = []
    z_values: List[float] = []

    keywords = {
        "x": ("scalex", "scalingx", "pixelsizex", "pixelwidth", "resolutionx", "xcalibration"),
        "y": ("scaley", "scalingy", "pixelsizey", "pixelheight", "resolutiony", "ycalibration"),
        "z": ("scalez", "scalingz", "pixelsizez", "resolutionz", "zcalibration"),
    }

    for record in records:
        for key, value in record.items():
            if not isinstance(value, str):
                continue
            lower_key = key.lower()
            number = parse_float(value)
            if number is None:
                continue
            if any(term in lower_key for term in keywords["x"]):
                x_values.append(number)
            if any(term in lower_key for term in keywords["y"]):
                y_values.append(number)
            if any(term in lower_key for term in keywords["z"]):
                z_values.append(number)

    def pick(values: List[float]) -> Optional[float]:
        if not values:
            return None
        return float(np.median(values))

    return pick(x_values), pick(y_values), pick(z_values)


def extract_pixel_sizes_from_ome(ome_xml: Optional[str]) -> tuple[Optional[float], Optional[float], Optional[float]]:
    if not ome_xml:
        return None, None, None
    try:
        root = ET.fromstring(ome_xml)
    except ET.ParseError:
        return None, None, None
    ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
    pixels = root.find(".//ome:Pixels", ns)
    if pixels is None:
        return None, None, None
    px = pixels.get("PhysicalSizeX")
    py = pixels.get("PhysicalSizeY")
    pz = pixels.get("PhysicalSizeZ")
    return (
        parse_float(px) if px is not None else None,
        parse_float(py) if py is not None else None,
        parse_float(pz) if pz is not None else None,
    )


def determine_pixel_sizes(
    series: tifffile.TiffPageSeries,
    ome_xml: Optional[str],
    metadata_dir: Optional[Path],
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    px, py, pz = extract_pixel_sizes_from_ome(ome_xml)

    if (px is None or py is None or ("Z" in series.axes and pz is None)) and metadata_dir:
        md_px, md_py, md_pz = extract_pixel_sizes_from_metadata(metadata_dir)
        px = px or md_px
        py = py or md_py
        if "Z" in series.axes:
            pz = pz or md_pz

    # Attempt to use TIFF resolution tags when everything else fails.
    if px is None or py is None:
        first_page = series.pages[0]
        x_res_tag = first_page.tags.get("XResolution")
        y_res_tag = first_page.tags.get("YResolution")
        unit_tag = first_page.tags.get("ResolutionUnit")
        if x_res_tag and y_res_tag and unit_tag:
            unit = unit_tag.value
            # Resolution unit 3 corresponds to centimetre in TIFF spec.
            cm_per_unit = 1.0 if unit == 3 else 2.54 if unit == 2 else None
            if cm_per_unit:
                x_res = x_res_tag.value[0] / x_res_tag.value[1]
                y_res = y_res_tag.value[0] / y_res_tag.value[1]
                if x_res:
                    px = px or (cm_per_unit / x_res) * 1e4  # convert cm to µm
                if y_res:
                    py = py or (cm_per_unit / y_res) * 1e4

    return px, py, pz


def project_channel_image(data: np.ndarray, axes: str, channel_idx: int) -> np.ndarray:
    channel_axis = axes.index("C")
    if channel_axis != 0:
        data = np.moveaxis(data, channel_axis, 0)
        axes = "C" + axes[:channel_axis] + axes[channel_axis + 1 :]
    channel_data = data[channel_idx]
    axis_order = list(axes[1:])  # axes after removing channel
    projection = channel_data
    keep_axes = {"Y", "X"}
    i = 0
    while i < len(axis_order):
        axis_name = axis_order[i]
        if axis_name in keep_axes:
            i += 1
            continue
        projection = projection.sum(axis=i)
        axis_order.pop(i)
    if set(axis_order) != {"Y", "X"}:
        raise ValueError("Unable to project image to 2D: missing Y/X axes")
    if axis_order == ["X", "Y"]:
        projection = np.swapaxes(projection, 0, 1)
    return projection.astype(np.float64, copy=False)


def otsu_threshold(image: np.ndarray) -> float:
    flattened = image[np.isfinite(image)].ravel()
    if flattened.size == 0:
        return float(np.nan)
    min_val = float(np.min(flattened))
    max_val = float(np.max(flattened))
    if math.isclose(min_val, max_val):
        return min_val
    hist, bin_edges = np.histogram(flattened, bins=256, range=(min_val, max_val))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return float(np.nan)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    sum_total = float(np.dot(hist, bin_centres))
    sum_background = 0.0
    weight_background = 0.0
    max_variance = -1.0
    threshold = bin_centres[0]
    for idx, count in enumerate(hist):
        weight_background += count
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break
        sum_background += count * bin_centres[idx]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        if between > max_variance:
            max_variance = between
            threshold = bin_centres[idx]
    return threshold


def _mask_touches_border(mask: np.ndarray) -> bool:
    return bool(
        mask[0, :].any()
        or mask[-1, :].any()
        or mask[:, 0].any()
        or mask[:, -1].any()
    )


def _extract_boundary_points(mask: np.ndarray) -> np.ndarray:
    boundary = mask.copy()
    interior = mask.copy()
    interior[1:-1, 1:-1] &= mask[:-2, 1:-1]
    interior[1:-1, 1:-1] &= mask[2:, 1:-1]
    interior[1:-1, 1:-1] &= mask[1:-1, :-2]
    interior[1:-1, 1:-1] &= mask[1:-1, 2:]
    boundary[1:-1, 1:-1] &= ~interior[1:-1, 1:-1]
    return np.column_stack(np.nonzero(boundary))


def _fit_circle(
    boundary_points: np.ndarray,
    pixel_size_x: float,
    pixel_size_y: float,
) -> Optional[dict]:
    if boundary_points.shape[0] < 3:
        return None

    y = boundary_points[:, 0].astype(np.float64) * pixel_size_y
    x = boundary_points[:, 1].astype(np.float64) * pixel_size_x

    if boundary_points.shape[0] > 8000:
        step = boundary_points.shape[0] // 8000 + 1
        y = y[::step]
        x = x[::step]

    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x ** 2 + y ** 2)
    try:
        solution, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None

    D, E, F = solution
    center_x = -D / 2.0
    center_y = -E / 2.0
    radius_sq = center_x ** 2 + center_y ** 2 - F
    if radius_sq <= 0:
        return None
    radius = math.sqrt(radius_sq)

    residual = np.sqrt(((x - center_x) ** 2 + (y - center_y) ** 2)) - radius
    rms_residual = float(np.sqrt(np.mean(residual ** 2))) if residual.size else float("nan")

    return {
        "radius_um": float(radius),
        "center_x_um": float(center_x),
        "center_y_um": float(center_y),
        "rms_residual_um": rms_residual,
    }


def estimate_droplet_geometry(
    image_2d: np.ndarray,
    pixel_size_x: Optional[float],
    pixel_size_y: Optional[float],
    pixel_size_z: Optional[float],
    z_planes: int,
) -> dict:
    result = {
        "pixel_size_x_um": float("nan"),
        "pixel_size_y_um": float("nan"),
        "pixel_size_z_um": float("nan"),
        "area_pixels": float("nan"),
        "area_um2": float("nan"),
        "equivalent_diameter_um": float("nan"),
        "volume_ul_sphere": float("nan"),
        "volume_ul_cylinder": float("nan"),
        "mask_touches_border": False,
        "circle_fit_radius_um": float("nan"),
        "circle_fit_center_x_um": float("nan"),
        "circle_fit_center_y_um": float("nan"),
        "circle_fit_rms_residual_um": float("nan"),
        "circle_fit_area_um2": float("nan"),
        "circle_fit_volume_ul": float("nan"),
        "circle_fit_volume_ul_cylinder": float("nan"),
    }

    if pixel_size_x is not None:
        result["pixel_size_x_um"] = float(pixel_size_x)
    if pixel_size_y is not None:
        result["pixel_size_y_um"] = float(pixel_size_y)
    if pixel_size_z is not None:
        result["pixel_size_z_um"] = float(pixel_size_z)

    if pixel_size_x is None or pixel_size_y is None:
        return result

    threshold = otsu_threshold(image_2d)
    if not np.isfinite(threshold):
        return result
    mask = image_2d >= threshold
    area_pixels = float(mask.sum())
    if area_pixels == 0:
        return result

    touches_border = _mask_touches_border(mask)
    result["mask_touches_border"] = touches_border

    area_um2 = area_pixels * pixel_size_x * pixel_size_y
    equivalent_radius_um = math.sqrt(area_um2 / math.pi)
    volume_sphere_um3 = (4.0 / 3.0) * math.pi * (equivalent_radius_um ** 3)
    volume_ul_sphere = volume_sphere_um3 / 1e9

    volume_ul_cylinder = float("nan")
    if pixel_size_z is not None and z_planes > 0:
        thickness_um = pixel_size_z * z_planes
        volume_cyl_um3 = area_um2 * thickness_um
        volume_ul_cylinder = volume_cyl_um3 / 1e9

    if touches_border:
        boundary_points = _extract_boundary_points(mask)
        circle_fit = _fit_circle(boundary_points, pixel_size_x, pixel_size_y)
        if circle_fit is not None:
            result["circle_fit_radius_um"] = circle_fit["radius_um"]
            result["circle_fit_center_x_um"] = circle_fit["center_x_um"]
            result["circle_fit_center_y_um"] = circle_fit["center_y_um"]
            result["circle_fit_rms_residual_um"] = circle_fit["rms_residual_um"]
            area_circle_um2 = math.pi * (circle_fit["radius_um"] ** 2)
            result["circle_fit_area_um2"] = area_circle_um2
            volume_circle_um3 = (4.0 / 3.0) * math.pi * (circle_fit["radius_um"] ** 3)
            result["circle_fit_volume_ul"] = volume_circle_um3 / 1e9
            if pixel_size_z is not None and z_planes > 0:
                thickness_um = pixel_size_z * z_planes
                cyl_volume_um3 = area_circle_um2 * thickness_um
                result["circle_fit_volume_ul_cylinder"] = cyl_volume_um3 / 1e9

    result.update(
        {
            "area_pixels": area_pixels,
            "area_um2": area_um2,
            "equivalent_diameter_um": 2.0 * equivalent_radius_um,
            "volume_ul_sphere": volume_ul_sphere,
            "volume_ul_cylinder": volume_ul_cylinder,
        }
    )
    return result


def locate_channel_index(
    series: tifffile.TiffPageSeries,
    channel_name: str,
    channel_index: Optional[int],
    label_sets: List[tuple[str, List[str]]],
) -> tuple[int, Optional[str], str]:
    if channel_index is not None:
        if channel_index < 0 or channel_index >= series.shape[series.axes.index("C")]:
            raise IndexError(
                f"Channel index {channel_index} out of bounds for axes {series.axes}."
            )
        resolved_label = None
        resolved_source = "manual"
        for source, labels in label_sets:
            if channel_index < len(labels) and labels[channel_index]:
                resolved_label = labels[channel_index]
                resolved_source = source
                break
        return channel_index, resolved_label, resolved_source

    if "C" not in series.axes:
        raise ValueError(
            "The image does not expose a channel axis. Provide --channel-index."
        )

    channel_axis = series.axes.index("C")
    channel_count = series.shape[channel_axis]

    for source, labels in label_sets:
        for idx, name in enumerate(labels):
            if idx >= channel_count:
                break
            if name and channel_name.lower() in name.lower():
                return idx, name, source

    # Fallback: try to infer from series.axes order.
    if channel_count == 1:
        resolved_label = None
        resolved_source = "implicit"
        for source, labels in label_sets:
            if labels:
                resolved_label = labels[0]
                resolved_source = source
                break
        return 0, resolved_label, resolved_source

    raise ValueError(
        "Unable to determine the GFP channel. Use --channel-index to specify it explicitly."
    )


def extract_channel_data(
    data: np.ndarray, axes: str, channel_idx: int
) -> np.ndarray:
    channel_axis = axes.index("C")
    if channel_axis != 0:
        data = np.moveaxis(data, channel_axis, 0)
    channel_data = data[channel_idx]
    # Collapse any remaining dimensions (e.g., Z, T) by summing.
    if channel_data.ndim > 2:
        collapse_axes = tuple(range(channel_data.ndim - 2))
        channel_data = channel_data.sum(axis=collapse_axes)
    return np.asarray(channel_data, dtype=np.float64).ravel()


def summarise_gfp_intensity(
    image_path: Path,
    channel_name: str,
    channel_index: Optional[int],
    volume_ul: float,
    metadata_dir: Optional[Path],
) -> dict:
    with tifffile.TiffFile(image_path) as tif:
        series = tif.series[0]
        data = series.asarray()
        if "C" not in series.axes:
            raise ValueError(
                f"Image {image_path} is missing a channel axis; specify --channel-index."
            )
        channel_names = extract_channel_names(tif.ome_metadata)
    metadata_channels: List[str] = []
    metadata_root = discover_metadata_directory(image_path, metadata_dir)
    if metadata_root is not None:
        metadata_channels = load_leica_channel_names(metadata_root)
    label_sets: List[tuple[str, List[str]]] = []
    if channel_names:
        label_sets.append(("ome", channel_names))
    if metadata_channels:
        label_sets.append(("metadata", metadata_channels))
    idx, resolved_label, resolved_source = locate_channel_index(
        series, channel_name, channel_index, label_sets
    )
    channel_data = extract_channel_data(data, series.axes, idx)
    pixel_size_x, pixel_size_y, pixel_size_z = determine_pixel_sizes(
        series, tif.ome_metadata, metadata_root
    )
    z_planes = (
        series.shape[series.axes.index("Z")] if "Z" in series.axes else (1 if pixel_size_z else 0)
    )
    try:
        channel_projection = project_channel_image(data, series.axes, idx)
    except ValueError:
        channel_projection = None
    geometry = (
        estimate_droplet_geometry(
            channel_projection,
            pixel_size_x,
            pixel_size_y,
            pixel_size_z,
            z_planes,
        )
        if channel_projection is not None
        else {
            "pixel_size_x_um": float("nan"),
            "pixel_size_y_um": float("nan"),
            "pixel_size_z_um": float("nan"),
            "area_pixels": float("nan"),
            "area_um2": float("nan"),
            "equivalent_diameter_um": float("nan"),
            "volume_ul_sphere": float("nan"),
            "volume_ul_cylinder": float("nan"),
        }
    )
    total_intensity = float(np.sum(channel_data))
    mean_intensity = float(np.mean(channel_data))
    max_intensity = float(np.max(channel_data))
    min_intensity = float(np.min(channel_data))
    reported_volume = volume_ul if volume_ul else float("nan")
    intensity_per_ul = total_intensity / reported_volume if np.isfinite(reported_volume) else float("nan")
    est_volume_ul = float("nan")
    for candidate in (
        geometry["circle_fit_volume_ul"],
        geometry["circle_fit_volume_ul_cylinder"],
        geometry["volume_ul_sphere"],
        geometry["volume_ul_cylinder"],
    ):
        if np.isfinite(candidate) and candidate > 0:
            est_volume_ul = candidate
            break
    intensity_per_estimated_ul = (
        total_intensity / est_volume_ul if np.isfinite(est_volume_ul) and est_volume_ul > 0 else float("nan")
    )
    return {
        "file": image_path.name,
        "channel_index": idx,
        "channel_name": resolved_label,
        "channel_source": resolved_source,
        "total_intensity": total_intensity,
        "mean_intensity": mean_intensity,
        "max_intensity": max_intensity,
        "min_intensity": min_intensity,
        "volume_ul": volume_ul,
        "intensity_per_ul": intensity_per_ul,
        "pixel_size_x_um": geometry["pixel_size_x_um"],
        "pixel_size_y_um": geometry["pixel_size_y_um"],
        "pixel_size_z_um": geometry["pixel_size_z_um"],
        "droplet_area_pixels": geometry["area_pixels"],
        "droplet_area_um2": geometry["area_um2"],
        "droplet_equivalent_diameter_um": geometry["equivalent_diameter_um"],
        "droplet_volume_ul_sphere": geometry["volume_ul_sphere"],
        "droplet_volume_ul_cylinder": geometry["volume_ul_cylinder"],
        "droplet_mask_touches_border": geometry["mask_touches_border"],
        "droplet_volume_ul_circlefit": geometry["circle_fit_volume_ul"],
        "droplet_volume_ul_circlefit_cylinder": geometry["circle_fit_volume_ul_cylinder"],
        "droplet_circle_radius_um": geometry["circle_fit_radius_um"],
        "droplet_circle_center_x_um": geometry["circle_fit_center_x_um"],
        "droplet_circle_center_y_um": geometry["circle_fit_center_y_um"],
        "droplet_circle_rms_residual_um": geometry["circle_fit_rms_residual_um"],
        "droplet_area_um2_circlefit": geometry["circle_fit_area_um2"],
        "intensity_per_estimated_ul": intensity_per_estimated_ul,
    }


def format_float(value: float, precision: int) -> str:
    return f"{value:.{precision}f}" if np.isfinite(value) else "nan"


def print_summary(rows: List[dict], precision: int) -> None:
    headers = [
        "file",
        "channel",
        "channel_source",
        "total_intensity",
        "mean_intensity",
        "max_intensity",
        "min_intensity",
        "volume_ul",
        "intensity_per_ul",
        "droplet_volume_ul_sphere",
        "droplet_volume_ul_cylinder",
        "droplet_volume_ul_circlefit",
        "droplet_volume_ul_circlefit_cylinder",
        "droplet_circle_radius_um",
        "mask_touches_border",
        "intensity_per_estimated_ul",
    ]
    if not rows:
        print("No results were generated. Check the error messages above for details.")
        return

    print("\t".join(headers))
    for row in rows:
        channel_label = (
            row["channel_name"]
            if row.get("channel_name")
            else f"index {row['channel_index']}"
        )
        values = [
            row["file"],
            channel_label,
            row.get("channel_source", ""),
            format_float(row["total_intensity"], precision),
            format_float(row["mean_intensity"], precision),
            format_float(row["max_intensity"], precision),
            format_float(row["min_intensity"], precision),
            format_float(row["volume_ul"], precision),
            format_float(row["intensity_per_ul"], precision),
            format_float(row["droplet_volume_ul_sphere"], precision),
            format_float(row["droplet_volume_ul_cylinder"], precision),
            format_float(row["droplet_volume_ul_circlefit"], precision),
            format_float(row["droplet_volume_ul_circlefit_cylinder"], precision),
            format_float(row["droplet_circle_radius_um"], precision),
            str(row.get("droplet_mask_touches_border", "")),
            format_float(row["intensity_per_estimated_ul"], precision),
        ]
        print("\t".join(values))


def export_results(rows: List[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".json":
        output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    elif output_path.suffix.lower() == ".csv":
        headers = list(rows[0].keys()) if rows else []
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
    else:
        raise ValueError("Output file must end with .json or .csv")


def main() -> None:
    args = parse_args()
    tiff_files = iter_tiff_files(args.paths)
    summaries: List[dict] = []
    errors: List[tuple[Path, Exception]] = []
    for file in tiff_files:
        try:
            summary = summarise_gfp_intensity(
                file,
                channel_name=args.channel_name,
                channel_index=args.channel_index,
                volume_ul=args.volume_ul,
                metadata_dir=args.metadata_dir,
            )
        except Exception as exc:  # pragma: no cover - CLI feedback
            errors.append((file, exc))
            continue
        summaries.append(summary)

    if errors:
        for file, exc in errors:
            print(f"[error] {file}: {exc}", file=sys.stderr)

    print_summary(summaries, precision=args.precision)

    if args.output and summaries:
        export_results(summaries, args.output)

    if summaries:
        return

    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()

