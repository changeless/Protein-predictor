"""Quantify GFP expression levels from Leica TIFF images.

This script aggregates fluorescence measurements from multi-channel
TIFF images (for example, droplets acquired on a Leica system) and
reports the total GFP intensity as well as intensity per microlitre.

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
import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import tifffile
from xml.etree import ElementTree as ET


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
        help="Sample volume in microlitres used to normalise the intensity.",
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
    total_intensity = float(np.sum(channel_data))
    mean_intensity = float(np.mean(channel_data))
    max_intensity = float(np.max(channel_data))
    min_intensity = float(np.min(channel_data))
    intensity_per_ul = total_intensity / volume_ul if volume_ul else float("nan")
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
    ]
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
    summaries = []
    for file in tiff_files:
        summary = summarise_gfp_intensity(
            file,
            channel_name=args.channel_name,
            channel_index=args.channel_index,
            volume_ul=args.volume_ul,
            metadata_dir=args.metadata_dir,
        )
        summaries.append(summary)

    print_summary(summaries, precision=args.precision)

    if args.output:
        export_results(summaries, args.output)


if __name__ == "__main__":
    main()

