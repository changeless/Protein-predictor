"""Utility to extract microscope metadata from Leica XML files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List
import xml.etree.ElementTree as ET

PathLike = Path | str


def flatten_xml_element(element: ET.Element, path: str = "", result: Dict[str, str] | None = None) -> Dict[str, str]:
    """Recursively flattens an XML element into a mapping of key paths to values.

    Parameters
    ----------
    element:
        The XML element to flatten.
    path:
        The hierarchical path of the current element. The root element should be passed
        with an empty string.
    result:
        Dictionary that will be populated with flattened values. If ``None`` a new
        dictionary is created.

    Returns
    -------
    Dict[str, str]
        A mapping where keys represent XML paths and values are the corresponding
        attribute or text values.
    """
    if result is None:
        result = {}

    if path:
        base_path = path
    else:
        base_path = element.tag

    # Capture element text when present and non-empty
    text = (element.text or "").strip()
    if text:
        result[base_path] = text

    # Record all attributes for this element
    for attr_name, attr_value in element.attrib.items():
        result[f"{base_path}/@{attr_name}"] = attr_value

    # Group children by tag to assign stable indices
    children_by_tag: Dict[str, List[ET.Element]] = {}
    for child in list(element):
        children_by_tag.setdefault(child.tag, []).append(child)

    for tag, children in children_by_tag.items():
        for index, child in enumerate(children):
            child_path = f"{base_path}/{tag}[{index}]"
            flatten_xml_element(child, child_path, result)

    return result


def parse_xml_file(path: PathLike) -> Dict[str, str]:
    """Parse an XML file and return a flattened representation."""
    tree = ET.parse(path)
    root = tree.getroot()
    return flatten_xml_element(root, root.tag)


def parse_metadata_directory(directory: PathLike) -> List[Dict[str, str]]:
    """Parse all XML metadata files in a directory.

    Parameters
    ----------
    directory:
        Path to the directory containing XML metadata files.

    Returns
    -------
    List[Dict[str, str]]
        A list of flattened metadata dictionaries, one per XML file. Each dictionary
        contains an additional key ``__file__`` that stores the relative file path.
    """
    directory_path = Path(directory)
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    results: List[Dict[str, str]] = []
    for xml_path in sorted(directory_path.rglob("*.xml")):
        metadata = parse_xml_file(xml_path)
        metadata["__file__"] = str(xml_path.relative_to(directory_path))
        results.append(metadata)

    return results


def write_json(records: Iterable[Dict[str, str]], output_path: PathLike) -> None:
    """Write metadata records to a JSON file."""
    with Path(output_path).open("w", encoding="utf-8") as fh:
        json.dump(list(records), fh, indent=2, ensure_ascii=False)


def write_csv(records: Iterable[Dict[str, str]], output_path: PathLike) -> None:
    """Write metadata records to a CSV file.

    The CSV header includes the union of keys across all records. Missing values are
    written as empty strings.
    """
    from csv import DictWriter

    record_list = list(records)
    if not record_list:
        Path(output_path).write_text("", encoding="utf-8")
        return

    # Build a sorted list of columns for deterministic output
    columns = sorted({key for record in record_list for key in record.keys()})

    with Path(output_path).open("w", encoding="utf-8", newline="") as fh:
        writer = DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for record in record_list:
            writer.writerow({key: record.get(key, "") for key in columns})


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract microscope metadata from Leica XML files and serialise the results.",
    )
    parser.add_argument(
        "directory",
        help="Directory containing metadata files (XML format).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="metadata_summary.json",
        help="Path to the output file (defaults to metadata_summary.json).",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices={"json", "csv"},
        default="json",
        help="Output format. Choose between json and csv (default: json).",
    )
    return parser


def main(args: List[str] | None = None) -> None:
    parser = build_argument_parser()
    parsed_args = parser.parse_args(args)

    records = parse_metadata_directory(parsed_args.directory)

    output_path = Path(parsed_args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if parsed_args.format == "json":
        write_json(records, output_path)
    else:
        write_csv(records, output_path)


if __name__ == "__main__":
    main()
