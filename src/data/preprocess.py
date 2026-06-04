"""Data preprocessing: convert raw Solomon/Homberger files to processed JSON and generate manifests."""

from __future__ import annotations

import json
from pathlib import Path

from src.data.solomon_loader import load_solomon_instance
from src.data.homberger_loader import load_homberger_instance
from src.domain.schema import VRPTWInstance


def preprocess_solomon(
    raw_dir: str | Path,
    processed_dir: str | Path,
    manifest_path: str | Path,
    dry_run: bool = False,
) -> list[str]:
    """Scan Solomon raw files, convert to processed JSON, and generate a manifest CSV.

    Args:
        raw_dir: Directory containing Solomon .txt files.
        processed_dir: Output directory for processed .json files.
        manifest_path: Output path for the manifest CSV.
        dry_run: If True, scan but do not write files.

    Returns:
        List of processed instance names.
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    manifest_path = Path(manifest_path)

    txt_files = sorted(raw_dir.glob("*.txt"))
    if not txt_files:
        return []

    if not dry_run:
        processed_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    processed_names = []

    for txt_file in txt_files:
        name = txt_file.stem
        try:
            instance = load_solomon_instance(txt_file)
            processed_names.append(name)

            if not dry_run:
                out_path = processed_dir / f"{name}.json"
                instance_dict = instance.model_dump()
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(instance_dict, f, indent=2)

            rows.append({
                "instance_name": name,
                "family": instance.family,
                "size": instance.size,
                "source": "solomon",
                "raw_path": str(txt_file.resolve()),
                "processed_path": str(processed_dir / f"{name}.json") if not dry_run else "",
                "capacity": instance.vehicle_capacity,
            })
        except Exception as e:
            print(f"Warning: failed to process {name}: {e}")

    if not dry_run and rows:
        import csv

        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["instance_name", "family", "size", "source", "raw_path", "processed_path", "capacity"],
            )
            writer.writeheader()
            writer.writerows(rows)

    return processed_names


def preprocess_homberger(
    raw_dir: str | Path,
    processed_dir: str | Path,
    manifest_path: str | Path,
    dry_run: bool = False,
) -> list[str]:
    """Scan Homberger raw files, convert to processed JSON, and generate a manifest CSV.

    Args:
        raw_dir: Directory containing Homberger .txt files.
        processed_dir: Output directory for processed .json files.
        manifest_path: Output path for the manifest CSV.
        dry_run: If True, scan but do not write files.

    Returns:
        List of processed instance names.
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    manifest_path = Path(manifest_path)

    txt_files = sorted(raw_dir.glob("*.txt"))
    if not txt_files:
        return []

    if not dry_run:
        processed_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    processed_names = []

    for txt_file in txt_files:
        name = txt_file.stem
        try:
            instance = load_homberger_instance(txt_file)
            processed_names.append(name)

            if not dry_run:
                out_path = processed_dir / f"{name}.json"
                instance_dict = instance.model_dump()
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(instance_dict, f, indent=2)

            rows.append({
                "instance_name": name,
                "family": instance.family,
                "size": instance.size,
                "source": "homberger",
                "raw_path": str(txt_file.resolve()),
                "processed_path": str(processed_dir / f"{name}.json") if not dry_run else "",
                "capacity": instance.vehicle_capacity,
            })
        except Exception as e:
            print(f"Warning: failed to process {name}: {e}")

    if not dry_run and rows:
        import csv

        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["instance_name", "family", "size", "source", "raw_path", "processed_path", "capacity"],
            )
            writer.writeheader()
            writer.writerows(rows)

    return processed_names


def prepare_dataset(
    dataset_name: str,
    raw_dir: Path,
    processed_dir: Path,
    manifest_path: Path,
    dry_run: bool = False,
) -> list[str]:
    """Prepare dataset by name.
    
    Args:
        dataset_name: Name of dataset ('solomon' or 'homberger')
        raw_dir: Directory containing raw files
        processed_dir: Output directory for processed files
        manifest_path: Output path for manifest CSV
        dry_run: If True, scan but do not write files
        
    Returns:
        List of processed instance names
    """
    if dataset_name == "solomon":
        return preprocess_solomon(raw_dir, processed_dir, manifest_path, dry_run)
    elif dataset_name == "homberger":
        return preprocess_homberger(raw_dir, processed_dir, manifest_path, dry_run)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
