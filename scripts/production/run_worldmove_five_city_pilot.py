"""Residential-anchor inference, native-grid matching and paired daily exposure calculations."""


from __future__ import annotations


import argparse


import csv


import hashlib


import json


import re


import subprocess


from collections import Counter


from dataclasses import dataclass


from datetime import datetime, timezone


from pathlib import Path


from typing import Any


import numpy as np


from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[1]


O3_FILENAME_RE = re.compile(r"CHAP_O3_H1K_(\d{8})_(\d{2})00\.nc$")


@dataclass
class AnchorResult:
    grid_index: np.ndarray
    valid: np.ndarray
    dominance: np.ndarray


@dataclass
class ExposureResult:
    complete: np.ndarray
    residential: np.ndarray
    dynamic: np.ndarray
    delta: np.ndarray
    diurnal_delta: np.ndarray
    invalid_pair_rate: float
    missing_slot_rate: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def qc_row(
    category: str,
    item: str,
    status: str,
    *,
    city: str = "",
    date: str = "",
    value: Any = "",
    unit: str = "",
    details: str = "",
) -> dict[str, Any]:
    return {
        "category": category,
        "item": item,
        "city": city,
        "date": date,
        "status": status,
        "value": value,
        "unit": unit,
        "details": details,
    }


def ensure_target_hourly_files(
    manifest_rows: list[dict[str, str]],
) -> tuple[dict[str, list[Path]], list[dict[str, Any]]]:
    by_date: dict[str, list[Path]] = {}
    qa: list[dict[str, Any]] = []
    for row in manifest_rows:
        archive = ROOT / row["recommended_path"]
        listing = subprocess.run(
            ["bsdtar", "-tf", str(archive)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        member_map: dict[tuple[str, int], str] = {}
        for member in listing:
            match = O3_FILENAME_RE.search(Path(member).name)
            if match:
                member_map[(match.group(1), int(match.group(2)))] = member

        target_members: list[str] = []
        output_paths: dict[str, list[Path]] = {}
        for iso_date in selected_dates_from_manifest(row):
            compact = iso_date.replace("-", "")
            paths: list[Path] = []
            for hour in range(24):
                key = (compact, hour)
                if key not in member_map:
                    raise ValueError(f"Archive lacks frozen hour {iso_date} {hour:02d}:00: {archive}")
                member = member_map[key]
                target_members.append(member)
                paths.append(archive.parent / member)
            output_paths[iso_date] = paths

        missing_members = [
            member
            for member in target_members
            if not (archive.parent / member).exists()
        ]
        if missing_members:
            subprocess.run(
                [
                    "bsdtar",
                    "-xkf",
                    str(archive),
                    "-C",
                    str(archive.parent),
                    *missing_members,
                ],
                check=True,
            )

        for iso_date, paths in output_paths.items():
            existing = [path for path in paths if path.exists()]
            passed = len(existing) == 24
            qa.append(
                qc_row(
                    "hourly_completeness",
                    "24_hourly_files",
                    "PASS" if passed else "FAIL",
                    date=iso_date,
                    value=len(existing),
                    unit="files",
                    details=f"archive={archive.name}",
                )
            )
            if not passed:
                raise ValueError(f"Incomplete extracted day: {iso_date}")
            by_date[iso_date] = paths
    return by_date, qa


def validate_netcdf_inputs(
    files_by_date: dict[str, list[Path]],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    reference_lat: np.ndarray | None = None
    reference_lon: np.ndarray | None = None
    reference_schema: tuple[Any, ...] | None = None
    reference_attrs: tuple[Any, ...] | None = None
    qa: list[dict[str, Any]] = []

    for iso_date, paths in files_by_date.items():
        date_passed = True
        for expected_hour, path in enumerate(paths):
            match = O3_FILENAME_RE.search(path.name)
            if match is None or int(match.group(2)) != expected_hour:
                raise ValueError(f"Unexpected O3 filename/hour: {path}")
            with Dataset(path) as dataset:
                if set(dataset.variables) != {"lat", "lon", "O3"}:
                    raise ValueError(f"Unexpected variables in {path.name}: {list(dataset.variables)}")
                lat = np.asarray(dataset.variables["lat"][:], dtype=np.float64)
                lon = np.asarray(dataset.variables["lon"][:], dtype=np.float64)
                ozone = dataset.variables["O3"]
                schema = (
                    dataset.data_model,
                    ozone.dimensions,
                    ozone.shape,
                    str(ozone.dtype),
                )
                attrs = (
                    str(ozone.getncattr("units")),
                    int(ozone.getncattr("_FillValue")),
                    float(ozone.getncattr("scale_factor")),
                    float(ozone.getncattr("add_offset")),
                )
                if reference_schema is None:
                    reference_schema = schema
                    reference_attrs = attrs
                    reference_lat = lat
                    reference_lon = lon
                else:
                    date_passed &= schema == reference_schema and attrs == reference_attrs
                    date_passed &= np.array_equal(lat, reference_lat)
                    date_passed &= np.array_equal(lon, reference_lon)
        qa.append(
            qc_row(
                "netcdf_schema",
                "24_files_consistent_with_reference",
                "PASS" if date_passed else "FAIL",
                date=iso_date,
                value=24,
                unit="files",
            )
        )
        if not date_passed:
            raise ValueError(f"NetCDF schema/coordinate mismatch: {iso_date}")

    assert reference_lat is not None and reference_lon is not None
    assert reference_schema is not None and reference_attrs is not None
    if not np.all(np.diff(reference_lat) < 0):
        raise ValueError("O3 latitude axis is not strictly descending")
    if not np.all(np.diff(reference_lon) > 0):
        raise ValueError("O3 longitude axis is not strictly ascending")
    if reference_attrs[0] != "µg/m3":
        raise ValueError(f"Unexpected O3 unit: {reference_attrs[0]}")

    metadata = {
        "data_model": reference_schema[0],
        "dimensions": list(reference_schema[1]),
        "shape": list(reference_schema[2]),
        "storage_dtype": reference_schema[3],
        "unit": reference_attrs[0],
        "fill_value": reference_attrs[1],
        "scale_factor": reference_attrs[2],
        "add_offset": reference_attrs[3],
        "latitude_first_last": [float(reference_lat[0]), float(reference_lat[-1])],
        "longitude_first_last": [float(reference_lon[0]), float(reference_lon[-1])],
        "time_representation": "one 2D hourly file; local GMT+8 hour encoded in filename",
    }
    return reference_lat, reference_lon, metadata, qa


def nearest_axis_indices(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    ascending = axis[0] < axis[-1]
    ordered = axis if ascending else axis[::-1]
    positions = np.searchsorted(ordered, values)
    positions = np.clip(positions, 1, len(ordered) - 1)
    left = positions - 1
    right = positions
    selected = np.where(
        np.abs(ordered[right] - values) < np.abs(ordered[left] - values),
        right,
        left,
    )
    return selected if ascending else len(axis) - 1 - selected


def load_worldmove(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=True) as archive:
        if set(archive.files) != {"grid", "poi", "pop", "traj"}:
            raise ValueError(f"Unexpected arrays in {path}: {archive.files}")
        trajectories = np.asarray(archive["traj"], dtype=np.int64)
        raw_grid = archive["grid"].item()
        pop_shape = archive["pop"].shape
        poi_shape = archive["poi"].shape
    grid = {int(index): coordinate for index, coordinate in raw_grid.items()}
    if trajectories.ndim != 2 or trajectories.shape[1] != 48:
        raise ValueError(f"WorldMove trajectory shape failed: {path}: {trajectories.shape}")
    if sorted(grid) != list(range(len(grid))):
        raise ValueError(f"WorldMove grid keys failed: {path}")
    if trajectories.min() < 0 or trajectories.max() >= len(grid):
        raise ValueError(f"WorldMove trajectory index failed: {path}")
    coordinates = np.asarray([grid[index] for index in range(len(grid))], dtype=np.float64)
    return trajectories, coordinates, {
        "traj_shape": list(trajectories.shape),
        "grid_count": len(grid),
        "pop_shape": list(pop_shape),
        "poi_shape": list(poi_shape),
    }


def infer_anchor(trajectories: np.ndarray, columns: np.ndarray) -> AnchorResult:
    anchor = np.full(len(trajectories), -1, dtype=np.int64)
    valid = np.zeros(len(trajectories), dtype=bool)
    dominance = np.zeros(len(trajectories), dtype=np.float64)
    for row_number, values in enumerate(trajectories[:, columns]):
        counts = Counter(int(value) for value in values)
        maximum = max(counts.values())
        winners = [value for value, count in counts.items() if count == maximum]
        dominance[row_number] = maximum / len(values)
        if len(winners) == 1 and dominance[row_number] >= 0.5:
            anchor[row_number] = winners[0]
            valid[row_number] = True
    return AnchorResult(anchor, valid, dominance)


def restricted_anchor(anchor: AnchorResult, valid: np.ndarray) -> AnchorResult:
    return AnchorResult(anchor.grid_index.copy(), valid.copy(), anchor.dominance.copy())


def build_city_o3_lookup(
    coordinates: np.ndarray,
    files_by_date: dict[str, list[Path]],
    o3_lat: np.ndarray,
    o3_lon: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]]]:
    lon_index = nearest_axis_indices(o3_lon, coordinates[:, 0])
    lat_index = nearest_axis_indices(o3_lat, coordinates[:, 1])
    outside = (
        (coordinates[:, 0] < o3_lon.min())
        | (coordinates[:, 0] > o3_lon.max())
        | (coordinates[:, 1] < o3_lat.min())
        | (coordinates[:, 1] > o3_lat.max())
    )
    lat_min, lat_max = int(lat_index.min()), int(lat_index.max())
    lon_min, lon_max = int(lon_index.min()), int(lon_index.max())

    lat_delta = np.abs(o3_lat[lat_index] - coordinates[:, 1])
    lon_delta = np.abs(o3_lon[lon_index] - coordinates[:, 0])
    distance_m = np.hypot(
        lat_delta * 111_320.0,
        lon_delta * 111_320.0 * np.cos(np.deg2rad(coordinates[:, 1])),
    )

    lookup: dict[str, np.ndarray] = {}
    diagnostics: dict[str, dict[str, float]] = {}
    for iso_date, paths in files_by_date.items():
        values = np.full((len(coordinates), 24), np.nan, dtype=np.float32)
        for hour, path in enumerate(paths):
            with Dataset(path) as dataset:
                crop = dataset.variables["O3"][lat_min : lat_max + 1, lon_min : lon_max + 1]
                crop_values = np.asarray(
                    crop.filled(np.nan) if np.ma.isMaskedArray(crop) else crop,
                    dtype=np.float32,
                )
                values[:, hour] = crop_values[
                    lat_index - lat_min,
                    lon_index - lon_min,
                ]
        values[outside, :] = np.nan
        lookup[iso_date] = values
        diagnostics[iso_date] = {
            "outside_coordinate_bounds_grid_proportion": float(outside.mean()),
            "missing_grid_hour_proportion": float((~np.isfinite(values)).mean()),
            "grid_with_any_missing_proportion": float((~np.isfinite(values)).any(axis=1).mean()),
            "median_nearest_cell_distance_m": float(np.median(distance_m)),
            "maximum_nearest_cell_distance_m": float(distance_m.max()),
        }
    return lookup, diagnostics


def exposure_arrays(
    trajectories: np.ndarray,
    anchor: AnchorResult,
    ozone: np.ndarray,
) -> ExposureResult:
    slot_hours = np.arange(48, dtype=np.int64) // 2
    # Promote both paths to float64 before averaging so algebraically identical
    # stationary schedules follow the same numerical precision as R.
    dynamic_slots = np.asarray(ozone[trajectories, slot_hours], dtype=np.float64)
    residential_hours = np.full((len(trajectories), 24), np.nan, dtype=np.float64)
    residential_hours[anchor.valid] = ozone[anchor.grid_index[anchor.valid], :]
    residential_slots = np.repeat(residential_hours, 2, axis=1)
    complete = (
        anchor.valid
        & np.all(np.isfinite(dynamic_slots), axis=1)
        & np.all(np.isfinite(residential_slots), axis=1)
    )
    residential = np.full(len(trajectories), np.nan, dtype=np.float64)
    dynamic = np.full(len(trajectories), np.nan, dtype=np.float64)
    residential[complete] = residential_slots[complete].mean(axis=1)
    dynamic[complete] = dynamic_slots[complete].mean(axis=1)
    delta = dynamic - residential
    dynamic_hourly = dynamic_slots.reshape(len(trajectories), 24, 2).mean(axis=2)
    diurnal_delta = np.full((len(trajectories), 24), np.nan, dtype=np.float64)
    diurnal_delta[complete] = dynamic_hourly[complete] - residential_hours[complete]
    required = np.concatenate([dynamic_slots, residential_slots], axis=1)
    required_valid_anchor = required[anchor.valid]
    missing_slot_rate = (
        float((~np.isfinite(required_valid_anchor)).mean())
        if len(required_valid_anchor)
        else np.nan
    )
    invalid_pair_rate = (
        float((anchor.valid & ~complete).sum() / anchor.valid.sum())
        if anchor.valid.any()
        else np.nan
    )
    return ExposureResult(
        complete,
        residential,
        dynamic,
        delta,
        diurnal_delta,
        invalid_pair_rate,
        missing_slot_rate,
    )


def summarize_exposure(
    city: str,
    iso_date: str,
    date_set: str,
    variant: str,
    anchor_name: str,
    anchor_filter: str,
    sample_count: int,
    anchor: AnchorResult,
    exposure: ExposureResult,
) -> dict[str, Any]:
    complete = exposure.complete
    if not complete.any():
        raise ValueError(f"No valid pairs: {city}, {iso_date}, {variant}")
    residential = exposure.residential[complete]
    dynamic = exposure.dynamic[complete]
    delta = exposure.delta[complete]
    q1, q3 = np.quantile(delta, [0.25, 0.75])
    return {
        "city": city,
        "date": iso_date,
        "date_set": date_set,
        "analysis_variant": variant,
        "anchor_rule": anchor_name,
        "anchor_filter": anchor_filter,
        "sampled_schedule_count": sample_count,
        "valid_schedule_count": int(complete.sum()),
        "anchor_excluded_count": int((~anchor.valid).sum()),
        "ambiguous_or_anchor_excluded_proportion": float((~anchor.valid).mean()),
        "missing_or_out_of_support_pair_rate": exposure.invalid_pair_rate,
        "missing_or_out_of_support_slot_rate": exposure.missing_slot_rate,
        "mean_R_ug_m3": float(residential.mean()),
        "mean_D_ug_m3": float(dynamic.mean()),
        "mean_delta_ug_m3": float(delta.mean()),
        "mean_abs_delta_ug_m3": float(np.abs(delta).mean()),
        "median_delta_ug_m3": float(np.median(delta)),
        "delta_q1_ug_m3": float(q1),
        "delta_q3_ug_m3": float(q3),
        "delta_iqr_ug_m3": float(q3 - q1),
        "positive_delta_proportion": float((delta > 0).mean()),
        "negative_delta_proportion": float((delta < 0).mean()),
        "zero_delta_proportion": float((delta == 0).mean()),
    }


def software_qa(
    trajectories: np.ndarray,
    anchor: AnchorResult,
    exposure: ExposureResult,
) -> tuple[int, float, int, float]:
    stationary = np.all(trajectories == trajectories[:, [0]], axis=1)
    tested = stationary & exposure.complete
    if not tested.any():
        raise ValueError("No stationary complete schedules available for QA")
    stationary_error = float(np.max(np.abs(exposure.delta[tested])))
    uniform_field = np.full((int(trajectories.max()) + 1, 24), 100.0, dtype=np.float64)
    uniform_exposure = exposure_arrays(trajectories, anchor, uniform_field)
    uniform_tested = uniform_exposure.complete
    if not uniform_tested.any():
        raise ValueError("No complete schedules available for artificial uniform-field QA")
    uniform_count = int(uniform_tested.sum())
    uniform_error = float(np.max(np.abs(uniform_exposure.delta[uniform_tested])))
    return int(tested.sum()), stationary_error, uniform_count, uniform_error

