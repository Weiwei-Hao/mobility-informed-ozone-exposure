"""Exact 2019 May–September hourly-file selection and completeness checks."""


from __future__ import annotations


import json


from collections import defaultdict


from datetime import datetime, timezone


from pathlib import Path


import numpy as np


from run_worldmove_five_city_pilot import (
    ROOT, build_city_o3_lookup, ensure_target_hourly_files, exposure_arrays,
    infer_anchor, load_worldmove, md5_file, qc_row, read_csv,
    restricted_anchor, sha256_file, software_qa, summarize_exposure,
    validate_netcdf_inputs, write_csv,
)




MONTHS = ("2019-05", "2019-06", "2019-07", "2019-08", "2019-09")


def target_files(config: dict, manifest: list[dict]) -> tuple[dict[str, list[Path]], list[dict]]:
    dates = config["temporal_window"]["dates"]
    files = {}
    qc = []
    for row in manifest:
        archive = ROOT / row["target_path"]
        first, last = row["target_dates"].split(" to ")
        selected = [d for d in dates if first <= d <= last]
        extracted_dir = archive.with_suffix("")
        missing = False
        for day in selected:
            compact = day.replace("-", "")
            paths = [extracted_dir / f"CHAP_O3_H1K_{compact}_{h:02d}00.nc" for h in range(24)]
            files[day] = paths
            missing |= not all(p.is_file() for p in paths)
        if missing:
            if not archive.is_file():
                raise FileNotFoundError(f"Missing extracted hours and raw archive: {archive.name}")
            ensure_target_hourly_files([{"recommended_path": row["target_path"],
                                         "selected_dates": ";".join(selected)}])
        for day in selected:
            count = sum(p.is_file() for p in files[day])
            qc.append(qc_row("hourly_completeness", "24_hourly_files",
                             "PASS" if count == 24 else "FAIL", date=day,
                             value=count, unit="files", details=f"archive={archive.name}"))
            if count != 24:
                raise ValueError(f"Missing hourly NetCDF: {day}")
    if list(files) != dates or len(files) != 153:
        raise ValueError("Extracted dates differ from config")
    return files, qc

