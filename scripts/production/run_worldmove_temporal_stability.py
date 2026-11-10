"""Frozen sample-index loading and equal-date exposure summaries."""


from __future__ import annotations


import csv


import hashlib


import json


from collections import defaultdict


from datetime import datetime, timezone


from pathlib import Path


import numpy as np


from run_worldmove_five_city_pilot import (
    ROOT,
    AnchorResult,
    build_city_o3_lookup,
    ensure_target_hourly_files,
    exposure_arrays,
    infer_anchor,
    load_worldmove,
    md5_file,
    qc_row,
    read_csv,
    restricted_anchor,
    sha256_file,
    software_qa,
    summarize_exposure,
    validate_netcdf_inputs,
    write_csv,
)










def load_sample_ids(config: dict, cities: list[str]) -> tuple[dict[str, np.ndarray], list[dict]]:
    source = config["frozen_schedule_sample"]
    ids_by_city = {}
    qc = []
    with np.load(ROOT / source["sample_ids_file"]) as archive:
        for city in cities:
            key = source["sample_id_arrays"][city]
            ids = np.asarray(archive[key], dtype=np.int64)
            if ids.shape != (5000,) or len(np.unique(ids)) != 5000 or not np.array_equal(ids, np.sort(ids)):
                raise ValueError(f"Saved sample IDs invalid: {city}")
            ids_by_city[city] = ids
            qc.append(qc_row("frozen_inputs", "saved_sample5000_ids", "PASS", city=city,
                             value=5000, unit="schedules", details=f"array={key}; no new random draw"))
    return ids_by_city, qc


def q_iqr(values: np.ndarray) -> tuple[float, float, float]:
    q1, q3 = np.quantile(values, [0.25, 0.75])
    return float(q1), float(q3), float(q3 - q1)


def grouped_summary(rows: list[dict], city: str, month: str | None, variant: str) -> dict:
    subset = [r for r in rows if r["city"] == city and
              (month is None or r["date"].startswith(month))]
    delta = np.asarray([r["mean_delta_ug_m3"] for r in subset], dtype=float)
    abs_delta = np.asarray([r["mean_abs_delta_ug_m3"] for r in subset], dtype=float)
    dq1, dq3, diqr = q_iqr(delta)
    aq1, aq3, aiqr = q_iqr(abs_delta)
    return {
        "city": city,
        "month": month or "May_July_September",
        "analysis_variant": variant,
        "date_count": len(subset),
        "date_equal_mean_R_ug_m3": float(np.mean([r["mean_R_ug_m3"] for r in subset])),
        "date_equal_mean_D_ug_m3": float(np.mean([r["mean_D_ug_m3"] for r in subset])),
        "date_equal_mean_delta_ug_m3": float(delta.mean()),
        "date_equal_mean_abs_delta_ug_m3": float(abs_delta.mean()),
        "daily_mean_delta_sd_ug_m3": float(delta.std(ddof=1)),
        "daily_mean_delta_q1_ug_m3": dq1,
        "daily_mean_delta_q3_ug_m3": dq3,
        "daily_mean_delta_iqr_ug_m3": diqr,
        "daily_mean_abs_delta_sd_ug_m3": float(abs_delta.std(ddof=1)),
        "daily_mean_abs_delta_q1_ug_m3": aq1,
        "daily_mean_abs_delta_q3_ug_m3": aq3,
        "daily_mean_abs_delta_iqr_ug_m3": aiqr,
        "proportion_dates_mean_delta_gt_0": float((delta > 0).mean()),
        "minimum_daily_mean_delta_ug_m3": float(delta.min()),
        "maximum_daily_mean_delta_ug_m3": float(delta.max()),
        "minimum_daily_mean_abs_delta_ug_m3": float(abs_delta.min()),
        "maximum_daily_mean_abs_delta_ug_m3": float(abs_delta.max()),
        "minimum_valid_schedule_count": int(min(r["valid_schedule_count"] for r in subset)),
        "maximum_valid_schedule_count": int(max(r["valid_schedule_count"] for r in subset)),
        "mean_missing_pair_rate": float(np.mean([r["missing_or_out_of_support_pair_rate"] for r in subset])),
        "mean_missing_slot_rate": float(np.mean([r["missing_or_out_of_support_slot_rate"] for r in subset])),
    }

