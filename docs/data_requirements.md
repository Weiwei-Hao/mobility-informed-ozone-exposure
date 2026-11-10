# Exact inputs and acquisition

## Public source products

WorldMove: Yuan, Yuan (2025), *WorldMove Dataset*, **version 1**, Figshare DOI `10.6084/m9.figshare.30023491.v1`. The locally archived version-1 metadata names Yuan Yuan and CC BY 4.0. Associated methods/data paper: Yuan et al. (2026), DOI `10.1038/s41597-026-06555-2`. Use the versioned release, not a changing live-city endpoint.

The version-specific Figshare API was also checked on 2026-11-09 and is consistent with the archived creator/licence metadata. The archived source `data.zip` is 3,088,383,990 bytes, Figshare file ID 57561772; its download URL and MD5 `63f64cbb68d754cc04069eb8dbeca3ad` are recorded in `config/worldmove_input_files.csv`. That file enumerates every required NPZ member and expected NPZ hash. Download through the legitimate Figshare release and extract only the listed 33 city members into their listed `data/raw/worldmove/<released_key>.npz` paths. No global archive or full city NPZs are redistributed in this candidate.

ChinaHighO3: Cheng, Fan; Li, Zhanqing; Wei, Jing (2023), *ChinaHighO3: Hourly Seamless 1 km Ground-Level O3 Dataset for China (2019)*, **version 1**, DOI `10.5281/zenodo.10035857`; methods DOI `10.1016/j.rse.2024.114482`. The official record/API was checked on 2026-11-09 and specifies version 1, CC BY 4.0 and Beijing local time. The study window is 2019-05-01 through 2019-09-30: 153 dates, 3,672 hourly fields. This is a modelled ozone product, not personal-exposure measurements.

Obtain the 22 RAR files listed in `config/o3_acquisition_manifest.csv` from that versioned record, verify their stored MD5/size records, and extract into `data/raw/china_high_o3/2019/<archive_stem>/CHAP_O3_H1K_YYYYMMDD_HH00.nc`. Required names/dates are enumerated in `config/input_inventory.json`. No national raster or raw archive is redistributed.

`data/upstream_sources.json` records source identifiers, creators, rights metadata and the **2026-11-09 verification/access date**. This date is not presented as the unknown original first-download date. Frozen source hashes and archive members identify the exact historical inputs independently of a download date. Internal correspondence used for source confirmation is not included.

## City and schedule selection

The exact 33-city list, 10 exclusions and input-only eligibility criteria are in the preserved frame JSON; the data package provides the 43-unit eligibility table with only necessary identity/support fields. The frame uses released WorldMove footprints, valid structure, defined mainland-China scope and at least 97.5% full-pool main-anchor complete-pair support on **every** study date, assessed before subsampling. All eligible units were retained; there are no population expansion weights.

The supplied `data/sample_indices/worldmove_formal_sample_ids.npz` contains all 33 arrays `<city>__sample5000`: 5,000 sorted unique original zero-based NPZ rows per city, selected before exclusions. Five reference-city IDs were reused exactly; additional cities used NumPy default_rng/PCG64 with seed 20260810, reset per city, without replacement. The metadata records file/array hashes. **Do not redraw any IDs**, including the reference cities; do not replenish excluded schedules.

WorldMove NPZ inputs contain `traj`, `grid`, `pop`, `poi`. Each `traj` row has 48 half-hour grid identifiers. Grid IDs map to released centroid longitude/latitude. The unique modal grid in slots 0–11 must occupy at least 6/12 slots; tied modes fail. The high-confidence subset requires at least 9/12 with the same anchor. Stationary schedules remain eligible.

## Matching and outcomes

Native ozone `lat`/`lon` coordinates are matched by nearest coordinate, with exact ties toward the lower coordinate; no interpolation or extrapolation. Two successive half-hour slots use hour `slot//2` on the same date and Beijing-time UTC+8 clock. This is the confirmed study convention, not an unresolved time-semantics question.

Stored O3 uint16 values use fill 65535, scale approximately 0.1 and offset 0 before concentration extraction in µg/m³. The upstream-input validator checks the original `(3599,4799)` schema. Full-day R averages fixed-anchor hourly concentrations; D follows the 48 locations. Calculate each schedule's full-day D−R before absolute value, then average complete schedules. Monthly summaries use actual dates; study-period summaries use 153 equal-date weights; across-city comparisons weight 33 cities equally. Both paths require complete ozone support; no imputation.

## Derived layer and small fixture

The repository data directory provides the actual analytical/plot inputs, frozen statistical outputs and all frozen IDs. `data/aliases.json` resolves four exact duplicates without storing repeated files. Only two tables were reduced to needed columns (eligibility and representative points); retained rows/values are unchanged and source/release hashes are recorded. Numerical figure-source views are included for inspection; image production is outside this repository scope. Data/array dictionaries describe units and row levels.

The real-input fixture contains exact 5,000 selected Yinchuan trajectories, native product axes and sparse decoded product-cell O3 for 2019-05-01. It is attributed to both CC BY 4.0 products and clearly marked synthetic/modelled. It is **not** invented demo data and **not** the full national NetCDF ingestion path. Fixed full-window support is inherited from the frozen project.
