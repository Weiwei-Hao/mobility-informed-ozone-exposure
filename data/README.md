# Frozen supporting data

The analytical layer has 47 distinct frozen payloads (10,414,660 bytes) for 51 indexed roles. Four exact duplicate views share one stored file. The separately indexed real-input fixture adds two files. Dictionaries, mappings and source metadata are documentation, not extra scientific results.

| Location | Content and analysis level |
|---|---|
| `tables/worldmove_formal_city_date.csv` | Primary 33 × 153 city-date exposure results |
| `tables/worldmove_formal_city_month.csv`, `worldmove_formal_city_may_sep.csv` | 165 city-month and 33 city-period aggregates |
| `tables/worldmove_formal_high_confidence.csv` | Frozen high-confidence results, including stored date-window variants |
| `tables/worldmove_mechanism_predictors_city*.csv` | T/E/H/A analytical inputs, with stored primary/high-confidence variants |
| `tables/worldmove_mechanism_*primary.csv`, `*sensitivity.csv`, `*loco*.csv` | Frozen coefficient/interval, sensitivity and held-city evaluation references |
| `tables/worldmove_mechanism_ml*.csv` | Fixed supplementary prediction scores and out-of-fold predictions |
| `tables/Figure*.csv` | Exact approved display views; retained precision/selection rather than re-derived views |
| `tables/FigureS1_eligibility_support_*.csv`, `study_unit_coordinates.csv` | Historical eligibility records, retained schedule counts and study-unit coordinates |
| `metadata/` | Frozen estimator/frame and descriptive-summary metadata |
| `sample_indices/` | All 33 arrays of 5,000 frozen zero-based source schedule row IDs, plus selection metadata |
| `real_subset/` | Actual decoded Yinchuan inputs for 2019-05-01, from synthetic schedules and modelled ozone |

`file_index.csv` contains original logical paths, current **data-directory-relative** `stored_path`, byte counts, hashes, roles and the two documented field-reduction transformations. `data_dictionary.csv` uses the same path convention. `array_dictionary.csv` gives original `logical_path` and current data-relative `path`; it records NPZ arrays, shapes, units and meanings. `aliases.json` resolves the historical input paths used by the preserved scientific functions. `figure_mapping.csv` and `paper_correspondence.csv` record the release-era mapping. Historical logical paths identify the source files.

## Final supplementary material

The final paper covers 33 city units and 153 dates. Ozone support requires product coverage and at least 97.5% complete-pair retention on every study date.

| Final item | Source in v1.0.0 |
|---|---|
| Table S1: 33-city results and Map IDs | `tables/worldmove_formal_city_may_sep.csv`, the `may_september_pooled` rows of `tables/worldmove_formal_high_confidence.csv`, `tables/worldmove_formal_sample_counts.csv` and `tables/study_unit_coordinates.csv`. The Map IDs also label Figure S1. |
| Table S2: six candidates with insufficient ozone support | Rows marked `EXCLUDED_O3_SUPPORT` in `tables/FigureS1_eligibility_support_eligibility.csv`. |
| Figure S2: primary and high-confidence retained schedules for 33 cities | Columns `o3_complete_valid_n` and `high_confidence_o3_complete_valid_n` in `tables/FigureS1_eligibility_support_counts.csv`. |

The 43-package eligibility file preserves the screening history. Published display names **Hohhot** and **Fangchenggang** correspond to the archived labels `Huhot` and `Fangcheng_Gang`.

This post-release guide supersedes the earlier cross-references for these items in the frozen indexes and dictionaries. Use the tagged release or Zenodo archive to verify the original package against its `MANIFEST.csv`. On the default branch, the recorded README hashes predate this clarification.

Concentrations and D/R differences use µg/m³; dates are ozone scenario dates. The dictionaries specify other units and denominators. Rows with different `variant` or date-window labels are not independent extra cities/dates: public checks apply the frozen primary selectors. Blank inapplicable/unavailable statistics remain blank; no imputation was performed.

Every analytical payload retains its frozen SHA-256. The real fixture retains its separate NPZ and metadata hashes. Its ozone concentrations start after native product decoding and are sparse cell values, not the national NetCDF product. Unit-test invented arrays live only in `tests/` and are never used as formal data.

Source attribution and licences are in `upstream_sources.json` and `../THIRD_PARTY_NOTICES.md`. Original study-derived data are licensed under CC BY 4.0 by Weiwei Hao; attributed upstream selections retain their source terms. See `LICENSE`. No complete upstream archives, national rasters, GIS basemap files, font files, private correspondence or manuscripts are included.
