# Data and code for mobility-informed ozone exposure assessment across 33 Chinese cities

Python code and supporting derived data for *Beyond residential averages: Mobility-related variability in ambient ozone exposure estimates across 33 Chinese cities*. The study pairs full-day residential-anchor and mobility-informed ambient ozone estimates across 33 city units and 153 dates in May–September 2019. Schedules are synthetic and ozone concentrations are modelled, not observed personal exposure.

Version **1.0.0**. Creator: **Weiwei Hao**, Anyang Institute of Technology.

For reproduction, use the [v1.0.0 release](https://github.com/Weiwei-Hao/mobility-informed-ozone-exposure/releases/tag/v1.0.0) or its [Zenodo archive (DOI 10.5281/zenodo.24113757)](https://doi.org/10.5281/zenodo.24113757). The default branch includes a [post-release guide to the final supplementary tables and figure](data/README.md#final-supplementary-material).

## Contents

- `data/tables/`: frozen exposure summaries, T/E/H/A inputs, model/sensitivity/LOCO results, supplementary prediction results and numerical figure-source tables.
- `data/sample_indices/`: exact frozen 33 × 5,000 original schedule row IDs.
- `data/real_subset/`: attributed, decoded single-day Yinchuan inputs for exposure checks.
- `data/file_index.csv` and dictionaries: file hashes, fields, units and paper correspondence.
- `scripts/`: core checks, exposure/predictor calculations and original statistical workflows.
- `config/`: frozen settings and upstream input inventories.

No image-production code or finished images are included. Full upstream archives and third-party GIS layers are not redistributed.

## Environment and reproduction

Tested with Python 3.11.15 and the versions pinned in `requirements.txt`. Core checks use NumPy, pandas, SciPy and statsmodels; the upstream and optional supplementary workflows additionally use netCDF4, scikit-learn and XGBoost.

From the repository root, use fresh output paths outside the repository:

```bash
python -B scripts/check_descriptive.py --output ../worldmove-run/descriptive.json
python -B scripts/check_results.py --output ../worldmove-run/models
python -B scripts/reproduce.py subset --report ../worldmove-run/subset.json
python -B -m unittest discover -s tests -v
```

The core check refits the frozen between-city and within-city primary/reduced models and original LOCO folds. Expected within-city R² is 0.445/0.576, held-city macro RMSE is 0.426/0.371 µg/m³, and 32/33 cities improve. The real-input subset checks time/grid matching, D/R and T/E/H/A after product decoding. Unit tests use separate invented arrays.

The full upstream pipeline and sensitivity code are retained; they were not rerun end-to-end during final packaging. See [reproduction guide](docs/reproduction_guide.md) and [data requirements](docs/data_requirements.md).

## Upstream sources

- Yuan, Yuan (2025), *WorldMove Dataset*, version 1: [Figshare](https://doi.org/10.6084/m9.figshare.30023491.v1); [associated paper](https://doi.org/10.1038/s41597-026-06555-2).
- Cheng, Fan; Li, Zhanqing; Wei, Jing (2023), *ChinaHighO3 H1K 2019*, version 1: [Zenodo](https://doi.org/10.5281/zenodo.10035857); [methods](https://doi.org/10.1016/j.rse.2024.114482).

Source records specify CC BY 4.0. Attribution, transformations and boundaries are described in [third-party notices](THIRD_PARTY_NOTICES.md).

## Citation and licence

Please cite this code/data release using `CITATION.cff`. Creator: Weiwei Hao, Anyang Institute of Technology. Repository: [Weiwei-Hao/mobility-informed-ozone-exposure](https://github.com/Weiwei-Hao/mobility-informed-ozone-exposure). No manuscript DOI or ORCID is asserted.

Original code is [MIT](LICENSE); original study-derived data are [CC BY 4.0](data/LICENSE). Attributed upstream selections retain their source terms. See [licensing scopes](LICENSE.md) and [third-party notices](THIRD_PARTY_NOTICES.md).
