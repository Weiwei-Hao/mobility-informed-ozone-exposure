# Reproduction guide

Use the four commands in the root README from the repository root. Inputs are read from `data/`; all generated results go to separate output directories. `data/aliases.json` resolves original logical input names to current data-relative paths. Production module filenames are kept stable to avoid unnecessary changes to imports.

## Supplied-data checks

`check_descriptive.py` checks 5,049 city-date rows, 165 city-month rows and 33 city-period rows, equal-date aggregation and the published descriptive rounding.

`check_results.py` refits primary/reduced between-city and city-demeaned models and the original 33 LOCO folds. It retains HC3 and model-specific Student-t intervals between cities, Driscoll–Kraay/Bartlett lag 7 with HAC correction and t32 within cities, original predictor-SD scaling, row order and equal-city anomaly RMSE. Coefficients, intervals and predictions are compared against frozen results. It does not run sensitivity or XGBoost training.

`reproduce.py subset` uses the actual 5,000 frozen Yinchuan schedules and sparse decoded native-product cells for 2019-05-01. It checks spatial/time matching, primary/strict anchors, R/D, signed/absolute differences and T/E/H/A. It inherits the fixed full-window support mask; it does not test national NetCDF decoding. These synthetic/modelled inputs are distinct from invented unit-test arrays.

## Preserved scientific workflows

For explicitly requested full linear/sensitivity refits from supplied derived data, run `python -B scripts/reproduce.py models --work-dir ../worldmove-full-models`. Add `--include-ml` only to explicitly run the original fixed supplementary prediction comparison. Neither broader workflow was executed in final slimming. Stored sensitivity/ML results remain available for inspection.

For upstream inputs, use `python -B scripts/release_pipeline.py analysis --input-root /path/to/legally-acquired/inputs --work-dir /path/to/new/run`. Exact archive members, hashes, schemas, IDs and directory paths are specified in `data_requirements.md` and `config/input_inventory.json`. Raw input paths must be populated separately. No automatic downloading or ID resampling occurs.

Use these public entrypoints rather than running production modules directly. They stage the original functions in a fresh workspace and bind validated input context. Predictor formula checks are preserved. Historical private-note checks and historical standalone executables are not requirements for using the supplied scientific functions.

The complete 33-city × 153-day national-input chain, all sensitivity/ML refits and modelled ozone product generation were not rerun during final packaging. No graphics, Word processing, font installation or GIS inputs are required for the supplied core statistical checks. Numerical figure-source tables remain inspectable; artwork production is outside this repository's scope.
