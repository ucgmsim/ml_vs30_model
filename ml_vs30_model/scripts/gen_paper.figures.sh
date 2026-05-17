#!/usr/bin/env zsh

# Break if any command fails
set -e

scripts_dir="/Users/claudy/dev/work/code/ml_vs30_model/ml_vs30_model/scripts"
dataset_ffp="${VS30_MODEL_BASE_DATA_DIR}/datasets/nz_site_db_dataset.parquet"
model_dir="${VS30_MODEL_BASE_DATA_DIR}/results/0513_121933_full_nzSiteDB_20Iterations"


out_dir="/Users/claudy/dev/work/tmp/vs30_plots"

### ------------------- Spatial Figures ------------------------------

### Site map
# echo "Generating site map..."
# python "${scripts_dir}/gen_paper_figures.py" gen-site-map "${dataset_ffp}" "${out_dir}"

### Vs30 map
echo "Generating Vs30 map..."
python "${scripts_dir}/gen_paper_figures.py" gen-vs30-map "${model_dir}/nz_vs30_results.nc" "${out_dir}" "ni"
python "${scripts_dir}/gen_paper_figures.py" gen-vs30-map "${model_dir}/nz_vs30_results.nc" "${out_dir}" "si"

### Residual map
# echo "Generating residual map..."
# python "${scripts_dir}/gen_paper_figures.py" gen-residual-map "${model_dir}/nz_vs30_results.nc" "${out_dir}" "ni"
# python "${scripts_dir}/gen_paper_figures.py" gen-residual-map "${model_dir}/nz_vs30_results.nc" "${out_dir}" "si"