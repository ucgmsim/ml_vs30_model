#!/usr/bin/env zsh
if [[ -z "$1" ]]; then
    echo "Usage: $0 <full-model-dir>"
    exit 1
fi

export MPLBACKEND=Agg

function csnotify {
   curl -d $1 ntfy.sh/W7T2QKNDH9Z4E3VJPRY8XACUL
}


# Test predictions
python model_cmds.py test-predictions ${VS30_MODEL_BASE_DATA_DIR}/datasets/nz_combined.parquet $1 ${VS30_MODEL_BASE_DATA_DIR}/datasets/nz_site_db_test_sites/test_sites.npy

# Estimate NZ
python model_cmds.py estimate-vs30-nz $1 ${VS30_MODEL_BASE_DATA_DIR}/grids/nz_input_grid_100m/input_grid.nc

# Add kriging to NZ estimates
python model_cmds.py add-krigged-vs30 $1

# Add foster & compute (krigged) residuals
python model_cmds.py add-kriged-vs30-foster-original-residuals $1

# Add other NZ estimates
python model_cmds.py add-other-nz-estimates $1/nz_vs30_results.nc --add-foster

# Add waterfall plots
python plot_cmds.py gen-feature-importance-plots --gen-waterfall-plots $1

# Add grid SHAP values
python model_cmds.py add-grid-SHAP-values $1 ${VS30_MODEL_BASE_DATA_DIR}/grids/nz_input_grid_100m/input_grid.nc --n-procs 16

csnotify "Post-processing complete"
