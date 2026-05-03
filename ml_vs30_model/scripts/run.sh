#!/usr/bin/env zsh

function csnotify {
   curl -d $1 ntfy.sh/W7T2QKNDH9Z4E3VJPRY8XACUL
}


# # NZ - No sample weights
# python model_cmds.py cv-train-catboost configs/run_configs/catboost_nz.yaml datasets/nz_dataset.parquet --id-suffix cv25_noSampleWeights_nzGeology --no-apply-vs30-sample-weights &

# # NZ - With sample weights
# python model_cmds.py cv-train-catboost configs/run_configs/catboost_nz.yaml datasets/nz_dataset.parquet --id-suffix cv25_vs30SampleWeights_nzGeology > /dev/null &


# Create NZ input grid
# python data_cmds.py create-nz-input-grid 0.01 /home/claudy/dev/work/data/vs30/grids/nz_input_grid_0p01 roughness topographic_slope compound_topographic_index geomorphon profile_curvature tangential_curvature terrain_ruggedness_index topographic_position_index vector_ruggedness_measure landform_entropy landform_shannon_index landform_uniformity absolute_depth_to_bedrock depth_to_groundwater nz_geology_category nz_distance_to_coast  --n-procs 12