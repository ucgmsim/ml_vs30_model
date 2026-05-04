#!/usr/bin/env zsh

function csnotify {
   curl -d $1 ntfy.sh/W7T2QKNDH9Z4E3VJPRY8XACUL
}


# # NZ - No sample weights
# python model_cmds.py cv-train-catboost configs/run_configs/catboost_nz.yaml datasets/nz_dataset.parquet --id-suffix cv25_noSampleWeights_nzGeology --no-apply-vs30-sample-weights &

# # NZ - With sample weights
# python model_cmds.py cv-train-catboost configs/run_configs/catboost_nz.yaml datasets/nz_dataset.parquet --id-suffix cv25_vs30SampleWeights_nzGeology > /dev/null &


# Create NZ NZTM input grid
python data_cmds.py create-nz-nztm-input-grid 100 100 /Users/claudy/dev/work/data/vs30/grids/nz_input_grid_100m nz_geology_category nz_distance_to_coast nz_nlm_groundwater_depth nz_nwt_groundwater_depth nzenvds_distance_rivers nzenvds_distance_rivers_vertical nzenvds_precip_ann nzenvds_slope_deg nzenvds_soil_acid_p nzenvds_soil_age nzenvds_soil_drainage nzenvds_soil_induration nzenvds_soil_particle_size nzenvds_topo_geomorphons nzenvds_topo_normalised_height nzenvds_topo_position nzenvds_topo_roughness nzenvds_topo_ruggedness nzenvds_topo_valley_depth nzenvds_topo_wetness --n-procs 16 ; csnotify "Grid generation complete"
