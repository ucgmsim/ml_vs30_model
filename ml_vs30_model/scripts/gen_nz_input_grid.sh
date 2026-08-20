#!/usr/bin/env zsh

function csnotify {
   curl -d $1 ntfy.sh/W7T2QKNDH9Z4E3VJPRY8XACUL
}


# Create NZ NZTM input grid (250m)
python data_cmds.py create-nz-nztm-input-grid 250 250 /home/claudy/dev/work/data/vs30/grids/nz_input_grid_250m nz_geology_age_min nz_geology_age_max nz_geology_age_mid nz_geology_age_ln_mid nzenvds_topo_roughness nz_geology_category nzenvds_distance_rivers_vertical nz_distance_to_coast nz_nlm_groundwater_depth nz_nwt_groundwater_depth depth_to_groundwater nz_combined_groundwater_depth nzenvds_topo_normalised_height mainrock_proxy subrock_min_proxy subrock_mean_proxy subrock_median_proxy subrock_max_proxy --n-procs 4 ; csnotify "Grid generation complete"

# # Create NZ NZTM input grid (100m) 
# python data_cmds.py create-nz-nztm-input-grid 100 100 /home/claudy/dev/work/data/vs30/grids/nz_input_grid_100m nz_geology_age_min nz_geology_age_max nz_geology_age_mid nz_geology_age_ln_mid nzenvds_topo_roughness nz_geology_category nzenvds_distance_rivers_vertical nz_nlm_groundwater_depth nz_nwt_groundwater_depth depth_to_groundwater nz_combined_groundwater_depth nzenvds_topo_normalised_height mainrock_proxy subrock_min_proxy subrock_mean_proxy subrock_median_proxy subrock_max_proxy --n-procs 4 ; csnotify "Grid generation complete"


