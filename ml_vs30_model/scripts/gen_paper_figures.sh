#!/usr/bin/env zsh

# Break if any command fails
set -e

scripts_dir="/Users/claudy/dev/work/code/ml_vs30_model/ml_vs30_model/scripts"
dataset_ffp="${VS30_MODEL_BASE_DATA_DIR}/datasets/nz_site_db_dataset.parquet"

geyin_dataset_ffp="${VS30_MODEL_BASE_DATA_DIR}/datasets/us_geyin_maurer.parquet"
nz_input_dataset_ffp="${VS30_MODEL_BASE_DATA_DIR}/grids/nz_input_grid_250m/input_grid.nc"
population_density_ffp="${VS30_MODEL_BASE_DATA_DIR}/other/nz_population/new-zealand-estimated-resident-population-grid-250-metre.shp"

foster_nz_dataset="${VS30_MODEL_BASE_DATA_DIR}/datasets/foster.parquet"
foster_nz_dataset_results="${VS30_MODEL_BASE_DATA_DIR}/results/foster/foster_results_nz_site_db.parquet"
foster_tif="${VS30_MODEL_BASE_DATA_DIR}/nz_estimates/foster_original/foster_paper_original.tif"

cv_model_dir="${VS30_MODEL_BASE_DATA_DIR}/results/ind_results/0624_182722_cv100_ngboostV4p14"
full_model_dir="${VS30_MODEL_BASE_DATA_DIR}/results/ind_results/0624_183433_full_ngboostV4p14"


out_dir="/Users/claudy/dev/work/tmp/vs30_plots"

default_fig_size="6.5, 4"
default_font_size="8"
default_gmt_fig_font_label="12p,Helvetica,black"
default_gmt_fig_minor_font_label="9p,Helvetica,black"
export fig_size=$default_fig_size
export fig_format="png"
export fig_dpi="500"
export fig_font_size=$default_font_size
export fig_linewidth="2.5"
export fig_group_linewidth="1.25"
export gmt_fig_font_label=$default_gmt_fig_font_label
export gmt_fig_minor_font_label=$default_gmt_fig_minor_font_label


### ------------------- Spatial Figures ------------------------------

### Site map
# echo "Generating site map..."
# python "${scripts_dir}/gen_paper_figures.py" gen-site-map "${dataset_ffp}" "${out_dir}"

# ## Site database Vs30 histogram
# echo "Generating Vs30 histogram..."
# export fig_size="6.5, 2.75"
# python "${scripts_dir}/gen_paper_figures.py" gen-vs30-hist "${dataset_ffp}" "${out_dir}"
# export fig_size=$default_fig_size

### Input histograms
# echo "Generating input variable histograms..."
# export fig_size="3.25, 2.5"
# python "${scripts_dir}/gen_paper_figures.py" input-variable-kde-distribution "${dataset_ffp}" "${nz_input_dataset_ffp}" compound_topographic_index "${out_dir}" --no-show-legend
# python "${scripts_dir}/gen_paper_figures.py" input-variable-kde-distribution "${dataset_ffp}" "${nz_input_dataset_ffp}" nzenvds_slope_deg "${out_dir}" --no-show-legend
# python "${scripts_dir}/gen_paper_figures.py" input-variable-kde-distribution "${dataset_ffp}" "${nz_input_dataset_ffp}" nz_combined_groundwater_depth "${out_dir}" --no-show-legend
# python "${scripts_dir}/gen_paper_figures.py" input-variable-kde-distribution "${dataset_ffp}" "${nz_input_dataset_ffp}" nz_geology_age_mid "${out_dir}"
# export fig_size=$default_fig_size

# ### Combined dataset comparison
# echo "Generating combined dataset comparison plots..."
# export fig_size="6.5, 6"
# python "${scripts_dir}/gen_paper_figures.py" combined-dataset-comparison "${dataset_ffp}" "${foster_nz_dataset}" "${geyin_dataset_ffp}" "${out_dir}"
# export fig_size=$default_fig_size

# ### Residual scatter plot
# echo "Generating residual scatter plot..."
# export fig_size="6.5, 2.5"
# python "${scripts_dir}/gen_paper_figures.py" gen-residual-scatter-plot "${cv_model_dir}/val_results.parquet" "${out_dir}" --full-model-dir "${full_model_dir}" --hide-x-label
# export fig_size="6.5, 2.75"
# python "${scripts_dir}/gen_paper_figures.py" gen-residual-scatter-plot "${foster_nz_dataset_results}" "${out_dir}" --is-foster
# export fig_size=$default_fig_size

## One-to-one plot
# export fig_size="3.5, 3.5"
# python "${scripts_dir}/gen_paper_figures.py" gen-one-to-one-plot "${cv_model_dir}/val_results.parquet" "${out_dir}" --full-model-dir "${full_model_dir}"
# python "${scripts_dir}/gen_paper_figures.py" gen-one-to-one-plot "${foster_nz_dataset_results}" "${out_dir}" --is-foster --no-show-legend
# export fig_size=$default_fig_size

# ## Global feature importance
# echo "Generating global feature importance plot..."
# export fig_font_size="3"
# python "${scripts_dir}/gen_paper_figures.py" gen-global-feature-importance "${cv_model_dir}" "${out_dir}"
# export fig_font_size=$default_font_size

# ### Standardized residuals CDF
# export fig_size="3.25, 2.5"
# echo "Generating standardized residuals CDF plot..."
# python "${scripts_dir}/gen_paper_figures.py" gen-std-res-cdf-plot "${cv_model_dir}/val_results.parquet" "${out_dir}" 
# export fig_size=$default_fig_size

# ### Predicted standard deviation vs Vs30 plot
# export fig_size="3.25, 2.5"
# echo "Generating predicted standard deviation vs Vs30 plot..."
# python "${scripts_dir}/gen_paper_figures.py" predicted-std-vs30 "${cv_model_dir}" "${out_dir}" 
# export fig_size=$default_fig_size

# ### Global feature importance
# echo "Generating global feature importance plot..."
# export fig_size="6.5, 2.5"
# python "${scripts_dir}/gen_paper_figures.py" gen-global-feature-importance "${cv_model_dir}" "${out_dir}"
# export fig_size=$default_fig_size

## Feature trend plots
echo "Generating feature trend plots..."
export fig_size="3.25, 2.75"
python "${scripts_dir}/gen_paper_figures.py" gen-feature-trend-plots "${cv_model_dir}" "${out_dir}" nzenvds_topo_roughness nz_geology_age_ln_mid nz_combined_groundwater_depth nzenvds_topo_normalised_height
export fig_size=$default_fig_size

# ## Vs30 map
# echo "Generating Vs30 map..."
# python "${scripts_dir}/gen_paper_figures.py" gen-vs30-map "${full_model_dir}" "${out_dir}" "ni" --town Taupo --town "Palmerston North" --town Napier --region Taranaki --region "East Cape" --label "a) North Island: ML Model (This Study)" --no-show-colorbar
# python "${scripts_dir}/gen_paper_figures.py" gen-vs30-map "${full_model_dir}" "${out_dir}" "si" --town Greymouth --town Haast --town Nelson --town Blenheim --label "c) South Island: ML Model (This Study)"

# ### Vs30 Subregion map
# echo "Generating Vs30 subregion map..."
# export gmt_fig_font_label="10p,Helvetica-Bold,black"
# export gmt_fig_minor_font_label="6p,Helvetica,black"
# python "${scripts_dir}/gen_paper_figures.py" gen-vs30-map "${full_model_dir}" "${out_dir}" "wellington" --grid-spacing "25e/25e" --show-towns --show-highways --plot-sites
# python "${scripts_dir}/gen_paper_figures.py" gen-vs30-map "${full_model_dir}" "${out_dir}" "wellington" --grid-spacing "25e/25e" --show-towns --show-highways --plot-kriged --plot-sites --label "a) ML Model (This Study)" --no-show-colorbar
# python "${scripts_dir}/gen_paper_figures.py" gen-vs30-map "${full_model_dir}" "${out_dir}" "wellington" --grid-spacing "25e/25e" --show-towns --show-highways --plot-foster --plot-sites --label "b) Foster et al. (2019)" --no-show-colorbar
# python "${scripts_dir}/gen_paper_figures.py" gen-vs30-map "${full_model_dir}" "${out_dir}" "canterbury_narrow" --grid-spacing "25e/25e" --show-towns --show-highways --plot-sites 
# python "${scripts_dir}/gen_paper_figures.py" gen-vs30-map "${full_model_dir}" "${out_dir}" "canterbury_narrow" --grid-spacing "25e/25e" --show-towns --show-highways --plot-kriged --plot-sites --label "d) ML Model (This Study)"
# python "${scripts_dir}/gen_paper_figures.py" gen-vs30-map "${full_model_dir}" "${out_dir}" "canterbury_narrow" --grid-spacing "25e/25e" --show-towns --show-highways --plot-foster --plot-sites --label "e) Foster et al. (2019)"
# python "${scripts_dir}/gen_paper_figures.py" gen-vs30-map "${full_model_dir}" "${out_dir}" "cromwell_region" --grid-spacing "25e/25e" --show-towns --show-highways --plot-sites
# python "${scripts_dir}/gen_paper_figures.py" gen-vs30-map "${full_model_dir}" "${out_dir}" "cromwell_region" --grid-spacing "25e/25e" --show-towns --show-highways --plot-kriged --plot-sites
# python "${scripts_dir}/gen_paper_figures.py" gen-vs30-map "${full_model_dir}" "${out_dir}" "cromwell_region" --grid-spacing "25e/25e" --show-towns --show-highways --plot-foster --plot-sites
# python "${scripts_dir}/gen_paper_figures.py" gen-vs30-map "${full_model_dir}" "${out_dir}" "auckland_hamilton_region" --grid-spacing "25e/25e" --show-towns --show-highways --plot-sites
# python "${scripts_dir}/gen_paper_figures.py" gen-vs30-map "${full_model_dir}" "${out_dir}" "auckland_hamilton_region" --grid-spacing "25e/25e" --show-towns --show-highways --plot-kriged --plot-sites
# python "${scripts_dir}/gen_paper_figures.py" gen-vs30-map "${full_model_dir}" "${out_dir}" "auckland_hamilton_region" --grid-spacing "25e/25e" --show-towns --show-highways --plot-foster --plot-sites
# python "${scripts_dir}/gen_paper_figures.py" gen-vs30-map "${full_model_dir}" "${out_dir}" "taurange_rotorua_region" --grid-spacing "25e/25e" --show-towns --show-highways --plot-sites
# python "${scripts_dir}/gen_paper_figures.py" gen-vs30-map "${full_model_dir}" "${out_dir}" "taurange_rotorua_region" --grid-spacing "25e/25e" --show-towns --show-highways --plot-kriged --plot-sites
# python "${scripts_dir}/gen_paper_figures.py" gen-vs30-map "${full_model_dir}" "${out_dir}" "taurange_rotorua_region" --grid-spacing "25e/25e" --show-towns --show-highways --plot-foster --plot-sites
# export gmt_fig_font_label=$default_gmt_fig_font_label
# export gmt_fig_minor_font_label=$default_gmt_fig_minor_font_label

# ## Residual map
# echo "Generating residual map..."
# python "${scripts_dir}/gen_paper_figures.py" gen-residual-map "${full_model_dir}/nz_vs30_results.nc" "${out_dir}" "ni" --use-kriged --label "b) North Island: ln(Foster) - ln(ML)" --no-show-colorbar
# python "${scripts_dir}/gen_paper_figures.py" gen-residual-map "${full_model_dir}/nz_vs30_results.nc" "${out_dir}" "si" --use-kriged --label "d) South Island: ln(Foster) - ln(ML)"

# ### Residual Subregion map
# echo "Generating residual subregion map..."
# export gmt_fig_font_label="10p,Helvetica-Bold,black"
# export gmt_fig_minor_font_label="6p,Helvetica,black"
# python "${scripts_dir}/gen_paper_figures.py" gen-residual-map "${full_model_dir}/nz_vs30_results.nc" "${out_dir}" "wellington" --grid-spacing "25e/25e" --show-towns --show-highways --use-kriged --label "c) ln(Foster) - ln(ML)" --no-show-colorbar
# python "${scripts_dir}/gen_paper_figures.py" gen-residual-map "${full_model_dir}/nz_vs30_results.nc" "${out_dir}" "canterbury_narrow" --grid-spacing "25e/25e" --show-towns --show-highways --use-kriged --label "f) ln(Foster) - ln(ML)"
# python "${scripts_dir}/gen_paper_figures.py" gen-residual-map "${full_model_dir}/nz_vs30_results.nc" "${out_dir}" "cromwell_region" --grid-spacing "25e/25e" --show-towns --show-highways --use-kriged
# python "${scripts_dir}/gen_paper_figures.py" gen-residual-map "${full_model_dir}/nz_vs30_results.nc" "${out_dir}" "auckland_hamilton_region" --grid-spacing "25e/25e" --show-towns --show-highways --use-kriged
# python "${scripts_dir}/gen_paper_figures.py" gen-residual-map "${full_model_dir}/nz_vs30_results.nc" "${out_dir}" "taurange_rotorua_region" --grid-spacing "25e/25e" --show-towns --show-highways --use-kriged
# export gmt_fig_font_label=$default_gmt_fig_font_label
# export gmt_fig_minor_font_label=$default_gmt_fig_minor_font_label

# ### NZ Vs30 Histogram
# echo "Generating NZ Vs30 histogram..."
# export fig_size="6.5, 2.75"
# python "${scripts_dir}/gen_paper_figures.py" create-nz-vs30-histogram "${full_model_dir}" "${foster_tif}" "${population_density_ffp}" "${out_dir}"
# export fig_size=$default_fig_size


### ------------------- Electronic Supplement - Spatial Figures -------------------

# # Input variables
# variables=(
#   "nz_geology_age_ln_mid"
#   "nzenvds_topo_roughness"
#   "nzenvds_distance_rivers_vertical"
#   "nz_combined_groundwater_depth"
#   "nzenvds_topo_normalised_height"
# )

# ### Input variable maps & distributions
# export fig_size="6.5, 2.75"
# for var in "${variables[@]}"; do
#     echo "Generating input variable map for ${var}..."
#     python "${scripts_dir}/gen_paper_figures.py" input-variable-map "${nz_input_dataset_ffp}" "${var}" nz_full "${out_dir}" 
# done
# export fig_size=$default_fig_size


### Predicted standard deviation
# python "${scripts_dir}/gen_paper_figures.py" gen-pred-std-map "${full_model_dir}" "${out_dir}" "nz_full"

