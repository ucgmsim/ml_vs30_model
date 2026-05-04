## Data

Everything is with respect to the environment variable `VS30_MODEL_BASE_DATA_DIR`, accessible via `constants.BASE_DATA_DIR` in the code.

### Data folder structure

- `cleaned_vs30_data` - Contains cleaned Vs30 datasets
- `datasets` - Contains ready to use NZ & US datasets, generated based on the data configs
- `grids` - NZ wide input data grids
- `input_data` - Input variable datasets
    - `foster_geological_category` - Foster geology category classifications (shape files)
    - `nz_coastline` - NZ coastline geo dataframe
    - `nz_geology` - NZ geology dataset from GNS (2023)
    - `nz_nlm` - NZ national liquefaction model
    - `nz_nwt` - NZ national water table
    - `nzenvds_v1p1_nztm` - Data from the [NZEnvDS dataset](https://datastore.landcareresearch.co.nz/dataset/nzenvds)
    - `raw` - Larger raw datasets (generally global)
        - `absolute_depth_to_bedrock` - Global absolute depth to bedrock
        - `compound_topographic_index` - Global compound topographic index
        - `depth_to_groundwater` - Global depth to groundwater
        - `geom_1KMent_GMTEDmd.tif` - Global landform entropy
        - `geom_1KMsha_GMTEDmd.tif` - Global landform Shannon index
        - `geom_1KMuni_GMTEDmd.tif` - Global landform uniformity
        - `geomorphon` - Global geomorphon
        - `profile_curvature` - Global profile curvature
        - `roughness` - Global roughness
        - `tangential_curvature` - Global tangential curvature
        - `terrain_ruggedness_index` - Global terrain ruggedness index
        - `topographic_position_index` - Global topographic position index 
        - `topographic_slope` - Global topographic slope
        - `us_sgmc` - Geological Map for the US
        - `vector_ruggedness_measure` - Global vector ruggedness measure
