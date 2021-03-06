# Section 1: Modules

## Section 1.1: stats_v3.py

**Statistics functions that can be used independent of this project**

1. ks_uniformity_test (provided by Andrey Kravtsov)
2. conf_interval (provided by Andrey Kravtsov)
3. sample_spherical_angle
4. sample_spherical_pos
5. get_D_rms
6. get_R_med
7. get_D_sph
8. get_D_sph_flipped
9. get_D_sph_flipped_from_angles (only for speeding up calculation)
10. random_choice_noreplace

## Section 1.2: helper_functions_v3.py

**Helper functions that are specific for this project**

1. elvis_name_template (String template, not a **function**)
2. caterpillar_name_template (String template, not a **function**)
3. to_spherical (get phi, theta for aitoff projections)
4. to_degree (from rad to degrees)
5. normalize (to normalize vectors)
6. read_MW
7. get_MW (get D_rms/R_med/D_sph for MW)
8. read_halo (to read data for each host)
9. extract_inside (to extract subhalo inside R_vir or 300 kpc)
10. read_specific (to read brightest/heaviest subhalos)
11. generate_distribution (to generate sample of random 11 subhalos with/without surv_probs)
12. generate_brightest_distribution_with_surv_probs (to generate sample of brightest 11 subhalos with surv_probs)
    
## Section 1.3: plot_functions_v3.py

**Plot functions for this project**

1. plot_2d_dist (provided by Andrey Kravtsov)
2. plot_vectors (plot vectors in Aitoff projections)
3. plot3D (plot 3D vectors)
4. plot_distribution_D_rms_dispersion
5. plot_poles_brightest
6. plot_poles_brightest_with_config
7. plot_D_sph_vs_k_brightest
8. plot_distribution_D_sph_dispersion
9. plot_hist_D_rms_over_R_med_vs_D_sph
10. plot_hist_D_rms_vs_D_sph
11. plot_general (scatter any 2 supported quantities with/without surv_probs)
    1.  R_med
    2.  D_rms
    3.  scaled_D_rms (which is D_rms/R_med)
    4.  D_sph_k for k=3,4,...,11

# Section 2: Python Notebooks and Scripts

## Section 2.1: Notebooks

1. test_notebook_v3.ipynb (generate all plots for a host halo)
   - set suite_name, saveimage, and executed_as_300kpc
2. test_notebook_general_v3.ipynb (generate all general plots)
   - set saveimage

## Section 2.2: Scripts

1. run_test_notebook_v3.py (run test_notebook_v3.ipynb for all hosts)
2. generate_distribution_v3.py 
   1. generate distribution of 11 subhalos accounting for surv_probs: IDs, D_rms, R_med, D_sph_k for k=3,...,11 
   2. D_sph_flipped is not currently supported
3. generate_brightest_distribution_v3.py
   1. generate distribution of 11 brightest subhalos accounting for surv_probs: D_rms, R_med, D_sph_k for k=3,...,11 
   2. D_sph_flipped is not currently supported
4. run_generate_all_v3.py (run generate_distribution_v3.py and generate_brightest_distribution_v3.py)

# Section 3: Configuration (in config.py)

Configuration file `config.py` defines configuration parameters for the functions in this package. It reads key input parameters from the file `config.ini`
which should contain the following lines. 

```python
raw_dir = put your path here to data from the ELVIS and Caterpillar suites
gendata_dir = put your path here for directory where data generated by this code will be saved
result_dir = path to directory where results will be 
MW_path = put path to the MW data from Pawlowsli and Kroupa 2020 here

[raw_name]
elvis_isolated_raw_name = elvis_isolated
caterpillar_raw_name = caterpillar_zrei8_5_fix

[cutoff]
MASS_CUTOFF = 5e8

[gen_paras]
ITERATIONS = 250000
ITERATIONS_BRIGHTEST = 50000
CHUNK_SIZE = 200

[gen_configs]
generate_without_surv_probs = True
generate_with_surv_probs = True
generate_brightest = True
```

## Section 3.1: Quantities that should be changed
1. raw_dir: path to the directory that containing the folders containing elvis_isolated and caterpillar raw data. Will make the directory if not already exists.
2. gendata_dir: path to the directory that containing the generated data. Will make the directory if not already exists.
3. MW_path: path to the file containing MW data
4. elvis_isolated_raw_name: name of the folder containing elvis_isolated raw data
5. caterpillar_raw_name: name of the folder containing caterpillar raw data
6. gendata_name_template: template name of folder containing generated data of random 11 subhalos with/without surv_probs with/without selected_by_Rvir
7. gendata_brightest_name_template: template name of folder containing generated data of brightest 11 subhalos with surv_probs with/without selected_by_Rvir
8. get_caterpillar_names: function to get the names of all caterpillar host
9. get_elvis_isolated_names: function to get the names of all elvis_isolated host

## Section 3.2: Quantities that can be changed, but shouldn't
1. MASS_CUTOFF (currently set to 5e8 M_sun): only select subhalos with M_peak above this mass cutoff
2. ITERATIONS (currently set to 250000): number of sample of random 11 subhalos
3. CHUNK_SIZE (currently set to 200): vectorization factor (experimentally optimized for run time)
4. ITERATIONS_BRIGHTEST (currently set to 50000): number of sample of brightest 11 subhalos
5. generate_without_surv_probs (currently set to True): generate sample of random 11 subhalos without surv_probs if True
6. generate_with_surv_probs (currently set to True): generate sample of random 11 subhalos with surv_probs if True
7. generate_brightest (currently set to True): generate sample of brightest 11 subhalos with surv_probs if True
8. get_suite_names: function to get the names of all suites

# Section 4: How to run
These are the steps to use this code
1. Configure config.py as noted in Section 3
2. Run run_generate_all_v3.py
3. Run run_test_notebook_v3.py
4. Go to test_notebook_general_v3.ipynb, set saveimage, X_type, Y_type and run the notebook to to generate general plots
