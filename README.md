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

# Section 2: Python Notebooks and Scripts

## Section 2.1: Notebooks

1. test_notebook_v3.ipynb (generate all plots for a host halo)
2. test_notebook_general_v3.ipynb (generate all general plots)

## Section 2.2: Scripts

1. run_test_notebook_v3.py (run test_notebook_v3.ipynb for all hosts)
   1. set caterpillar_dir and elvis_dir
2. generate_distribution_v3.py 
   1. generate distribution of 11 subhalos accounting for surv_probs: IDs, D_rms, R_med, D_sph_k for k=3,...,11 
   2. D_sph_flipped is not currently supported
   3. prev_suite_dir: directory to the raw data
   4. data_dir: directory to the generated data
3. generate_brightest_distribution_v3.py
   1. generate distribution of 11 brightest subhalos accounting for surv_probs: D_rms, R_med, D_sph_k for k=3,...,11 
   2. D_sph_flipped is not currently supported
   3. set prev_suite_dir: directory to the raw data
   4. set data_dir: directory to the generated data
4. run_generate_all_v3.py (run generate_distribution_v3.py or generate_brightest_distribution_v3.py)