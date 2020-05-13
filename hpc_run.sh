python3 hpc_run.py \
        --model_name 6_region \
        --ts_data_left_index 2017-01 \
        --ts_data_right_index 2017-01 \
        --run_mode plan \
        --baseload_integer False \
        --baseload_ramping False \
        --ts_subsampling None \
        --num_iterations 1 \
        --logging_level INFO

python3 hpc_run.py \
        --model_name 6_region \
        --ts_data_left_index 2017-01 \
        --ts_data_right_index 2017-12 \
        --run_mode plan \
        --baseload_integer False \
        --baseload_ramping False \
        --ts_subsampling clustering \
        --num_days_subsample 48 \
        --subsample_blocks days \
        --num_iterations 2 \
        --logging_level INFO

python3 hpc_run.py \
        --model_name 6_region \
        --ts_data_left_index 2017-01 \
        --ts_data_right_index 2017-12 \
        --run_mode plan \
        --baseload_integer False \
        --baseload_ramping False \
        --ts_subsampling importance \
        --num_days_subsample 48 \
        --num_days_high 16 \
        --subsample_blocks days \
        --num_iterations 2 \
        --logging_level INFO
