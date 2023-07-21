python main.py advection \
    --tag advect1D_ex1_match_norm \
    --init_cond example1 \
    --num_hidden_layers 2 \
    --hidden_features 20 \
    --early_stop_accum_step 2000 \
    -sr 5000 \
    --dt 0.05 \
    -T 20 \
    -g 0