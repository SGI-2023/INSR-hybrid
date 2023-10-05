
python main.py advection \
                    --tag advect1D_siren_omega0_5_lr_0.00001_hidden_3_features_128_harmonics_3\
                    --init_cond example1 \
                    --num_hidden_layers 3 \
                    --hidden_features 128 \
                    --early_stop_accum_step 20400 \
                    --network siren \
                    -sr 5000 \
                    --max_n_iters 10000 \
                    --dt 0.05 \
                    -T 0 \
                    --lr 0.00001\
                    -g 0 \
                    --omega_0 5
