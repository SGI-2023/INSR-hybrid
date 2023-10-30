for N_harmonics in 3
do
    for i_omega0 in 30
    do
        for i_n_width in  64
        do
            for i_hidden_layers in 3
            do
                for i_learning_rate in 0.0001
                do
                    python main.py advection \
                        --tag advect1D_complexIC_simpleNN_gradCT_PE\
                        --init_cond example2 \
                        --num_hidden_layers ${i_hidden_layers} \
                        --hidden_features ${i_n_width} \
                        --network positional \
                        -sr 5000 \
                        --nonlinearity softplus \
                        --n_harmonic_functions ${N_harmonics} \
                        --max_n_iters 20000 \
                        --early_stop_accum_step 1000 \
                        --dt 0.05 \
                        -T 15 \
                        --lr ${i_learning_rate}\
                        -g 0 \
                        --omega_0 ${i_omega0}
                done
            done
        done
    done
done