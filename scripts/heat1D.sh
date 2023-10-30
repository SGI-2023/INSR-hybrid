for i_omega0 in 10
do
    for i_n_width in  256
    do
        for i_hidden_layers in  3
        do
            for i_learning_rate in 0.001
            do
                python main.py diffusion \
                    --tag heat_lap2\
                    --init_cond example1_lap \
                    --num_hidden_layers ${i_hidden_layers} \
                    --hidden_features ${i_n_width} \
                    --network siren \
                    -sr 5000 \
                    --max_n_iters 20000 \
                    --early_stop_accum_step 2000 \
                    --dt 0.05 \
                    -k 0.005\
                    -T 10 \
                    -L 2 \
                    --lr ${i_learning_rate}\
                    -g 0 \
                    --omega_0 ${i_omega0}
            done
        done
    done
done