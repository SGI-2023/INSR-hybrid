for N_harmonics in 1 3 9 27
do
    for i_omega0 in 0.1 5 30 40 60
    do
        for i_n_width in 16 32 64 128
        do 
            for i_hidden_layers in 2 3 4 5
            do
                for i_learning_rate in 0.00001 0.00003 0.0001 0.0003 0.001
                do
                    python main.py advection \
                        --tag advect1D_positional_harmonics_${N_harmonics}_omega0_${i_omega0}_width_${i_n_width}_hidden_${i_hidden_layers}_lr_${i_learning_rate}\
                        --init_cond example1 \
                        --num_hidden_layers ${i_hidden_layers} \
                        --hidden_features ${i_n_width} \
                        --network positional \
                        -sr 5000 \
                        --n_harmonic_functions ${N_harmonics} \
                        --max_n_iters 20000 \
                        --dt 0.05 \
                        -T 0 \
                        --lr ${i_learning_rate}\
                        -g 0 \
                        --omega_0 ${i_omega0} 

                done
            done
        done
    done
done
