for i_omega0 in 0.1 0.5 1 5 10 20
do
    for i_n_harmonic_functions in 1 10 20 30 40 50
    do 

        for i_learning_rate in 0.00001 0.00003 0.0001 0.0003 0.001 0.003 0.01 0.03
        do

            for i_hidden_layers in 2 3 4 5
            do
                python main.py advection \
                    --tag advect1D_omega0_${i_omega0}n_harmonics_${i_n_harmonic_functions}_lr_${i_learning_rate}_hidden_${i_hidden_layers}\
                    --init_cond example1 \
                    --num_hidden_layers ${i_hidden_layers} \
                    --hidden_features 32 \
                    --early_stop_accum_step 20400 \
                    --network positional \
                    -sr 5000 \
                    --max_n_iters 8000 \
                    --dt 0.05 \
                    -T 0 \
                    --lr ${i_learning_rate}\
                    -g 0 \
                    --omega0 ${i_omega0}  \
                    --n_harmonic_functions ${i_n_harmonic_functions}


            done
        done

    done
done
