 
for nonlinearity_type in 'softplus' 'sin' 'relu'
do
    for N_harmonics in 1 3 
    do
        for i_omega0 in 5 30 
        do
            for i_n_width in  32  128
            do 
                for i_hidden_layers in 2  5 
                do
                    for i_learning_rate in 0.00001 0.0001  
                    do
                        python main.py advection \
                            --tag advect1D_positional_harmonicss_${N_harmonics}_nonlin_${nonlinearity_type}_${N_harmonics}_omega0_${i_omega0}_width_${i_n_width}_hidden_${i_hidden_layers}_lr_${i_learning_rate}\
                            --init_cond example1 \
                            --num_hidden_layers ${i_hidden_layers} \
                            --hidden_features ${i_n_width} \
                            --network positional \
                            -sr 5000 \
                            --nonlinearity ${nonlinearity_type} \
                            --n_harmonic_functions ${N_harmonics} \
                            --max_n_iters 20000 \
                            --dt 0.05 \
                            -T 15 \
                            --lr ${i_learning_rate}\
                            -g 0 \
                            --omega_0 ${i_omega0} 
                        
                        echo 'y'

                    done
                done
            done
        done
    done

    for i_omega0 in 5 30  
    do
        for i_n_width in  32   128
        do 
            for i_hidden_layers in   2  5 
            do
                for i_learning_rate in 0.00001 0.0001  
                do
                    python main.py advection \
                        --tag advect1D_sumfreq_omega0_${i_omega0}_width_${i_n_width}_hidden_${i_hidden_layers}_lr_${i_learning_rate}\
                        --init_cond example2 \
                        --num_hidden_layers ${i_hidden_layers} \
                        --hidden_features ${i_n_width} \
                        --network siren \
                        -sr 5000 \
                        --max_n_iters 5000 \
                        --dt 0.05 \
                        --nonlinearity ${nonlinearity_type} \
                        -T 10 \
                        --lr ${i_learning_rate}\
                        -g 0 \
                        --omega_0 ${i_omega0} 
                done
            done
        done
    done
done