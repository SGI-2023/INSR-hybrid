for i_omega0 in 5 30 
do
    for i_n_width in  128 256
    do 
        for i_hidden_layers in   5 6
        do
            for i_learning_rate in 0.00001 0.0001
            do
                for i_rel_w in 10 0.1 0.01
                do
                    python main.py advection \
                        --tag gridsearch_test_advect1D_siren_omega0_${i_omega0}_width_${i_n_width}_hidden_${i_hidden_layers}_lr_${i_learning_rate}_relw_${i_rel_w}\
                        --init_cond example1 \
                        --num_hidden_layers ${i_hidden_layers} \
                        --hidden_features ${i_n_width} \
                        --network siren \
                        -sr 5000 \
                        --max_n_iters 20000 \
                        --dt 0.05 \
                        -T 0 \
                        --lr ${i_learning_rate}\
                        -g 0 \
                        --omega_0 ${i_omega0} \
                        --rel_w ${i_rel_w}
                done
            done
        done
    done
done