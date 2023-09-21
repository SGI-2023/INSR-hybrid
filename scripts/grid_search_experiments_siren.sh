for i_omega0 in 30 
do
    for i_n_width in  256
    do 
        for i_hidden_layers in   6
        do
            for i_learning_rate in 0.0001
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
                    -T 30 \
                    --lr ${i_learning_rate}\
                    -g 0 \
                    --omega_0 ${i_omega0} 
            done
        done
    done
done