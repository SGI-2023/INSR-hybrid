for i_omega0 in  30 20 15 10 5
do
    for i_n_width in  128
    do 
        for i_hidden_layers in   4
        do
            for i_learning_rate in 0.0003
            do
                python main.py burger \
                    --tag burger_sumfreq_omega0_${i_omega0}_width_${i_n_width}_hidden_${i_hidden_layers}_lr_${i_learning_rate}\
                    --init_cond example1 \
                    --num_hidden_layers ${i_hidden_layers} \
                    --hidden_features ${i_n_width} \
                    --network siren \
                    -sr 4000 \
                    --max_n_iters 20000 \
                    --dt 0.005 \
                    -T 200 \
                    --lr ${i_learning_rate} \
                    -g 0 \
                    --mu 0.03 \
                    --omega_0 ${i_omega0} 
            done
        done
    done
done