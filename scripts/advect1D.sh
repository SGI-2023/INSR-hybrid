for i_omega0 in 0.1 0.5 5 20 40 60
do
    for i_n_width in 16 32 64
    do 

        for i_learning_rate in 0.000003 0.00001 0.00003 0.0001 0.0003 0.001 0.003
        do

            for i_hidden_layers in 2 3 4 5
            do
                python main.py advection \
                    --tag advect1D_siren_omega0_${i_omega0}_lr_${i_learning_rate}_hidden_${i_hidden_layers}_features_${i_n_width}\
                    --init_cond example1 \
                    --num_hidden_layers ${i_hidden_layers} \
                    --hidden_features ${i_n_width} \
                    --early_stop_accum_step 20400 \
                    --network siren \
                    -sr 5000 \
                    --max_n_iters 10000 \
                    --dt 0.05 \
                    -T 0 \
                    --lr ${i_learning_rate}\
                    -g 0 \
                    --omega_0 ${i_omega0} 

            done
        done

    done
done
