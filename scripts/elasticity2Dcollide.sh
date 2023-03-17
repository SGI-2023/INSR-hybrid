python main.py elasticity \
    --tag icml_elas2Dcol_extra \
    --num_hidden_layers 3 \
    --hidden_features 68 \
    -sr 100 \
    -vr 100 \
    -T 10 \
    -g 3 \
    --dt 0.1 \
    --max_n_iter 20000 \
    --lr 1e-5 \
    --dim 2 \
    --energy 'arap' 'kinematics' 'collision_sphere' 'external' 'volume' \
    --ratio_volume 1e3 \
    --ratio_arap 2e1 \
    --ratio_collide 1e4  \
    --ratio_kinematics 1e1 \
    -f_ext_x 0 \
    -f_ext_y ' -2e2' \
    -T_ext 2 \
    -we
    # --early_stop \


python main.py elasticity \
    --tag icml_elas2Dcol_base \
    --num_hidden_layers 3 \
    --hidden_features 68 \
    -sr 100 \
    -vr 100 \
    -T 10 \
    -g 3 \
    --dt 0.1 \
    --max_n_iter 20000 \
    --lr 1e-5 \
    --dim 2 \
    --energy 'arap' 'kinematics' 'collision_sphere' 'external' 'volume' \
    --ratio_volume 1e3 \
    --ratio_arap 2e1 \
    --ratio_collide 1e4  \
    --ratio_kinematics 1e1 \
    -f_ext_x 0 \
    -f_ext_y ' -2e2' \
    -T_ext 2
    # --early_stop \