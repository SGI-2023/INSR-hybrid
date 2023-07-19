python main.py fluid \
    --tag fluid2d_tlgnM \
    --init_cond taylorgreen_multi \
    --num_hidden_layers 0 \
    --hidden_features 64 \
    --config_tcn_path configs_hash/config_hash.json \
    --network hashgrid \
    --nonlinearity relu \
    --lr 0.01 \
    -sr 128 \
    -vr 32 \
    --dt 0.05 \
    -T 100 \
    -g 0