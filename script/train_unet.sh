python -m neopolyp.train \
        --model unet \
        --batch_size 16 \
        --max_epochs 50 \
        --num_workers 4 \
        --lr 0.0001 \
        --split_ratio 0.96 \
        --data_path bkai-igh-neopolyp \
        -w -wk 626a895d3032d1de89568d5eb599298b39db1c48