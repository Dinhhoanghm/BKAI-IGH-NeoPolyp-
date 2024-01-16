python -m neopolyp.train \
        --model unet \
        --batch_size 16 \
        --max_epochs 50 \
        --num_workers 4 \
        --lr 0.0001 \
        --split_ratio 0.96 \
        --data_path bkai-igh-neopolyp \
        -w -wk 844fc0e4bcb3ee33a64c04b9ba845966de80180e