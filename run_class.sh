#!/bin/bash
## classification test with different size of training data
for tn in 100 500 1000 2000 5000 10000 20000 30000
do
    CUDA_VISIBLE_DEVICES=3 python main.py --combine False --residue True --weighted True --learning_rate 0.001 --knn 10 --batch_size 64 --aggre_layer_num 1 --aggre_type max --embed_dims 2048 2048 --mode class --train_num $tn
done
