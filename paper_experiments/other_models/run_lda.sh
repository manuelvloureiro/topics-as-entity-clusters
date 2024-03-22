#!/usr/bin/env bash


model='lda'
epochs=1000

for dataset in wikipedia ccnews mlsum; do
  for topics in 100 300; do
    for seed in {0..9}; do
      echo paper_experiments/stats/log-$model-topics$topics-$dataset-seed$seed-epochs$epochs.txt
      python paper_experiments/other_models/train_lda_mallet.py \
        --dataset $dataset \
        --num-epochs $epochs \
        --num-topics $topics \
        --seed $seed \
        > paper_experiments/stats/log-$model-topics$topics-$dataset-seed$seed-epochs$epochs.txt
    done
  done
done