#!/usr/bin/env bash


model='ctm250'
epochs=250
device=2
epochsstep=10

for dataset in wikipedia ccnews mlsum; do
  for topics in 100 300; do
    for seed in {0..9}; do
      echo paper_experiments/stats/log-$model-topics$topics-$dataset-seed$seed.txt
      python paper_experiments/other_models/train_ctm.py \
        --dataset $dataset \
        --num-epochs $epochs \
        --epochs-step $epochsstep \
        --num-topics $topics \
        --seed $seed \
        --device $device \
        > paper_experiments/stats/log-$model-topics$topics-$dataset-seed$seed-epochs$epochs.txt
    done
  done
done