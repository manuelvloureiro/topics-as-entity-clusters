#!/usr/bin/env bash


model='kmeansfaiss'
device=0

for topics in 100 300; do
  for dataset in ccnews; do
    for seed in {0..5}; do
      for type in wikidata wikipedia; do
        echo paper_experiments/stats/log-$model-topics$topics-$dataset-seed$seed-$type.txt
        python paper_experiments/train_kmeansfaiss.py \
          --dataset $dataset \
          --embedding-type $type \
          --num-topics $topics \
          --seed $seed \
          --device $device \
          > paper_experiments/stats/log-$model-topics$topics-$dataset-seed$seed-$type.txt
      done
      for alpha in 1 2 .5 5 .2; do
        echo paper_experiments/stats/log-$model-topics$topics-$dataset-seed$seed-both-$alpha.txt
        python paper_experiments/train_kmeansfaiss.py \
          --dataset $dataset \
          --embedding-type both \
          --alpha $alpha \
          --num-topics $topics \
          --seed $seed \
          --device $device \
          > paper_experiments/stats/log-$model-topics$topics-$dataset-seed$seed-both-$alpha.txt
      done
    done
  done
done