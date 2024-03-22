#!/usr/bin/env bash


package='spark.driver.extraClassPath="spark-packages/spark-xml_2.12-0.14.0.jar" spark.executor.extraClassPath="spark-xml_2.12-0.14.0.jar"'
jars='paper_experiments/other_models/WikiPDA/spark-packages/spark-xml_2.12-0.14.0.jar'

unset PYSPARK_DRIVER_PYTHON

model='wikipda'

for topics in 100 300; do
  for seed in {0..9}; do
    echo paper_experiments/stats/log-$model-topics$topics-seed$seed.txt
    spark-submit \
      --deploy-mode client \
      --driver-cores 1 \
      --driver-memory 900G \
      --executor-memory 50G \
      --jars $jars \
      paper_experiments/other_models/WikiPDA/EnrichedLinks_TrainLDA.py \
      --k $topics \
      --seed $seed
      > paper_experiments/stats/log-$model-topics$topics-seed$seed.txt
    spark-submit \
      --deploy-mode client \
      --driver-cores 1 \
      --driver-memory 100G \
      --executor-memory 6G \
      --jars $jars \
      paper_experiments/other_models/WikiPDA/ComputeCoherence.py \
      --k $topics \
      --seed $seed
      >> paper_experiments/stats/log-$model-topics$topics-seed$seed.txt
  done
done
