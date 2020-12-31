NOW=$( date '+%Y%m%d%H%M' )
gcloud ai-platform jobs submit training sw_deep_learning_training_$NOW \
    --staging-bucket=gs://slalom-stl-kaggle-datasets \
    --job-dir=gs://slalom-stl-kaggle-datasets/fake-comments  \
    --package-path=src/trainer \
    --module-name=trainer.task \
    --region=us-central1 \
    --runtime-version='2.3' \
    --python-version='3.7' \
    --scale-tier=BASIC_GPU \
    -- \
    --glove_path=gs://slalom-stl-kaggle-datasets/fake-comments/glove.6B.100d.txt \
    --output_path=gs://slalom-stl-kaggle-datasets/fake-comments/results