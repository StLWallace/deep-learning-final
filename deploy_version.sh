MODEL_DIR="gs://slalom-stl-kaggle-datasets/fake-comments/results/model_1/"
VERSION_NAME="V1"
MODEL_NAME="sw_final_1"
FRAMEWORK="TENSORFLOW"
REGION="us-central1"

# gcloud components install beta

gcloud beta ai-platform versions create $VERSION_NAME \
  --model=$MODEL_NAME \
  --origin=$MODEL_DIR \
  --runtime-version=2.3 \
  --framework=$FRAMEWORK \
  --python-version=3.7 \
  --region=$REGION
  --machine-type=mls1-c1-m2