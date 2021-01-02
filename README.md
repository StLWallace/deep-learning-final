# deep-learning-final
This repo is used to train a LSTM model on the Kaggle Fake News dataset. The package is setup to be used with GCP's AI Platform for training. Other contributors to the project are Sahil Rastogi and Andrew Lin.

## run_training.sh
This is run via bash to submit the training job on GCP AI Platform. To run, change the parameters to your Cloud Storage locations.

## src/trainer/task.py
This is the main Python module that contains the dataset preparation and model training logic.
