# Hand Gesture Recognition in manual assembly tasks using GCN


This is the source code to the paper "Hand Gesture Recognition of Methods-Time Measurement-1 motions in manual assembly tasks using Graph Convolutional Networks"

Make sure to have all the requirements installed via `pip install -r requirements.txt`


# Data

The 21 x-y-z key points from the training and validation data are already provided in this repository. For demo pruposes, run video2landmark.py that creates key points and their corresponding label from the video and annotation file in data/video_example. 

# Training

Training the model can is done via train.py and training parameters (models, lr, bs, additional joints, etc.) are defined in config.py. 
