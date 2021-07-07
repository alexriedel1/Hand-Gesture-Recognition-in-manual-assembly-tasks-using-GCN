# Hand Gesture Recognition in manual assembly tasks using GCN

This is the source code to the paper "Hand Gesture Recognition of Methods-Time Measurement-1 motions in manual assembly tasks using Graph Convolutional Networks"
Make sure to have all the requirements installed via `pip install -r requirements.txt`. For working with all the models covered in the paper, install [Pytorch Geometric Temporal](https://pytorch-geometric-temporal.readthedocs.io/en/latest/notes/installation.html).

# Data

In this repository, the skeletal data is already extracted from the video data via MediaPipe Hands in /data/graph_data. For demo pruposes, run video2landmark.py that creates key points and their corresponding label from the video and annotation file in data/video_example. 

# Training & Testing

Training the model can is done via train.py and training parameters (models, lr, bs, additional joints, etc.) are defined in config.py. The training and validation split is defined in data/get_data_from_csv.py.

To get the final results using 2s-AGCN from the paper, run eval.py (either with the "Release" class or without, as defined in config.py `no_release`). To inference on an example video including visualization, run inference.py. 


# Contact

For any questions leave a comment or contact me at alexander.riedel@eah-jena.de
