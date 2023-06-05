# Hand Gesture Recognition in manual assembly tasks using GCN
<p align="center">
<img src="/imgs/structure.PNG" width="100%" height="100%">
</p>

This is the source code to the paper ["Hand Gesture Recognition of Methods-Time Measurement-1 motions in manual assembly tasks using Graph Convolutional Networks"](https://www.tandfonline.com/doi/full/10.1080/08839514.2021.2014191).

```
@article{doi:10.1080/08839514.2021.2014191,
author = {Alexander Riedel and Nico Brehm and Tobias Pfeifroth},
title = {Hand Gesture Recognition of Methods-Time Measurement-1 Motions in Manual Assembly Tasks Using Graph Convolutional Networks},
journal = {Applied Artificial Intelligence},
volume = {0},
number = {0},
pages = {1-23},
year  = {2021},
publisher = {Taylor & Francis},
doi = {10.1080/08839514.2021.2014191},
URL = {https://doi.org/10.1080/08839514.2021.2014191}}
```


Make sure to have all the requirements installed via `pip install -r requirements.txt`. For working with all the models covered in the paper, install [Pytorch Geometric Temporal](https://pytorch-geometric-temporal.readthedocs.io/en/latest/notes/installation.html).

# Data

In this repository, the skeletal data is already extracted from the video data via MediaPipe Hands in [`/data/graphdata`](/data/graphdata). For demo pruposes, you can run [`utils/video2landmark.py`](utils/video2landmark.py) to create key points and their corresponding label from the video and annotation file in [`data/video_example`](data/video_example).

You'll find the complete dataset here: https://drive.google.com/open?id=1JVPoCJdb8SNbQFLQAc1zDo0EoM6EukEC&authuser=alex.riedel%40gmail.com&usp=drive_fs
Each video corresponds to a .csv file containing the annotations (e.g. `0.csv`) and four .csv files containing the extracted hand keypoints in the original video, horizontally flipped video, vertically flipped video, and horizontally+vertically flipped video (=data augmentation!). 

# Training & Testing

Training the model can is done via [`train.py`](train.py) and the training parameters (models, lr, bs, additional joints, etc.) are defined in [`config.py`](config.py). The training and validation split is defined in [`data/get_data_from_csv.py`](data/get_data_from_csv.py).


To get the final results using 2s-AGCN from the paper, run [`eval.py`](eval.py) (either with the "Release" class or without, as defined in [`config.py`](config.py) `no_release`). 

To inference on an example video including visualization, run [`inference.py`](inference.py). 

<p align="center">
<img src="/imgs/grasp.png" width="60%" height="60%">
</p>

# Contact

For any questions leave a comment or contact me at alexander.riedel@eah-jena.de
