import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import cv2
import mediapipe as mp
import ast
import time
import os 

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 

from model import lstm, stconv, aagcn, SAM, loss, msg3d
from config import CFG
from utils import adj_mat

curr_dir = os.path.dirname(__file__)


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.4, max_num_hands=1)

is_cuda = torch.cuda.is_available()

if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(device)

"""
Ordering of the get_dummies Pandas

Grasp   Move    Negative    Position    Reach   Release
0       0       1           0           0       0
"""

def progressBar(pil_im, bgcolor, color, x, y, w, h, progress):   	 
    #im = Image.open(imgPath)    
    drawObject = ImageDraw.Draw(pil_im)
    
    '''BG'''    
    drawObject.ellipse((x+w,y,x+h+w,y+h),fill=bgcolor)    
    drawObject.ellipse((x,y,x+h,y+h),fill=bgcolor)    
    drawObject.rectangle((x+(h/2),y, x+w+(h/2), y+h),fill=bgcolor)
    
    '''PROGRESS'''    
    if(progress<=0):        
        progress = 0.01    
    if(progress>1):        
        progress=1    
    w = w*progress    
    drawObject.ellipse((x+w,y,x+h+w,y+h),fill=color)    
    drawObject.ellipse((x,y,x+h,y+h),fill=color)    
    drawObject.rectangle((x+(h/2),y, x+w+(h/2), y+h),fill=color)
    return pil_im

def results2landmarks(results):
  landmarks_per_frame = []

  if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
      for landmark_pos in hand_landmarks.landmark:
        landmarks_per_frame.append([landmark_pos.x, landmark_pos.y, landmark_pos.z])
        #landmarks_per_frame.append([1*i, 1*i, 1*i])
      mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

  else: #if no landmarks found
      for landmark_name in range(NUM_LANDMARKS):
        #landmarks_per_frame.append([1*i, 1*i, 1*i])
        landmarks_per_frame.append([0, 0, 0])

  landmarks_per_frame = np.array(landmarks_per_frame)
  return landmarks_per_frame

    
graph = aagcn.Graph(adj_mat.num_node, adj_mat.self_link, adj_mat.inward, adj_mat.outward, adj_mat.neighbor)
model = aagcn.Model(num_class=CFG.num_classes, num_point=21, num_person=1, graph=graph, drop_out=0.5, in_channels=3)

model.cuda() 
model.eval()
MODEL_PATH = os.path.join(curr_dir, "trained_models/3_AAGCN_Focal_seqlen32_release_SAM_joints1_joints2_ori/f10.8439268867924529_valloss246.87600708007812_epoch12.pth")

model.load_state_dict(torch.load(MODEL_PATH)["model_state_dict"])

NUM2CLASSES_GER = {
    0: "Greifen",
    1: "Bringen",
    2: "Negativ",    
    3: "Fuegen",    
    4: "Hinlangen",   
    5: "Loslassen"
}
NUM2CLASSES_EN = {
    0: "Grasp",
    1: "Move",
    2: "Negative",    
    3: "Position",    
    4: "Reach",   
    5: "Release"
}

CLASSES2NUM_EN = dict((v,k) for k,v in NUM2CLASSES_EN.items())

class_number_remap = {
  0:2,
  1:3,
  2:0,
  3:4,
  4:1,
  5:5
}


cap = cv2.VideoCapture(os.path.join(curr_dir, "data/video_example/2.mp4"))
gt_data = pd.read_csv(os.path.join(curr_dir, "data/video_example/2.csv"), squeeze = True)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
fps = cap.get(cv2.CAP_PROP_FPS)

NUM_LANDMARKS = 21

landmarks_sliding = np.zeros((32, 21, 3))

mtm_motion_before = None
argmax_before = 2
accum_frames = 0

i = 0
while cap.isOpened():
  time_s = time.perf_counter()
  ret, image = cap.read()
  if not ret:
    break
  
  # Flip the image horizontally for a later selfie-view display, and convert
  # the BGR image to RGB.

  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  # To improve performance, optionally mark the image as not writeable to
  # pass by reference.
  image.flags.writeable = False

  results = hands.process(image)

  # Draw the hand annotations on the image.
  image.flags.writeable = True
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

  landmarks_per_frame = results2landmarks(results)

  landmarks_sliding[:-1] = landmarks_sliding[1:]
  landmarks_sliding[-1] = landmarks_per_frame
  landmarks_sliding_input = torch.from_numpy(landmarks_sliding[None, :, :]).cuda().float()


  #getting model predictions
  logits = model(landmarks_sliding_input)
  probas = F.softmax(logits).cpu().detach().numpy()

  #visualize predictions on rendered frame
  pil_image = Image.fromarray(image)
  fnt = ImageFont.truetype("arial.ttf", 20)
  for n, c in NUM2CLASSES_EN.items():
    offtop = 50
    offleft = 200
    draw = ImageDraw.Draw(pil_image)
    draw.rectangle((offleft-180, offtop+n*30, offleft-180+350, offtop+n*30+20), fill='black')
    txt = f"{int(probas[0, n]*100):02d}% {c}"
    draw.text((offleft-180,offtop+n*30), txt, fill=(0, 255, 0), font=fnt)
    
    pil_image = progressBar(pil_image, (0,0,0), (0,255,0), offleft-40, offtop+n*30+7, 200, 10, probas[0, n])

  #get highest prob (maybe smooth this over n-frames)

  #print(probas_full)
  argmax = np.argmax(probas)
  

  gt = CLASSES2NUM_EN[gt_data.iloc[i]]
  accum_frames += i

  #for gt
  argmax = gt
  mtm_motion = NUM2CLASSES_EN[argmax]
  if mtm_motion != mtm_motion_before:
    print(class_number_remap[argmax_before], round(i*(1/fps), 3))
    mtm_motion_before = mtm_motion
    argmax_before = argmax
    print(class_number_remap[argmax_before], round(i*(1/fps), 3))
    #i = 0
    

  numpy_image=np.array(pil_image)  
  #opencv_image=cv2.cvtColor(numpy_image, cv2.COLOR_BGR) 
  time_e = time.perf_counter()

  print("Dur:", time_e - time_s)

  i += 1
  cv2.imshow('MediaPipe Hands', numpy_image)
  if cv2.waitKey(1) & 0xFF == 27:
    break
hands.close()
cap.release()