import cv2
import mediapipe as mp
from pprint import pprint
import enum
import pandas as pd
import numpy as np
import os

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

"""The 21 hand landmarks."""
hand_landmarks_dict ={
  "WRIST" : 0,
  "THUMB_CMC" : 1,
  "THUMB_MCP" : 2,
  "THUMB_IP" : 3,
  "THUMB_TIP" : 4,
  "INDEX_FINGER_MCP" : 5,
  "INDEX_FINGER_PIP" : 6,
  "INDEX_FINGER_DIP" : 7,
  "INDEX_FINGER_TIP" : 8,
  "MIDDLE_FINGER_MCP" : 9,
  "MIDDLE_FINGER_PIP" : 10,
  "MIDDLE_FINGER_DIP" : 11,
  "MIDDLE_FINGER_TIP" : 12,
  "RING_FINGER_MCP" : 13,
  "RING_FINGER_PIP" : 14,
  "RING_FINGER_DIP" : 15,
  "RING_FINGER_TIP" : 16,
  "PINKY_MCP" : 17,
  "PINKY_PIP" : 18,
  "PINKY_DIP" : 19,
  "PINKY_TIP" : 20,
}

curr_dir = os.path.dirname(__file__)

examples = [23]
for example in examples:
    print("PROCESSING VIDEO ", example)
    landmarks_dict_frames = {}
    landmarks_dict_points = {}

    base_path = os.path.join(curr_dir, "../data/video_example")
    df_labels = pd.read_csv(f"{base_path}/{example}.csv", header=None, names=["LABEL"])


    augmentations = ["original", "flip-vert", "flip-hor", "flip-hor-vert"]
    for aug in augmentations:
        hands = mp_hands.Hands(min_detection_confidence=0.4, min_tracking_confidence=0.5, max_num_hands=1)

        cap = cv2.VideoCapture(f"{base_path}/{example}.mp4")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        i = 0
        while cap.isOpened():
          
          ret, image = cap.read()
          if not ret:
            break
                      
          landmarks_dict_points = {}

          label = df_labels["LABEL"].iloc[i]
          if label == "NO":
            continue
          # Augmentation flipping
          if aug == "original":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
          elif aug == "flip-vert":
            image = cv2.cvtColor(cv2.flip(image, 0), cv2.COLOR_BGR2RGB)
          elif aug == "flip-hor":
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
          elif aug == "flip-hor-vert":
            image = cv2.cvtColor(cv2.flip(image, -1), cv2.COLOR_BGR2RGB)

          # To improve performance, optionally mark the image as not writeable to
          # pass by reference.
          image.flags.writeable = False
          results = hands.process(image)

          # Draw the hand annotations on the image.
          image.flags.writeable = True
          image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
          
          draw_blank = np.full((720, 1280, 3), 255, np.uint8)

          if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
              
              for landmark_pos, landmark_name in zip(hand_landmarks.landmark, hand_landmarks_dict.keys()):
                landmarks_dict_points[landmark_name] = (landmark_pos.x, landmark_pos.y, landmark_pos.z)
              mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

          else: #if no landmarks found
              for landmark_name in (hand_landmarks_dict.keys()):
                  landmarks_dict_points[landmark_name] = (0, 0, 0)
                  label = "Negative"
          
          landmarks_dict_points["LABEL"] = label
          landmarks_dict_frames[i] = landmarks_dict_points
          i += 1
          cv2.imshow('MediaPipe Hands', image)
          if cv2.waitKey(1) & 0xFF == 27:
            break

        df_landmarks = pd.DataFrame(landmarks_dict_frames).transpose()
        df_landmarks_labels = pd.concat([df_landmarks, df_labels], axis=1)
        df_landmarks.to_csv(f"{base_path}/{example}_mdc04_mtc05_Train_{aug}.csv")

