import pandas as pd 
import os
from config import CFG

curr_dir = os.path.dirname(__file__)

train_ids = [0, 2, 5, 7, 9, 10, 11, 13, 21, 22, 23]
val_ids = [6, 12, 20]


def dfs_from_ids(ids, get_augmented=True):
    dfs = []
    for i in ids:
        df = pd.read_csv(os.path.join(curr_dir, f"graphdata/{i}_mdc04_mtc05_Train_original.csv"), index_col=0)
        if get_augmented:
            df_f0 = pd.read_csv(os.path.join(curr_dir, f"graphdata/{i}_mdc04_mtc05_Train_flip-vert.csv"), index_col=0)
            df_f1 = pd.read_csv(os.path.join(curr_dir, f"graphdata/{i}_mdc04_mtc05_Train_flip-hor.csv"), index_col=0)
            df_f2 = pd.read_csv(os.path.join(curr_dir, f"graphdata/{i}_mdc04_mtc05_Train_flip-hor-vert.csv"), index_col=0)

            dfs.extend([df, df_f0, df_f1, df_f2])
        else:
            dfs.append(df)
    return dfs

def get_train_data():
    train_dfs = dfs_from_ids(train_ids)
    df_train = pd.concat(train_dfs)
    return df_train

def get_val_data():
    val_dfs = dfs_from_ids(val_ids)
    df_val = pd.concat(val_dfs)
    return df_val

def get_original_data():
    train_dfs = dfs_from_ids(train_ids, get_augmented=False)
    val_dfs = dfs_from_ids(val_ids, get_augmented=False)
    train = pd.concat(train_dfs)
    val = pd.concat(val_dfs)
    return train, val


if __name__ == "__main__":
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
    import json
    import numpy as np

    df  = get_original_data()
    df.reset_index(drop=True, inplace=True)
    df = df.replace("Postion", "Position")
    df["LABEL"] = df["LABEL"].astype('category')
    print(df["LABEL"])

    df["LABEL"] = df["LABEL"].cat.codes

    edge_index = [[0,1], [0,5], [0,9], [0,17], [1,2], [2,3], [3,4], [6,5], [6,7], [7,8], [9,10], [10,11], [11,12], [13,14], [14,15], [15,16], [17,18], [18,19], [19,20]]
    json_name = "C:/Users/REH/Google Drive/mtm recognition/HAND_LSTM/data/mtm_1.json"
    
    df.rename(hand_landmarks_dict, axis=1, inplace=True)
    print(df)
    df.to_json(json_name, orient="columns")

    edge_dict = {"edges" : edge_index}
    with open(json_name, "r+") as file:
        data = json.load(file)
        data.update(edge_dict)
        file.seek(0)
        json.dump(data, file)

    f = open(json_name)
    df_json = json.load(f)
