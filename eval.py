import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd 
import os
import time
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
import sklearn

from model import lstm, aagcn, stconv, SAM
from data.handpose_dataset import HandPoseDatasetNumpy, df_to_numpy
from data.get_data_from_csv import get_train_data, get_val_data
from config import CFG
from utils import training_supervision, adj_mat

is_cuda = torch.cuda.is_available()

if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(device)

curr_dir = os.path.dirname(__file__)

df_val = get_val_data()

#no release
if CFG.no_release:
    df_val = df_val.replace("Release", "Position")
    
print("[INFO] TEST DATA DISTRIBUTION")
print(df_val["LABEL"].value_counts())


#Ordering of the get_dummies Pandas
#Grasp   Move    Negative    Position    Reach   Release
#0       0       1           0           0       0


def eval_func(model, criterion, data_loader, epoch):
    model.eval()
    preds = []
    groundtruth = []
    t0 = time.time()
    loss_total = 0
    global_step = 0
    iters = len(data_loader)
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):
            labels = labels.cuda().long()
            inputs = inputs.cuda().float()

            last_label = labels[:, -1, :]
            last_label = torch.argmax(last_label, 1)

            last_out = model(inputs)
            loss = criterion(last_out, last_label)

            preds.append(last_out.cpu().detach().numpy())
            groundtruth.append(last_label.cpu().detach().numpy())
            loss_total += loss

            if i%CFG.print_freq == 1 or i == iters-1:
                t1 = time.time()
                print(f"[EVAL] Iteration: {i}/{iters} | Val-Loss: {loss_total/i} | ETA: {((t1-t0)/i * iters) - (t1-t0)}s")

    return loss_total, np.array(preds),  np.array(groundtruth).flatten()

class FocalLoss(nn.Module):
    
    def __init__(self, weight=None, 
                 gamma=2., reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob, 
            target_tensor, 
            weight=self.weight,
            reduction = self.reduction
        )
 
val_numpy = df_to_numpy(df_val)

test_set_1 = HandPoseDatasetNumpy(val_numpy, distances=False)
test_loader_1 = DataLoader(test_set_1, batch_size=CFG.batch_size, drop_last=True)
graph = aagcn.Graph(adj_mat.num_node, adj_mat.self_link, adj_mat.inward, adj_mat.outward, adj_mat.neighbor)
model_1 = aagcn.Model(num_class=CFG.num_classes, num_point=21, num_person=1, graph=graph, drop_out=0.5, in_channels=3)

test_set_2 = HandPoseDatasetNumpy(val_numpy, distances=True)
test_loader_2 = DataLoader(test_set_2, batch_size=CFG.batch_size, drop_last=True)
graph = aagcn.Graph(adj_mat.num_node, adj_mat.self_link, adj_mat.inward, adj_mat.outward, adj_mat.neighbor)
model_2 = aagcn.Model(num_class=CFG.num_classes, num_point=21, num_person=1, graph=graph, drop_out=0.5, in_channels=3)
print(f"[INFO] TESTING ON {len(test_set_1)} DATAPOINTS")

if CFG.no_release:
    MODEL_PATH_1 =  os.path.join(curr_dir, "trained_models/1AAGCN_Focal_seqlen32_no_release_SAM_joints1_joints2_ori/f10.86875_valloss179.26463317871094_epoch17.pth")
    MODEL_PATH_2 =  os.path.join(curr_dir, "trained_models/6AAGCN_Focal_seqlen32_no_release_SAM_joints1_joints2_oridist/f10.8345518867924528_valloss234.12368774414062_epoch12.pth")
else:
    MODEL_PATH_1 =  os.path.join(curr_dir, "trained_models/3_AAGCN_Focal_seqlen32_release_SAM_joints1_joints2_ori/f10.8439268867924529_valloss246.87600708007812_epoch12.pth")
    MODEL_PATH_2 =  os.path.join(curr_dir, "trained_models/7_AAGCN_Focal_seqlen32_release_SAM_joints1_joints2_oridist/f10.8142688679245284_valloss310.2437744140625_epoch13.pth")

def train_eval():
    model_1.load_state_dict(torch.load(MODEL_PATH_1)["model_state_dict"])
    model_2.load_state_dict(torch.load(MODEL_PATH_2)["model_state_dict"])
    model_1.cuda()
    model_2.cuda()

    criterion = FocalLoss()

    #VAL
    val_loss_1, preds_val_1, gt_val_1 = eval_func(model_1, criterion, test_loader_1, 0)
    print(f"[EVAL] VALIDATION LOSS MODEL 1 {val_loss_1}")
    print(classification_report(gt_val_1, np.argmax(preds_val_1, axis=2).flatten(), target_names=CFG.classes, digits=4))

    val_loss_2, preds_val_2, gt_val_2 = eval_func(model_2, criterion, test_loader_2, 0)
    print(f"[EVAL] VALIDATION LOSS MODEL 2 {val_loss_2}")
    print(classification_report(gt_val_2, np.argmax(preds_val_2, axis=2).flatten(), target_names=CFG.classes, digits=4))
    
    preds_val = 0.5*preds_val_1 + 0.5*preds_val_2
    preds_val = np.argmax(preds_val, axis=2).flatten()    

    f1_val = f1_score(gt_val_1, preds_val, average="micro")

    print("[EVAL] Classification Report")
    print(f"F1-VAL: {f1_val}")
    print(classification_report(gt_val_1, preds_val, target_names=CFG.classes, digits=4))


if __name__ == "__main__":
    train_eval()
