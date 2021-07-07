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

from model import lstm, stconv, aagcn, SAM, loss, msg3d
from data.handpose_dataset import HandPoseDatasetNumpy, df_to_numpy
from data.get_data_from_csv import get_train_data, get_val_data
from config import CFG
from utils import training_supervision, adj_mat
from torchsummary import summary

is_cuda = torch.cuda.is_available()

if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(device)

curr_dir = os.path.dirname(__file__)
os.makedirs(f"{curr_dir}/trained_models/{CFG.experiment_name}", exist_ok=True)

df_train = get_train_data()
#misspelling on annotation :(
df_train = df_train.replace("Postion", "Position")
df_val = get_val_data()

if CFG.no_release:
    df_train = df_train.replace("Release", "Position")
    df_val = df_val.replace("Release", "Position")
    
print("[INFO] TRAIN DATA DISTRIBUTION")
print(df_train["LABEL"].value_counts())
print("[INFO] VALIDATION DATA DISTRIBUTION")
print(df_val["LABEL"].value_counts())


#Order of One Hot Encoding
#Grasp   Move    Negative    Position    Reach   Release
#0       0       1           0           0       0

if CFG.debug:
    df_train = df_train[:500]

def train_func(model, data_loader, criterion, optimizer, scheduler, epoch):
    model.train()
    iters = len(data_loader)
    global_step = epoch*len(data_loader)
    preds = []
    groundtruth = []
    t0 = time.time()
    loss_total = 0
    for i, (inputs, labels) in enumerate(data_loader):
        labels = labels.cuda().long()
        inputs = inputs.cuda().float()

        last_label = labels[:, -1, :]
        last_label = torch.argmax(last_label, 1)

        model.zero_grad()
        last_out = model(inputs)

        #first forward-backward pass
        loss = criterion(last_out, last_label)
        loss.backward()
        
        if CFG.sam:
            optimizer.first_step(zero_grad=True)#

            # second forward-backward pass
            criterion(model(inputs), last_label).backward()#
            optimizer.second_step(zero_grad=True)#
        else:
            optimizer.step()
        
        current_lr = optimizer.param_groups[0]['lr']

        preds.append(last_out.cpu().detach().numpy())
        groundtruth.append(last_label.cpu().detach().numpy())

        loss_total += loss
        global_step += 1
        writer.add_scalar('Loss/Train', loss, global_step)
        writer.add_scalar('LR', current_lr, global_step)
        
        if i%CFG.print_freq == 1 or i == iters-1:
            t1 = time.time()
            print(f"[TRAIN] Epoch: {epoch}/{CFG.epochs} | Iteration: {i}/{iters} | Loss: {loss_total/i} | LR: {current_lr} | ETA: {((t1-t0)/i * iters) - (t1-t0)}s")

    return loss_total, np.argmax(preds, axis=2).flatten(),  np.array(groundtruth).flatten()

def eval_func(model, criterion, data_loader, epoch):
    model.eval()
    preds = []
    groundtruth = []
    t0 = time.time()
    loss_total = 0
    global_step = len(train_loader)*epoch
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
                print(f"[EVAL] Epoch: {epoch}/{CFG.epochs} | Iteration: {i}/{iters} | Val-Loss: {loss_total/i} | ETA: {((t1-t0)/i * iters) - (t1-t0)}s")

    writer.add_scalar('Loss/Validation', loss_total/i, global_step)
    return loss_total, np.argmax(preds, axis=2).flatten(),  np.array(groundtruth).flatten()


train_numpy = df_to_numpy(df_train)
val_numpy = df_to_numpy(df_val)
 
train_set = HandPoseDatasetNumpy(train_numpy)
val_set = HandPoseDatasetNumpy(val_numpy)

train_loader = DataLoader(train_set, batch_size=CFG.batch_size, drop_last=True, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=CFG.batch_size, drop_last=True, pin_memory=True)

print(f"[INFO] TRAINING ON {len(train_set)} DATAPOINTS")
print(f"[INFO] VALIDATION ON {len(val_set)} DATAPOINTS")


writer = SummaryWriter(f'C:/Users/REH/Google Drive/mtm recognition/runs/{CFG.experiment_name}')

def train_eval():
    if CFG.model_type == "LSTM":
        model = lstm.LSTMClassifier(input_dim=CFG.num_feats*21, hidden_dim=CFG.lstm_hidden_layers, num_layers=CFG.lstm_num_layers, output_dim=CFG.num_classes, seq_len=CFG.sequence_length, batch_size=CFG.batch_size)
    
    if CFG.model_type == "STCONV":
        model = stconv.STConvOpolkaModel(in_channels=CFG.num_feats, spatial_channels=CFG.stconv_spatial_channels, out_channels=CFG.stconv_out_channels, classes=CFG.num_classes)

    if CFG.model_type == "AAGCN":
        graph = adj_mat.Graph(adj_mat.num_node, adj_mat.self_link, adj_mat.inward, adj_mat.outward, adj_mat.neighbor)
        model = aagcn.Model(num_class=CFG.num_classes, num_point=21, num_person=1, graph=graph, drop_out=0.5, in_channels=CFG.num_feats)

    if CFG.model_type == "MSG3D":
        model = msg3d.Model(num_class=6, num_point=21, num_person=1, num_gcn_scales=13, num_g3d_scales=6,graph=msg3d.AdjMatrixGraph)

    start_epoch = 0
    model.cuda()
    print(summary(model, (CFG.sequence_length, 21, CFG.num_feats)))

    if CFG.loss_fn == "BCE":
        class_weights = class_weight.compute_class_weight('balanced',np.unique(df_train["LABEL"]),df_train["LABEL"])
        class_weights = torch.tensor(class_weights).cuda().float()
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    if CFG.loss_fn == "Focal":
        criterion = loss.FocalLoss()

    if CFG.sam:
        optimizer_base = torch.optim.Adam
        optimizer = SAM.SAM(model.parameters(), optimizer_base,  lr=CFG.lr, rho=0.5, adaptive=True)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=CFG.min_lr)

    for epoch in range(start_epoch, CFG.epochs+start_epoch):
        global_step = len(train_loader)*epoch

        #TRAIN
        train_loss, preds_train, gt_train = train_func(model, train_loader, criterion, optimizer, scheduler, epoch)
        train_grad_flow_plot = training_supervision.get_plot_grad_flow(model)

        f1_train = f1_score(gt_train, preds_train, average="micro")
        writer.add_scalar('Accuracy/Train', f1_train, global_step)
        print(f"[TRAIN] Training F1-Score {f1_train}")

        #Model Gradients
        names, gradmean = training_supervision.get_model_grads(model)
        _limits = np.array([float(i) for i in range(len(gradmean))]) 
        _num = len(gradmean)
        writer.add_histogram_raw(tag="ModelGrads/MeanGradientFlow", min=0.0, max=0.5, num=_num,
                                sum=gradmean.sum(), sum_squares=np.power(gradmean, 2).sum(), bucket_limits=_limits,
                                bucket_counts=gradmean, global_step=global_step)
        
        #VAL
        val_loss, preds_val, gt_val = eval_func(model,criterion, val_loader, epoch)
        

        f1_val_micro = f1_score(gt_val, preds_val, average="micro")
        f1_val_macro = f1_score(gt_val, preds_val, average="macro")
        writer.add_scalar('Accuracy/Validation/F1-Micro', f1_val_micro, global_step)
        writer.add_scalar('Accuracy/Validation/F1-Macro', f1_val_macro, global_step)
        print(f"[EVAL] Validation F1-Score Micro {f1_val_micro}")
        print(f"[EVAL] Validation F1-Score Macro {f1_val_macro}")

        #Conf Mat
        cm = sklearn.metrics.confusion_matrix(gt_val, preds_val)
        cm_plot = training_supervision.plot_confusion_matrix(cm, CFG.classes)
        writer.add_figure("Confusion Matrix/Validation", cm_plot, global_step)

        #Model Weights
        names, params = training_supervision.get_model_weights(model)
        for n, p in zip(names, params):
            writer.add_histogram(f"ModelWeights/{n}", p, global_step) 

        print("[EVAL] Classification Report")
        print(classification_report(gt_val, preds_val, target_names=CFG.classes, digits=3))

        scheduler.step(val_loss) #for reduce lr on plateau
        
        PATH = f"{curr_dir}/trained_models/{CFG.experiment_name}/f1{f1_val_micro}_valloss{val_loss}_epoch{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'f1_micro_val-score' : f1_val_micro, 
            }, PATH)
        print("[INFO] MODEL SAVED")


if __name__ == "__main__":
    train_eval()
