from models.ConditionPredictor import ConditionPredictor
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from data.dataset import PredictorDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import yaml
import time
import os


DATA_FOLDER_NAME = "ritsu"
SAVE_PATH = os.path.join(".", "checkpoints", DATA_FOLDER_NAME)


if not os.path.exists(SAVE_PATH): os.mkdir(SAVE_PATH)

SAVE_PATH = os.path.join(SAVE_PATH, "predictor")
if not os.path.exists(SAVE_PATH): os.mkdir(SAVE_PATH)

TENSORBOARD_PATH = os.path.join(SAVE_PATH, "logs")
if not os.path.exists(TENSORBOARD_PATH): os.mkdir(TENSORBOARD_PATH)

writer = SummaryWriter(log_dir=TENSORBOARD_PATH)

def predictor_collate_fn(batch):
    feats, f0, vuv, energy = zip(*batch)
    # (이미 tensor면 아래 줄 필요 없음)
    feats = [x if isinstance(x, torch.Tensor) else torch.tensor(x) for x in feats]
    f0 = [x if isinstance(x, torch.Tensor) else torch.tensor(x) for x in f0]
    # vuv가 (T,)일 수도 있고 (T,1)일 수도 있으니 확실하게 (T,1)로!
    vuv = [x if isinstance(x, torch.Tensor) else torch.tensor(x) for x in vuv]
    energy = [x if isinstance(x, torch.Tensor) else torch.tensor(x) for x in energy]
    feats_padded = pad_sequence(feats, batch_first=True, padding_value=0.0)
    f0_padded = pad_sequence(f0, batch_first=True, padding_value=0.0)
    vuv_padded = pad_sequence(vuv, batch_first=True, padding_value=0.0)
    energy_padded = pad_sequence(energy, batch_first=True, padding_value=0.0)
    return feats_padded, f0_padded, vuv_padded, energy_padded

hparam_f = open('hparams.yaml', 'r')
hparams = yaml.load(hparam_f, Loader=yaml.FullLoader)
train_hparams = hparams['train']
predictor_hparams = hparams['predictor']
hparam_f.close()

DATASET_PATH = os.path.join(".", "data", "preprocessed", DATA_FOLDER_NAME)

train_dataset = PredictorDataset(os.path.join(DATASET_PATH, "train"))
train_loader = DataLoader(train_dataset, batch_size=train_hparams['batch_size'], shuffle=True, collate_fn=predictor_collate_fn)

valid_dataset = PredictorDataset(os.path.join(DATASET_PATH, "validation"))
valid_loader = DataLoader(valid_dataset, batch_size=train_hparams['batch_size'], shuffle=False, collate_fn=predictor_collate_fn)

model = ConditionPredictor(input_dim=predictor_hparams['input_dim'], hidden_dim=predictor_hparams['hidden_dim']).to('cuda')

loss_f0 = nn.MSELoss()
loss_energy = nn.BCELoss()
loss_vuv = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=train_hparams['learning_rate'])

epochs = train_hparams['total_epoch']
model.train()

for epoch in range(epochs):
    total_loss = 0
    total_f0 = 0
    total_energy = 0
    total_vuv = 0

    for i, batch in enumerate(train_loader):
        input_feature, gt_f0, gt_vuv, gt_energy = [x.cuda() for x in batch]

        optimizer.zero_grad()
        pred_f0, pred_energy, pred_vuv = model(input_feature)

        l_f0 = loss_f0(pred_f0, gt_f0)
        l_energy = loss_energy(pred_energy, gt_energy)
        l_vuv = loss_vuv(pred_vuv, gt_vuv)
        loss = l_f0 + l_energy + l_vuv

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_f0 += l_f0.item()
        total_energy += l_energy.item()
        total_vuv += l_vuv.item()

        global_step = epoch * len(train_loader) + i
        writer.add_scalar("Loss/total", loss.item(), global_step)
        writer.add_scalar("Loss/f0", l_f0.item(), global_step)
        writer.add_scalar("Loss/energy", l_energy.item(), global_step)
        writer.add_scalar("Loss/vuv", l_vuv.item(), global_step)
    
    # validation
    model.eval()
    val_total_loss = 0
    val_f0 = 0
    val_energy = 0
    val_vuv = 0

    with torch.no_grad():
        for batch in valid_loader:
            input_feature, gt_f0, gt_vuv, gt_energy = [x.cuda() for x in batch]
            pred_f0, pred_energy, pred_vuv = model(input_feature)

            l_f0 = loss_f0(pred_f0, gt_f0)
            l_energy = loss_energy(pred_energy, gt_energy)
            l_vuv = loss_vuv(pred_vuv, gt_vuv)
            loss = l_f0 + l_energy + l_vuv

            val_total_loss += loss.item()
            val_f0 += l_f0.item()
            val_energy += l_energy.item()
            val_vuv += l_vuv.item()

    avg_loss = total_loss / len(train_loader)
    avg_f0 = total_f0 / len(train_loader)
    avg_energy = total_energy / len(train_loader)
    avg_vuv = total_vuv / len(train_loader)

    avg_val_loss = val_total_loss / len(valid_loader)
    avg_val_f0 = val_f0 / len(valid_loader)
    avg_val_energy = val_energy / len(valid_loader)
    avg_val_vuv = val_vuv / len(valid_loader)

    print(f"Epoch {epoch + 1}: Loss ={avg_loss : .4f} | Valid Loss ={avg_val_loss : .4f}")

    writer.add_scalar("Loss/epoch_total", avg_loss, epoch + 1)
    writer.add_scalar("Loss/epoch_f0", avg_f0, epoch + 1)
    writer.add_scalar("Loss/epoch_energy", avg_energy, epoch + 1)
    writer.add_scalar("Loss/epoch_vuv", avg_vuv, epoch + 1)

    writer.add_scalar("Loss/val_total", avg_val_loss, epoch + 1)
    writer.add_scalar("Loss/val_f0", avg_val_f0, epoch + 1)
    writer.add_scalar("Loss/val_energy", avg_val_energy, epoch + 1)
    writer.add_scalar("Loss/val_vuv", avg_val_vuv, epoch + 1)

    if (epoch + 1) % 10 == 0:
        save_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
        torch.save(model.state_dict(), os.path.join(SAVE_PATH, f"predictor_{save_time}_epoch{epoch + 1}.pth"))

writer.close()