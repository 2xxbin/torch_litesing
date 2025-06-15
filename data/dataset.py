from torch.utils.data import Dataset
import numpy as np
import torch
import os

class PredictorDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir

        self.file_list = [base.replace("_bap.npy", "") for base in os.listdir(os.path.join(data_dir, "bap"))]

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        key = self.file_list[index]

        note = np.load(os.path.join(self.data_dir, "note", key + "_note.npy"))
        phone = np.load(os.path.join(self.data_dir, "phone", key + "_phone.npy"))
        energy = np.load(os.path.join(self.data_dir, "energy", key + "_energy.npy"))

        if note.ndim == 1: note = note[:, None]
        if phone.ndim == 1: phone = phone[:, None]
        if energy.ndim == 1: energy = energy[:, None]
        
        f0 = np.load(os.path.join(self.data_dir, "f0", key + "_f0.npy"))
        vuv = np.load(os.path.join(self.data_dir, "vuv", key + "_vuv.npy"))

        feats = np.concatenate([note, phone], axis=-1)

        feats = torch.from_numpy(feats).float()
        f0 = torch.from_numpy(f0).float()
        vuv = torch.from_numpy(vuv).float()
        energy = torch.from_numpy(energy).float()

        if f0.ndim == 1 : f0 = f0.unsqueeze(-1)
        if vuv.ndim == 1: vuv = vuv.unsqueeze(-1)
        if energy.ndim == 1: energy = energy.unsqueeze(-1)

        return feats, f0, vuv, energy
