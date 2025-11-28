# utils.py
import torch
import os

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_class_names(data_path):
    return sorted([folder for folder in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, folder))])
