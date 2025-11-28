"""
Training and inference functions for different models.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import LSTMPredictor, LSTMAE, LSTMVAE, vae_loss
from utils import PredDataset, RecDataset


class EarlyStop:
    """Early stopping helper."""
    
    def __init__(self, patience=5):
        self.best = np.inf
        self.count = 0
        self.patience = patience
        self.ckpt = None
    
    def step(self, val):
        if val < self.best - 1e-6:
            self.best = val
            self.count = 0
            return True
        self.count += 1
        return False
    
    def should_stop(self):
        return self.count >= self.patience


# ============= LSTM Predictor =============
def train_predictor(X_train, w, d, config, device):
    """Train LSTM predictor."""
    ds = PredDataset(X_train, w)
    if len(ds) == 0:
        return None
    
    dl = DataLoader(
        ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"]
    )
    
    model = LSTMPredictor(
        d,
        config["hidden"],
        config["n_layers"],
        config["dropout"]
    ).to(device)
    
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = nn.MSELoss()
    stopper = EarlyStop(config["patience"])
    
    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    
    for epoch in range(config["epochs"]):
        model.train()
        losses = []
        
        for xb, yb in dl:
            xb = xb.to(device).float()
            yb = yb.to(device).float()
            
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            
            losses.append(loss.item())
        
        val = float(np.mean(losses)) if losses else np.inf
        
        if stopper.step(val):
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        
        if stopper.should_stop():
            break
    
    model.load_state_dict(best_state)
    model.to(device)
    return model


def infer_predictor(model, X, w, config, device):
    """Run inference with LSTM predictor."""
    ds = PredDataset(X, w)
    if len(ds) == 0:
        return X.copy()
    
    dl = DataLoader(
        ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"]
    )
    
    outs = []
    model.eval()
    
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device).float()
            pred = model(xb).cpu().numpy()
            outs.append(pred)
    
    Ypred = np.vstack(outs)
    
    T, d = X.shape
    full = X.copy()
    full[w:] = Ypred
    full[:w] = X[:w]
    
    return full


# ============= LSTM Autoencoder =============
def train_autoencoder(X_train, w, d, config, device):
    """Train LSTM autoencoder."""
    ds = RecDataset(X_train, w)
    if len(ds) == 0:
        return None
    
    dl = DataLoader(
        ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"]
    )
    
    model = LSTMAE(
        d,
        config["hidden"],
        config["n_layers"],
        config["dropout"]
    ).to(device)
    
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = nn.MSELoss()
    stopper = EarlyStop(config["patience"])
    
    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    
    for epoch in range(config["epochs"]):
        model.train()
        losses = []
        
        for xb, yb in dl:
            xb = xb.to(device).float()
            yb = yb.to(device).float()
            
            opt.zero_grad()
            recon = model(xb)
            loss = loss_fn(recon, yb)
            loss.backward()
            opt.step()
            
            losses.append(loss.item())
        
        val = float(np.mean(losses)) if losses else np.inf
        
        if stopper.step(val):
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        
        if stopper.should_stop():
            break
    
    model.load_state_dict(best_state)
    model.to(device)
    return model


def infer_autoencoder(model, X, w, config, device):
    """Run inference with LSTM autoencoder."""
    ds = RecDataset(X, w)
    if len(ds) == 0:
        return X.copy()
    
    dl = DataLoader(
        ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"]
    )
    
    model.eval()
    outs = []
    
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device).float()
            recon = model(xb).cpu().numpy()
            outs.append(recon)
    
    recon_seq = np.vstack(outs)
    
    T, d = X.shape
    R = np.zeros((T, d), dtype=np.float32)
    C = np.zeros((T, d), dtype=np.float32)
    
    for i in range(recon_seq.shape[0]):
        R[i:i+w] += recon_seq[i]
        C[i:i+w] += 1.0
    
    C[C == 0] = 1.0
    return R / C


# ============= LSTM VAE =============
def train_vae(X_train, w, d, config, device):
    """Train LSTM VAE."""
    ds = RecDataset(X_train, w)
    if len(ds) == 0:
        return None
    
    dl = DataLoader(
        ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"]
    )
    
    model = LSTMVAE(
        d,
        config["hidden"],
        z_dim=config.get("z_dim", 64),
        num_layers=config["n_layers"],
        dropout=config["dropout"]
    ).to(device)
    
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"])
    stopper = EarlyStop(config["patience"])
    
    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    
    for epoch in range(config["epochs"]):
        model.train()
        losses = []
        
        for xb, yb in dl:
            xb = xb.to(device).float()
            yb = yb.to(device).float()
            
            opt.zero_grad()
            recon, mu, logvar = model(xb)
            loss, _, _ = vae_loss(recon, yb, mu, logvar)
            loss.backward()
            opt.step()
            
            losses.append(loss.item())
        
        val = float(np.mean(losses)) if losses else np.inf
        
        if stopper.step(val):
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        
        if stopper.should_stop():
            break
    
    model.load_state_dict(best_state)
    model.to(device)
    return model


def infer_vae(model, X, w, config, device):
    """Run inference with LSTM VAE."""
    ds = RecDataset(X, w)
    if len(ds) == 0:
        return X.copy()
    
    dl = DataLoader(
        ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"]
    )
    
    model.eval()
    outs = []
    
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device).float()
            recon, _, _ = model(xb)
            outs.append(recon.cpu().numpy())
    
    recon_seq = np.vstack(outs)
    
    T, d = X.shape
    R = np.zeros((T, d), dtype=np.float32)
    C = np.zeros((T, d), dtype=np.float32)
    
    for i in range(recon_seq.shape[0]):
        R[i:i+w] += recon_seq[i]
        C[i:i+w] += 1.0
    
    C[C == 0] = 1.0
    return R / C
