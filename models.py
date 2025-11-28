"""
Neural network models for anomaly detection.
"""
import torch
import torch.nn as nn


class LSTMPredictor(nn.Module):
    """Many-to-one LSTM: (B, w, d) → predict next x (B, d)."""
    
    def __init__(self, d, hidden=128, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            d, hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden, d)
    
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        z = h[-1]
        y = self.fc(z)
        return y


class LSTMAE(nn.Module):
    """Seq2Seq LSTM Autoencoder: (B, w, d) → reconstruct (B, w, d)."""
    
    def __init__(self, d, hidden=128, num_layers=1, dropout=0.2):
        super().__init__()
        self.encoder = nn.LSTM(
            d, hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.decoder = nn.LSTM(
            d, hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden, d)
    
    def forward(self, x):
        enc_out, (h, c) = self.encoder(x)
        dec_out, _ = self.decoder(x, (h, c))
        y = self.fc(dec_out)
        return y


class LSTMVAE(nn.Module):
    """Seq2Seq LSTM VAE: (B,w,d) → recon (B,w,d) with latent z."""
    
    def __init__(self, d, hidden=128, z_dim=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.d = d
        self.hidden = hidden
        self.z_dim = z_dim
        self.num_layers = num_layers
        
        self.encoder = nn.LSTM(
            d, hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.mu = nn.Linear(hidden, z_dim)
        self.logvar = nn.Linear(hidden, z_dim)
        
        self.z_to_h = nn.Linear(z_dim, hidden)
        self.decoder = nn.LSTM(
            d, hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden, d)
    
    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        _, (h, c) = self.encoder(x)
        h_last = h[-1]
        
        mu = self.mu(h_last)
        logvar = self.logvar(h_last)
        z = self.reparam(mu, logvar)
        
        h0_1 = torch.tanh(self.z_to_h(z))
        h0 = h0_1.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)
        
        dec_out, _ = self.decoder(x, (h0, c0))
        y = self.fc(dec_out)
        
        return y, mu, logvar


def vae_loss(recon, target, mu, logvar):
    """Compute VAE loss (reconstruction + KL divergence)."""
    mse = nn.functional.mse_loss(recon, target, reduction='mean')
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + 0.001 * kl, mse.item(), kl.item()
