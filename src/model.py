"""
Module for defining the deep autoencoder architecture for anomaly detection
"""

import torch
import torch.nn as nn
import numpy as np

class DeepAutoencoder(nn.Module):
    """Architecture : 33 -> 128 -> 64 -> 32 -> 16 -> 8 -> 16 -> 32 -> 64 -> 128 -> 33"""

    def __init__(self, input_dim = 33, latent_dim = 8):
        super(DeepAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), 
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64), #Batch normalization stabilizes training and thus allows for higher learning rates. 
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 16),
            nn.ReLU(),nn.BatchNorm1d(16), 
            nn.Linear(16, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16), 
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32), 
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64), 
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128), 
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        """Forward pass through the autoencoder."""
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction
    
    def encode(self, x):
        """Encode input to latent space."""
        return self.encoder(x)
    
    def decode(self, latent):
        """Decode latent representation to input space."""
        return self.decoder(latent)

def get_reconstruction_error(model, data_loader, device = 'cpu'):
    """Calculate reconstruction error for a dataset."""
    model.eval()
    errors = []
    with torch.no_grad():  # No gradient computation
        for batch in data_loader:
            # Unpack the batch (input, target)
            if isinstance(batch, (list, tuple)):
                data = batch[0]  # Get input data
            else:
                data = batch  # Single tensor
            
            # Move data to device
            data = data.to(device)
            
            # Get reconstruction
            reconstruction = model(data)
            
            # Calculate Mean Squared Error per sample
            # Shape: (batch_size,)
            mse = torch.mean((data - reconstruction) ** 2, dim = 1)
            
            errors.extend(mse.cpu().numpy())
    return np.array(errors)

def print_model_summary(model, input_dim = 33):
    """Print a summary of the model architecture."""
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: DeepAutoencoder")
    print(f"Input dimension: {model.input_dim}")
    print(f"Latent dimension: {model.latent_dim}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")