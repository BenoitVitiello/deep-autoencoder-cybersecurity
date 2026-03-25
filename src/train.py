"""
Training functions for the deep autoencoder
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
try:
    from .preprocessing import load_preprocessed_data
    from .model import DeepAutoencoder, print_model_summary
except ImportError:
    from preprocessing import load_preprocessed_data
    from model import DeepAutoencoder, print_model_summary


def create_dataloaders(X_train, X_val, X_test, batch_size = 256):
    """
    Create PyTorch DataLoaders from numpy arrays
    """
    #Tensors
    X_train_tensor = torch.FloatTensor(X_train)
    X_val_tensor = torch.FloatTensor(X_val)
    X_test_tensor = torch.FloatTensor(X_test)
    
    #Datasets
    train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, X_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, X_test_tensor)
    
    #DataLoaders
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
    
    print(f"DataLoaders created:")
    print(f"Batch size : {batch_size}")
    print(f"Train batches : {len(train_loader)}")
    print(f"Val batches : {len(val_loader)}")
    print(f"Test batches : {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


def initialize_model(input_dim = 33, latent_dim = 8, device = None):
    """
    Initialize the deep autoencoder model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = DeepAutoencoder(input_dim = input_dim, latent_dim = latent_dim)
    model = model.to(device)
    
    print_model_summary(model, input_dim = input_dim)
    return model, device


def setup_training(model, learning_rate = 0.001):
    """
    Setup loss function, optimizer, and scheduler
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode = 'min',
        factor = 0.5,
        patience = 5,
    )
    
    print(f"Training setup:")
    print(f"Optimizer : Adam")
    print(f"Learning rate : {learning_rate}")
    print(f"Loss function : MSE")
    print(f"Scheduler : ReduceLROnPlateau")
    
    return criterion, optimizer, scheduler


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch
    """
    model.train()
    train_loss = 0.0
    
    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        reconstruction = model(data)
        loss = criterion(reconstruction, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss = train_loss / len(train_loader)
    return train_loss


def validate_one_epoch(model, val_loader, criterion, device):
    """
    Validate the model for one epoch
    """
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            target = target.to(device)
            
            reconstruction = model(data)
            loss = criterion(reconstruction, target)
            
            val_loss += loss.item()
    
    val_loss = val_loss / len(val_loader)
    return val_loss


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                device, num_epochs = 50, early_stopping_patience = 10, 
                model_save_path = '../models/autoencoder.pth'):
    """
    Complete training loop with early stopping
    """
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        #Training
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        #Validation
        val_loss = validate_one_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        #Update scheduler
        scheduler.step(val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{num_epochs}] | "
                  f"Train Loss : {train_loss:.6f} | "
                  f"Val Loss : {val_loss:.6f}")
        
        #Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            model_path = Path(model_save_path).parent
            model_path.mkdir(parents = True, exist_ok = True)
            torch.save(model.state_dict(), model_save_path)
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Val Loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1
        
        #Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    print("\n" + "=" * 70)
    print(f"Validation loss: {best_val_loss:.6f}")

    
    return {
        'train_losses' : train_losses,
        'val_losses' : val_losses,
        'best_val_loss' : best_val_loss,
        'num_epochs' : len(train_losses)
    }


def plot_training_history(train_losses, val_losses, save_path = None):
    """
    Plot training and validation loss curves
    """
    plt.figure(figsize = (12, 5))
    
    # Linear scale
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label = 'Training Loss', linewidth = 2)
    plt.plot(val_losses, label = 'Validation Loss', linewidth = 2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss', fontweight = 'bold')
    plt.legend()
    plt.grid(True, alpha = 0.3)
    
    # Log scale
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label = 'Training Loss', linewidth = 2)
    plt.plot(val_losses, label = 'Validation Loss', linewidth = 2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss (Log Scale)', fontweight = 'bold')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha = 0.3)
    
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents = True, exist_ok = True)
        plt.savefig(save_path, dpi = 150, bbox_inches = 'tight')
    plt.show()

def train_autoencoder(data_dir = '../data/processed',
                      model_dir = '../models',
                      results_dir = '../results',
                      input_dim = 33,
                      latent_dim = 8,
                      batch_size = 256,
                      learning_rate = 0.001,
                      num_epochs = 50,
                      early_stopping_patience = 10,
                      device = None):
    """
    Complete training pipeline (combines all functions above, standalone use)
    """
    
    # Load data
    data = load_preprocessed_data(data_dir)
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    
    print(f"Train: {X_train.shape}")
    print(f"Val:   {X_val.shape}")
    print(f"Test:  {X_test.shape}")
    
    #DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(X_train, X_val, X_test, batch_size)
    
    #Model & params
    model, device = initialize_model(input_dim, latent_dim, device)

    criterion, optimizer, scheduler = setup_training(model, learning_rate)
    
    #Train
    model_save_path = f"{model_dir}/autoencoder.pth"
    history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                        device, num_epochs, early_stopping_patience, model_save_path)
    
    #Plot history
    plot_save_path = f"{results_dir}/training_history.png"
    plot_training_history(history['train_losses'], history['val_losses'], plot_save_path)
    
    return history


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description = 'Train deep autoencoder for anomaly detection')
    parser.add_argument('--data_dir', type = str, default = '../data/processed')
    parser.add_argument('--model_dir', type = str, default = '../models')
    parser.add_argument('--results_dir', type = str, default = '../results')
    parser.add_argument('--input_dim', type = int, default = 33)
    parser.add_argument('--latent_dim', type = int, default = 8)
    parser.add_argument('--batch_size', type = int, default = 256)
    parser.add_argument('--learning_rate', type = float, default = 0.001)
    parser.add_argument('--num_epochs', type = int, default = 50)
    parser.add_argument('--early_stopping_patience', type = int, default = 10)
    parser.add_argument('--device', type = str, default = None)
    
    args = parser.parse_args()
    
    history = train_autoencoder(
        data_dir = args.data_dir,
        model_dir = args.model_dir,
        results_dir = args.results_dir,
        input_dim = args.input_dim,
        latent_dim = args.latent_dim,
        batch_size = args.batch_size,
        learning_rate = args.learning_rate,
        num_epochs = args.num_epochs,
        early_stopping_patience = args.early_stopping_patience,
        device = args.device
    )
    

    print(f"Validation loss: {history['best_val_loss']:.6f}")