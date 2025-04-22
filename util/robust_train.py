import os
import configparser
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch.nn.functional as F
from torchsummary import summary
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE

import torchvision.utils as vutils
import random

class ProcessedImageDataset(Dataset):
    """Dataset class for processed images."""
    
    def __init__(self, image_folder, csv_file, augmentation_type, augmentation_value, transform=None):
        """
        Args:
            image_folder (str): Path to the folder with preprocessed images.
            csv_file (str): Path to the CSV file containing labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_folder = image_folder
        self.transform = transform

        self.augmentation_value = augmentation_value  
        self.augmentation_type = augmentation_type
        
        # Read the CSV file
        self.data = pd.read_csv(csv_file)
        self.data['timestamp'] = self.data['timestamp'].apply(lambda x: str(x).replace("M_", ""))
        
        # Create a mapping from filenames to labels
        self.label_map = dict(zip(self.data['timestamp'], self.data['steering']))

        # Get the list of image filenames
        self.image_filenames = sorted([f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))])

        # Duplicate filenames for both original and augmented images
        self.image_filenames = [(f, 1) for f in self.image_filenames] + [(f, 0) for f in self.image_filenames]

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.image_filenames)

    def __getitem__(self, idx):
        """Return the image, label, and augmentation type at the specified index."""
        img_name, augment_flag = self.image_filenames[idx]
        image = Image.open(os.path.join(self.image_folder, img_name)).convert('L')  # Convert to grayscale (L mode)
        
        timestamp = img_name.replace(".png", "")  # Adjust based on actual file extension
        label = self.label_map.get(timestamp, -1)  # Default to -1 if timestamp not found
        
        augmentation_type = 0
        if augment_flag:
            image, augmentation_type = self.add_augmentations(image)

        if self.transform:
            image = self.transform(image)
        
        # Return image, label, and augmentation type (augmented or none)
        return image, label, augmentation_type

    def add_augmentations(self, image):
        if self.augmentation_type == 'mb':  
            # Convert the image to a numpy array
            image_np = np.array(image)

            # Define the vertical motion blur kernel
            degree = int(np.random.rand() * self.augmentation_value) + 1  # Random integer from 1 to 4
            kernel = np.zeros((degree, degree))
            kernel[:, int((degree - 1) / 2)] = np.ones(degree)
            kernel = kernel / degree

            # Apply the motion blur using OpenCV's filter2D function
            blurred_image = cv2.filter2D(image_np, -1, kernel)

            # Convert back to PIL image
            augmented_image = Image.fromarray(blurred_image)

            return augmented_image, degree  # Return augmented image and applied augmentation type

        else:
            # Convert the image to a numpy array
            image = np.array(image)
            # Convert back to PIL image for brightness adjustments
            image = transforms.ToPILImage()(image)  
            # Apply random brightness adjustment
            brightness_value = torch.rand(1).item() * self.augmentation_value  # Brightness adjustment in range [augmentation_value, 2*augmentation_value ]
            color_jitter = transforms.ColorJitter(brightness=brightness_value)
            augmented_image = color_jitter(image)

            return augmented_image, round(self.augmentation_value, 2)  # Return image and factor with 2 decimal places


class Decoder(nn.Module):
    def __init__(self, latent_dim, flattened_size):
        super(Decoder, self).__init__()
        self.input_layer = nn.Linear(latent_dim, flattened_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        z = self.input_layer(z)
        x = z.view(z.size(0), 256, 5, 4)
        return self.decoder(x)

class VAE(nn.Module):
    def __init__(self, latent_dim, flattened_size, flattened_size_decoder, n_gaussians=3):
        super(VAE, self).__init__()
        self.n_gaussians = n_gaussians
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.25)
        )

        self.flattened_size = flattened_size
        self.fc_mu = nn.ModuleList([nn.Linear(flattened_size, latent_dim) for _ in range(n_gaussians)])
        self.fc_log_var = nn.ModuleList([nn.Linear(flattened_size, latent_dim) for _ in range(n_gaussians)])
        self.fc_weights = nn.Linear(flattened_size, n_gaussians)
        
        self.decoder = Decoder(latent_dim, flattened_size_decoder)

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        
        # Add numerical stability
        x = torch.clamp(x, min=-1e6, max=1e6)  # Prevent extreme values
        
        mu = [mu_layer(x) for mu_layer in self.fc_mu]
        log_var = [log_var_layer(x) for log_var_layer in self.fc_log_var]
        weights = F.softmax(self.fc_weights(x), dim=-1)  # Compute softmax over Gaussian components

        # Stabilize weight logits
        weight_logits = self.fc_weights(x)
        weight_logits = torch.clamp(weight_logits, min=-50, max=50)  # Prevent overflow
        weights = F.softmax(weight_logits, dim=-1)
        
        # Add epsilon to avoid zero probabilities
        weights = weights + 1e-8
        weights = weights / weights.sum(dim=-1, keepdim=True)
        
        return mu, log_var, weights

    def reparameterize(self, mus, log_vars, weights):
        
        # Convert to double precision for sampling
        weights = weights.double()
        
        batch_size = mus[0].size(0)
        gaussian_idx = torch.multinomial(
            weights, 
            1,
            replacement=True  # Ensure valid sampling even with precision issues
        ).squeeze()
        
        # Convert back to original precision
        mu = torch.stack(mus)[gaussian_idx, torch.arange(batch_size)].float() 
        log_var = torch.stack(log_vars)[gaussian_idx, torch.arange(batch_size)].float()
        
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var, weights = self.encode(x)
        z = self.reparameterize(mu, log_var, weights)
        return self.decode(z), mu, log_var, weights

def gaussian_kl_divergence(mu, log_var):
    # KL divergence for each Gaussian component
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)

def loss_function(recon_x, x, mu, log_var, weights, beta=1e-2):
    # Reconstruction loss
    recon_loss = nn.MSELoss(reduction='sum')(recon_x, x)

    # KL divergence for each Gaussian
    kld_losses = torch.stack([gaussian_kl_divergence(m, lv) for m, lv in zip(mu, log_var)])

    # Weighted sum of KL divergences
    kld_loss = torch.sum(weights * kld_losses.T, dim=1).sum()

    # Entropy of mixture weights
    entropy = -torch.sum(weights * torch.log(weights), dim=1).sum()

    total_loss = recon_loss + beta * (kld_loss - entropy)
    return total_loss, recon_loss, kld_loss

def train(model, train_dataloader, optimizer, scheduler, device, config):
    model.train()
    train_losses = {'total': [], 'recon': [], 'kl': []}
    
    for epoch in range(config.getint('Training', 'epochs')):
        total_loss = 0
        recon_loss = 0
        kl_loss = 0
        
        with tqdm(total=len(train_dataloader), desc=f'Epoch {epoch+1}', unit='batch') as pbar:
            for batch_idx, (data, _, _) in enumerate(train_dataloader):  
                try:
                    data = data.to(device)
                    
                    # Forward pass with gradient scaling
                    with torch.cuda.amp.autocast():  # Mixed precision training
                        recon_batch, mu, log_var, weights = model(data)
                        loss, recon, kl = loss_function(
                            recon_batch, data, mu, log_var, weights,
                            config.getfloat('Training', 'beta')
                        )

                    # Gradient handling
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # Gradient clipping and NaN checks
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        max_norm=config.getfloat('Training', 'clip_grad'),
                        error_if_nonfinite=True  # Catch NaN gradients
                    )
                    
                    # Check for invalid gradients
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any():
                                raise RuntimeError(f"NaN gradients in {name}")
                                
                    optimizer.step()

                except RuntimeError as e:
                    print(f"\nError in batch {batch_idx}: {str(e)}")
                    print("Skipping problematic batch...")
                    continue

                # Update tracking
                total_loss += loss.item()
                recon_loss += recon.item()
                kl_loss += kl.item()
                
                pbar.set_postfix({
                    'Total': f"{loss.item():.4f}",
                    'Recon': f"{recon.item():.4f}", 
                    'KL': f"{kl.item():.4f}"
                })
                pbar.update(1)

        # Validation and learning rate update
        avg_total = total_loss / len(train_dataloader)
        scheduler.step(avg_total)

    return train_losses


def plot_latent_space(model, dataloader, output_dir, augmentation_type, augmentation_value):
    model.eval()
    device = next(model.parameters()).device  # Get the device from the model
    latent_vectors = []
    labels_is_noisy = []
    labels_actual = []

    with torch.no_grad():
        for batch_idx, (data, labels, is_noisy) in enumerate(dataloader):
            data = data.to(device)
            # Encode the image
            mu, log_var, weights = model.encode(data)
            z = model.reparameterize(mu, log_var, weights)
            latent_vectors.append(z.cpu().numpy())
            labels_is_noisy.extend(is_noisy.cpu().numpy())
            labels_actual.extend(labels.cpu().numpy())

    # Convert lists to numpy arrays
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    labels_is_noisy = np.array(labels_is_noisy)
    labels_actual = np.array(labels_actual)

    # Perform t-SNE dimensionality reduction to 2D
    tsne = TSNE(n_components=2, random_state=42)
    latent_tsne = tsne.fit_transform(latent_vectors)

    # Plot t-SNE with is_noisy labels
    if augmentation_type == 'mb':
        colors = ['green' if label < 2 else 'red' for label in labels_is_noisy]
    else:
        colors = ['green' if label == 0.0 else 'red' for label in labels_is_noisy]
    plt.figure(figsize=(7, 5))
    plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=colors, s=4, alpha=0.8, linewidths=0.2)
    plt.legend(handles=[mpatches.Patch(color='green', label='Clean Image'), mpatches.Patch(color='red', label='Augmented Image')], loc='upper left')
    plt.xlabel('t-SNE Component 1', fontsize=10)
    plt.ylabel('t-SNE Component 2', fontsize=10)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'TSNE_is_noisy_{augmentation_type}_{augmentation_value}.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Plot t-SNE with actual labels
    colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'purple', 'orange', 'lime', 'brown', 'black', 'magenta']
    boundaries = np.arange(-0.40, 0.40, 0.1)
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries, cmap.N, extend='both')
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=labels_actual, cmap=cmap, norm=norm, s=10, alpha=0.9, linewidths=0.01)
    cbar = plt.colorbar(scatter, boundaries=boundaries, ticks=np.arange(-0.40, 0.40, 0.1))
    cbar.set_label('Action Values ($\mathbf{a}$)')
    plt.xlabel('t-SNE Component 1', fontsize=10)
    plt.ylabel('t-SNE Component 2', fontsize=10)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'TSNE_labels_{augmentation_type}_{augmentation_value}.png'), dpi=300, bbox_inches='tight')
    plt.show()

def export_onnx(model, sample_input, save_path):
    """Export model to ONNX format"""
    print("\nExporting model to ONNX format...")
    torch.onnx.export(
        model,
        sample_input,
        save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output', 'mu', 'log_var', 'weights'],
        dynamic_axes={'input': {0: 'batch_size'}, 
                     'output': {0: 'batch_size'}}
    )
    print(f"Successfully saved ONNX model to: {save_path}")

def generate_encodings_csv(model, dataloader, output_dir, csv_filename):
    model.eval()
    device = next(model.parameters()).device  # Get the device from the model
    encoding_data = []

    with torch.no_grad():
        for img, label, _ in dataloader:  
            img = img.to(device)
            mu, log_var, weights = model.encode(img)
            z = model.reparameterize(mu, log_var, weights)

            for i in range(len(img)):
                encoding_data.append({
                    'steering': label[i].item(),
                    'latent': z[i].cpu().numpy(),
                })

    # Create DataFrame
    df = pd.DataFrame(encoding_data)
    z_df = pd.DataFrame(df['latent'].to_list(), columns=[f'latent_{i}' for i in range(df['latent'][0].shape[0])])
    final_df = pd.concat([df[['steering']], z_df], axis=1)
    
    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    final_df.to_csv(os.path.join(output_dir, csv_filename), index=False)
    print(f'Encodings saved to {os.path.join(output_dir, csv_filename)}')

def save_reconstruction(model, train_dataloader, output_dir, config, num_samples=5):
    model.eval()
    device = next(model.parameters()).device  # Get the device from the model


    with torch.no_grad():
        # Select random indices from the dataset
        sample_indices = random.sample(range(len(train_dataloader.dataset)), num_samples)
        
        original_images = []
        reconstructed_images = []
        
        for idx in sample_indices:
            # Get data for the selected index
            img, _, _ = train_dataloader.dataset[idx]
            img = img.unsqueeze(0).to(device)  # Add batch dimension and move to device
            
            # Generate reconstructed images
            recon_batch, _, _, _ = model(img)
            
            # Move data to CPU for visualization
            original_images.append(img.cpu().squeeze(0))
            reconstructed_images.append(recon_batch.cpu().squeeze(0))
        
        # Create a single plot with all images
        fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
        
        for i in range(num_samples):
            axes[0, i].imshow(original_images[i].permute(1, 2, 0), cmap='gray')
            axes[0, i].axis('off')
            axes[1, i].imshow(reconstructed_images[i].permute(1, 2, 0), cmap='gray')
            axes[1, i].axis('off')
        
        axes[0, 0].set_title("Original Images", fontsize=12)
        axes[1, 0].set_title("Reconstructed Images", fontsize=12)
        
        title = f"robust_reconstruction_{config['Augmentation']['type']}_{config['Augmentation']['value']}"
        fig.suptitle(title, fontsize=14)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.savefig(os.path.join(output_dir, f"{title}.png"))
        plt.close()



def main():
    # Read configuration
    print("Initializing training pipeline...")
    config = configparser.ConfigParser()
    config.read('GMVAE_config.ini')
    
    # Create output directories
    print("\nCreating output directories:")
    save_dir = config['Paths']['save_dir']
    plot_dir = config['Paths']['plot_dir']
    encoding_dir = config['Paths']['encoding_dir']
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(encoding_dir, exist_ok=True)
    print(f" - Model save directory: {save_dir}")
    print(f" - Plot save directory: {plot_dir}")

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Dataset preparation
    print("\nLoading dataset...")
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = ProcessedImageDataset(
        config['Paths']['image_folder'],
        config['Paths']['csv_file'],
        config['Augmentation']['type'],
        config.getfloat('Augmentation', 'value'),
        transform=transform
    )
    
    # Split dataset
    train_size = int(config.getfloat('Dataset', 'train_ratio') * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.getint('Dataset', 'batch_size'),
        shuffle=True,
        num_workers=config.getint('Dataset', 'num_workers')
    )
    print(f"Training samples: {len(train_dataset)}")

    # Model initialization
    print("\nInitializing VAE model...")
    latent_dim = config.getint('Model', 'latent_dim')
    model = VAE(
        latent_dim=latent_dim,
        flattened_size=eval(config.get('Model', 'encoder_flatten')),
        flattened_size_decoder=eval(config.get('Model', 'decoder_flatten')),
        n_gaussians=config.getint('Model', 'n_gaussians')
    ).to(device)
    print(f"Model architecture:\n{summary(model, (1, 80, 64))}")

    # Check if the ONNX model already exists
    onnx_path = os.path.join(save_dir, f"GMVAE_robust_{config['Augmentation']['type']}_{config['Augmentation']['value']}.onnx")
    torch_path = os.path.join(save_dir, f"GMVAE_robust_{config['Augmentation']['type']}_{config['Augmentation']['value']}.pt")

    # Optimizer setup
    optimizer = optim.Adam(model.parameters(), 
                            lr=config.getfloat('Training', 'lr'),weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    # Training phase
    print("\nStarting training...")
    train_losses = train(model, train_loader, optimizer, scheduler, device, config)

    # Save training curves
    loss_plot_path = os.path.join(plot_dir, 'robust_loss_curve.png')
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses['total'], label='Train Total Loss')
    plt.plot(train_losses['recon'], label='Train Reconstruction Loss')
    plt.plot(train_losses['kl'], label='Train KL Divergence Loss')
    plt.title('Training Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Saved loss curve to: {loss_plot_path}")

    # Save PyTorch model
    torch_path = os.path.join(save_dir, f"GMVAE_robust_{config['Augmentation']['type']}_{config['Augmentation']['value']}.pt")
    torch.save(model.state_dict(), torch_path)
    print(f'Saved PyTorch model to {torch_path}')

    # Export the trained model to ONNX format
    print("\nExporting model to ONNX format...")
    dummy_input = torch.randn(1, 1, 80, 64).to(device)  # Example input shape
    export_onnx(model, dummy_input, onnx_path)

    if config['Augmentation']['type'] == 'mb':
        # Generate latent space visualization
        print("\nGenerating latent space visualization...")
        plot_latent_space(model, train_loader, plot_dir, config['Augmentation']['type'], config['Augmentation']['value'])
        
    # Generate encodings CSV
    print("\nGenerating encodings CSV...")
    csv_filename = f"encodings_{latent_dim}_robust_{config['Augmentation']['type']}_{config['Augmentation']['value']}.csv"
    generate_encodings_csv(model, train_loader, encoding_dir, csv_filename)

    # Save the reconstructed images in a seperate directory
    save_reconstruction(model,train_loader,plot_dir,config,num_samples = 5)

    print("\nTraining and post-processing complete!")


    test_loader = DataLoader(
        test_dataset,
        batch_size=config.getint('Dataset', 'batch_size'),
        shuffle=False,
        num_workers=config.getint('Dataset', 'num_workers')
    )

    model.eval()
    test_mse = 0
    with torch.no_grad():
        for batch_idx, (data,_,_) in enumerate(test_loader):
            data = data.to(device)
            outputs,_,_,_ = model(data)
            test_mse += torch.mean((outputs - data) ** 2)

    test_mse /= len(test_loader)
    print(f"Test MSE: {test_mse.item():.4f}")


if __name__ == "__main__":
    main()
