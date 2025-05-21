# Training diffusion generator
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.generator import DiffusionModel
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.data.peptide_dataset import PeptideDataset

def train_diffusion(model, dataloader, optimizer, device, epochs=10, num_classes=24, save_path='diffusion_model.pth'):
    model.train()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, padding_mask) in tqdm(enumerate(dataloader), total=len(dataloader)):
            data = data.to(device)
            padding_mask = padding_mask.to(device)
            data = F.one_hot(data, num_classes=num_classes).float()
            
            t = torch.randint(0, model.T, (data.shape[0],), device=device)
            noisy_data, noise = model.add_noise(data, t)
            predicted_noise = model(noisy_data, t, padding_mask=padding_mask)
            
            loss = F.mse_loss(predicted_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
        scheduler.step(avg_loss)
    
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_dataset = PeptideDataset('data/peptides.csv')
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    num_classes = train_dataset.vocab_size

    diffusion_model = DiffusionModel(num_classes=num_classes, device=device)
    diffusion_optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=1e-4)
    
    print("Training diffusion model...")
    train_diffusion(diffusion_model, train_loader, diffusion_optimizer, device, epochs=100, num_classes=num_classes, save_path='./ckpt/diffusion.pth')