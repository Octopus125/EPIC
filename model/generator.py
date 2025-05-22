import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual) 
        out = self.relu(out)
        return out

class Unet1D(nn.Module):
    def __init__(self, vocab_size=32, time_embedding_size=8, hidden_size=768):
        super().__init__()
        self.vocab_size = vocab_size
        
        self.embedding = nn.Linear(vocab_size + time_embedding_size, hidden_size)
        
        self.encoder = nn.Sequential(
            ResidualBlock1D(hidden_size, hidden_size),  
            nn.MaxPool1d(2),  
            ResidualBlock1D(hidden_size, hidden_size * 2),  
            nn.MaxPool1d(2),  
            ResidualBlock1D(hidden_size * 2, hidden_size * 4),  
            nn.MaxPool1d(2)  
        )
        

        self.middle = nn.Sequential(
            ResidualBlock1D(hidden_size * 4, hidden_size * 4),  
            ResidualBlock1D(hidden_size * 4, hidden_size * 4)  
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden_size * 4, hidden_size * 2, kernel_size=2, stride=2),  
            ResidualBlock1D(hidden_size * 2, hidden_size * 2),  
            nn.ConvTranspose1d(hidden_size * 2, hidden_size, kernel_size=2, stride=2),  
            ResidualBlock1D(hidden_size, hidden_size),  
            nn.ConvTranspose1d(hidden_size, hidden_size, kernel_size=2, stride=2),  
            ResidualBlock1D(hidden_size, hidden_size)  
        )
        
        
        self.output = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, padding_mask=None):
        x = x.float()
        
        embeddings = self.embedding(x)
        
        # [batch, seq_len, hidden] -> [batch, hidden, seq_len]
        embeddings = embeddings.transpose(1, 2)
        
        encoded = self.encoder(embeddings)
        
        middle = self.middle(encoded)
        
        decoded = self.decoder(middle)
        
        # [batch, hidden, seq_len] -> [batch, seq_len, hidden]
        decoded = decoded.transpose(1, 2)
        
        logits = self.output(decoded)
        return logits
    
    
class DiffusionModel(nn.Module):
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, num_classes=24, device='cpu'):
        super().__init__()
        self.T = T
        self.device = device
        self.register_buffer('betas', torch.linspace(beta_start, beta_end, T, device=device))
        self.register_buffer('alphas', 1. - self.betas)
        self.register_buffer('alpha_bars', torch.cumprod(self.alphas, dim=0))
        self.bert_denoiser = Unet1D(vocab_size=num_classes).to(device)
        self.num_classes = num_classes
        
    def forward(self, x, t, padding_mask=None):
        t_emb = self.get_time_embedding(t)
        t_emb = t_emb.unsqueeze(1).expand(-1, x.size(1), -1)  
        x = torch.cat([x, t_emb*0.1], dim=-1)
        return self.bert_denoiser(x)
        
    def get_time_embedding(self, t):
        t = t.float()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, 8, 2).float() / 8)).to(t.device)
        pos_enc = torch.zeros(t.shape[0], 8).to(t.device)
        pos_enc[:, 0::2] = torch.sin(t.unsqueeze(1) * inv_freq)
        pos_enc[:, 1::2] = torch.cos(t.unsqueeze(1) * inv_freq)
        return pos_enc
        
    def add_noise(self, x_0, t):
        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[t]).view(-1, 1, 1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1. - self.alpha_bars[t]).view(-1, 1, 1)
        noise = torch.randn_like(x_0)
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        return x_t, noise
        
    def denoise_to_x0(self, x_t, t, model):
        with torch.no_grad():
            t_tensor = torch.full((x_t.shape[0],), t, device=x_t.device)
            predicted_noise = model(x_t, t_tensor)
            sqrt_alpha_bar = torch.sqrt(self.alpha_bars[t])
            sqrt_one_minus_alpha_bar = torch.sqrt(1. - self.alpha_bars[t])
            x_0 = (x_t - sqrt_one_minus_alpha_bar * predicted_noise) / sqrt_alpha_bar
        return x_0
    
    def add_noise_step(self, x_t, t):
        if t >= self.T - 1:
            raise ValueError(f"t must be less than T-1, got {t}")
        
        sqrt_beta = torch.sqrt(self.betas[t+1]).view(-1, 1, 1)
        sqrt_one_minus_beta = torch.sqrt(1. - self.betas[t+1]).view(-1, 1, 1)
        
        noise = torch.randn_like(x_t)
        x_t1 = sqrt_one_minus_beta * x_t + sqrt_beta * noise
        
        return x_t1
    
    def add_noise_several_steps(self, x_t, t, steps):
        if t >= self.T - steps:
            raise ValueError(f"t must be less than T-1, got {t}")
        
        sqrt_alpha_bars_ratio = torch.sqrt(self.alpha_bars[t+steps] / self.alpha_bars[t]).view(-1, 1, 1)
        sqrt_one_minus_alpha_bars_ratio = torch.sqrt(1. - self.alpha_bars[t+steps] / self.alpha_bars[t]).view(-1, 1, 1)

        noise = torch.randn_like(x_t)
        x_t_steps = sqrt_alpha_bars_ratio * x_t + sqrt_one_minus_alpha_bars_ratio * noise
        return x_t_steps
    