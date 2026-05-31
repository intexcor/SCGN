import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import argparse
from pathlib import Path

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        return x + self.bn2(F.leaky_relu(self.conv2(F.leaky_relu(self.bn1(self.conv1(x)), 0.2)), 0.2))

class JointSphereGenerator(nn.Module):
    def __init__(self, z_dim=128, cond_dim=64, img_channels=1):
        super().__init__()
        self.img_channels = img_channels
        self.fc = nn.Sequential(
            nn.Linear(z_dim + cond_dim, 256 * 4 * 4),
            nn.LeakyReLU(0.2)
        )
        self.conv = nn.Sequential(
            ResBlock(256),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            ResBlock(128),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            ResBlock(64),
            nn.ConvTranspose2d(64, img_channels, 4, 2, 1)
        )
        self.radius = (32 * 32 * img_channels) ** 0.5 

    def forward(self, z, cond):
        z_cond = torch.cat([z, cond], dim=1)
        z_cond = F.normalize(z_cond, p=2, dim=1)
        
        x = self.fc(z_cond)
        x = x.view(-1, 256, 4, 4)
        x = self.conv(x)
        x_flat = x.view(x.shape[0], -1)
        
        # Projection to the Image Hypersphere
        x_norm = F.normalize(x_flat, p=2, dim=1) * self.radius
        return x_norm.view_as(x)

def get_dataset(name):
    transform = transforms.Compose([
        transforms.Resize(32), 
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) if name != 'cifar' else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    if name == 'mnist':
        return torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform), 1
    elif name == 'fmnist':
        return torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform), 1
    elif name == 'cifar':
        return torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform), 3
    else:
        raise ValueError("Unknown dataset")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fmnist', 'cifar'])
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--gamma', type=float, default=5.0, help="Condition matching weight")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset, channels = get_dataset(args.dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    
    z_dim = 128
    cond_dim = 64
    
    # Emulate continuous embeddings (e.g., CLIP text embeddings)
    dummy_text_embeddings = F.normalize(torch.randn(10, cond_dim, device=device), p=2, dim=1)
    
    model = JointSphereGenerator(z_dim, cond_dim, channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-4) 
    
    out_dir = Path(f"scgn_{args.dataset}_runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting Joint Space Chamfer Training on {args.dataset.upper()}...")
    radius = (32 * 32 * channels) ** 0.5
    
    for step in range(args.steps + 1):
        try:
            real, labels = next(data_iter)
        except:
            data_iter = iter(dataloader)
            real, labels = next(data_iter)
            
        real = real.to(device)
        labels = labels.to(device)
        B = real.shape[0]
        
        real_flat = real.view(B, -1)
        real_img_norm = F.normalize(real_flat, p=2, dim=1)
        real_cond = dummy_text_embeddings[labels]
        
        # Unpaired random conditional generation
        gen_labels = labels[torch.randperm(B)] 
        gen_cond = dummy_text_embeddings[gen_labels]
        
        z = torch.randn(B, z_dim, device=device)
        z = F.normalize(z, p=2, dim=1)
        
        gen = model(z, gen_cond)
        gen_flat = gen.view(B, -1)
        gen_img_norm = F.normalize(gen_flat, p=2, dim=1)
        
        # Joint Space Chamfer Distance
        sim_img = torch.mm(gen_img_norm, real_img_norm.t())
        sim_cond = torch.mm(gen_cond, real_cond.t())        
        
        C = (1.0 - sim_img) + args.gamma * (1.0 - sim_cond)
        
        loss_gen_to_real = C.min(dim=1)[0].mean()
        loss_real_to_gen = C.min(dim=0)[0].mean()
        
        loss = loss_gen_to_real + loss_real_to_gen
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 500 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
            
        if step % 5000 == 0 or step == args.steps:
            with torch.no_grad():
                test_z = torch.randn(100, z_dim, device=device)
                test_z = F.normalize(test_z, p=2, dim=1)
                test_labels = torch.arange(10, device=device).repeat_interleave(10)
                test_cond = dummy_text_embeddings[test_labels]
                
                samples = model(test_z, test_cond)
                samples = (samples * 0.5 + 0.5).clamp(0, 1)
                vutils.save_image(samples, out_dir / f"samples_step_{step:05d}.png", nrow=10)

if __name__ == "__main__":
    main()
