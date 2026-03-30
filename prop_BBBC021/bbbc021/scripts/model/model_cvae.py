import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalVAE(nn.Module):
    def __init__(self, num_classes=13, img_channels=3, latent_dim=128, cond_dim=32):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim

        self.label_emb = nn.Embedding(num_classes, cond_dim)

        # Encoder: 256 -> 128 -> 64 -> 32 -> 16 -> 8
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, 2, 1),   # 256 -> 128
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, 4, 2, 1),             # 128 -> 64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),            # 64 -> 32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),           # 32 -> 16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),           # 16 -> 8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.enc_out_dim = 512 * 8 * 8

        self.fc_mu = nn.Linear(self.enc_out_dim + cond_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_out_dim + cond_dim, latent_dim)

        self.fc_dec = nn.Linear(latent_dim + cond_dim, 512 * 8 * 8)

        # Decoder: 8 -> 16 -> 32 -> 64 -> 128 -> 256
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 8 -> 16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 16 -> 32
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 32 -> 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 64 -> 128
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, img_channels, 4, 2, 1),  # 128 -> 256
            nn.Tanh()
        )

    def encode(self, x, y):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        y_emb = self.label_emb(y)
        h = torch.cat([h, y_emb], dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        y_emb = self.label_emb(y)
        z = torch.cat([z, y_emb], dim=1)
        h = self.fc_dec(z)
        h = h.view(h.size(0), 512, 8, 8)
        x_hat = self.decoder(h)
        return x_hat

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, y)
        return x_hat, mu, logvar

    def sample(self, y, device):
        z = torch.randn(len(y), self.latent_dim, device=device)
        return self.decode(z, y)
        