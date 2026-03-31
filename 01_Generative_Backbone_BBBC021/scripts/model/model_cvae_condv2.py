import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalVAECondV2(nn.Module):
    def __init__(self, num_classes=13, img_channels=3, latent_dim=128, cond_dim=32):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim

        self.label_emb = nn.Embedding(num_classes, cond_dim)

        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.enc_out_dim = 512 * 8 * 8

        self.fc_mu = nn.Linear(self.enc_out_dim + cond_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_out_dim + cond_dim, latent_dim)

        self.fc_dec = nn.Linear(latent_dim + cond_dim, 512 * 8 * 8)

        self.cond_to_map_8 = nn.Linear(cond_dim, 8 * 8)
        self.cond_to_map_16 = nn.Linear(cond_dim, 16 * 16)
        self.cond_to_map_32 = nn.Linear(cond_dim, 32 * 32)
        self.cond_to_map_64 = nn.Linear(cond_dim, 64 * 64)
        self.cond_to_map_128 = nn.Linear(cond_dim, 128 * 128)

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(513, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(257, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(129, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(65, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(33, img_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def encode(self, x, y):
        h = self.encoder(x).view(x.size(0), -1)
        y_emb = self.label_emb(y)
        h = torch.cat([h, y_emb], dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _cond_map(self, y_emb, linear, size):
        m = linear(y_emb).view(y_emb.size(0), 1, size, size)
        return m

    def decode(self, z, y):
        y_emb = self.label_emb(y)
        h = self.fc_dec(torch.cat([z, y_emb], dim=1)).view(z.size(0), 512, 8, 8)

        h = torch.cat([h, self._cond_map(y_emb, self.cond_to_map_8, 8)], dim=1)
        h = self.dec1(h)

        h = torch.cat([h, self._cond_map(y_emb, self.cond_to_map_16, 16)], dim=1)
        h = self.dec2(h)

        h = torch.cat([h, self._cond_map(y_emb, self.cond_to_map_32, 32)], dim=1)
        h = self.dec3(h)

        h = torch.cat([h, self._cond_map(y_emb, self.cond_to_map_64, 64)], dim=1)
        h = self.dec4(h)

        h = torch.cat([h, self._cond_map(y_emb, self.cond_to_map_128, 128)], dim=1)
        h = self.dec5(h)

        return h

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, y)
        return x_hat, mu, logvar

    def sample(self, y, device):
        z = torch.randn(len(y), self.latent_dim, device=device)
        return self.decode(z, y)
        