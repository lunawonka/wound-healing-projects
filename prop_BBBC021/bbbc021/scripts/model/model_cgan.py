import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, num_classes=13, latent_dim=128, cond_dim=128, img_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Embed the discrete MoA label into a continuous vector
        self.label_emb = nn.Embedding(num_classes, cond_dim)
        
        # The input to the first spatial layer will be latent_dim + cond_dim
        self.init_size = 4  # We start generating at 4x4 spatial resolution
        self.l1 = nn.Sequential(nn.Linear(latent_dim + cond_dim, 1024 * self.init_size * self.init_size))
        
        # Upsampling blocks: 4 -> 8 -> 16 -> 32 -> 64 -> 128 -> 256
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(1024),
            
            # 4x4 -> 8x8
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 64x64 -> 128x128
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 128x128 -> 256x256
            nn.ConvTranspose2d(32, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh() # Output is scaled to [-1, 1] to match your transforms
        )

    def forward(self, z, y):
        # Combine noise and label embedding
        y_emb = self.label_emb(y)
        gen_input = torch.cat((z, y_emb), dim=1)
        
        # Project and reshape into a spatial tensor
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 1024, self.init_size, self.init_size)
        
        # Upsample into final image
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, num_classes=13, img_channels=3, img_size=256):
        super().__init__()
        self.img_size = img_size
        
        # We embed the condition label into a spatial channel (1 x 256 x 256)
        # so we can concatenate it directly to the image channels (3 + 1 = 4 channels)
        self.label_emb = nn.Embedding(num_classes, img_size * img_size)
        
        # Downsampling blocks: 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.Dropout2d(0.25)] # Dropout is vital for GAN stability
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(img_channels + 1, 32, bn=False), # No BN on first layer
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            *discriminator_block(512, 1024),
        )

        # The output spatial dimension is 4x4 after 6 downsampling steps
        ds_size = img_size // (2 ** 6) # 256 / 64 = 4
        self.adv_layer = nn.Linear(1024 * ds_size * ds_size, 1)

    def forward(self, img, y):
        # Expand the label embedding into an extra image channel
        y_spatial = self.label_emb(y).view(y.size(0), 1, self.img_size, self.img_size)
        
        # Concatenate condition channel to image (now 4 channels)
        d_in = torch.cat((img, y_spatial), dim=1)
        
        out = self.model(d_in)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        
        return validity # We output raw logits (BCEWithLogitsLoss will handle the Sigmoid)
        