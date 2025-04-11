import torch.nn as nn
import torch

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=64):
        super().__init__()
        
        # Encoder (down)
        self.enc1 = self.contract_block(in_channels, base_channels, 4, 2, 1)
        self.enc2 = self.contract_block(base_channels, base_channels*2, 4, 2, 1)
        self.enc3 = self.contract_block(base_channels*2, base_channels*4, 4, 2, 1)
        self.enc4 = self.contract_block(base_channels*4, base_channels*8, 4, 2, 1)
        
        # Bottleneck
        self.bottleneck = self.contract_block(base_channels*8, base_channels*8, 4, 2, 1)
        
        # Decoder (up)
        self.dec4 = self.expand_block(base_channels*8, base_channels*8, 4, 2, 1)
        self.dec3 = self.expand_block(base_channels*8*2, base_channels*4, 4, 2, 1)
        self.dec2 = self.expand_block(base_channels*4*2, base_channels*2, 4, 2, 1)
        self.dec1 = self.expand_block(base_channels*2*2, base_channels, 4, 2, 1)
        
        # Final
        self.final = nn.Sequential(
            nn.ConvTranspose2d(base_channels*2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # for a binary mask
        )
    
    def contract_block(self, in_ch, out_ch, kernel_size, stride, padding):
        block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )
        return block
    
    def expand_block(self, in_ch, out_ch, kernel_size, stride, padding):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        return block
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)  # b,64,h/2,w/2
        e2 = self.enc2(e1) # b,128,h/4,w/4
        e3 = self.enc3(e2) # b,256,h/8,w/8
        e4 = self.enc4(e3) # b,512,h/16,w/16
        
        # Bottleneck
        b  = self.bottleneck(e4)  # b,512,h/32,w/32
        
        # Decoder
        d4 = self.dec4(b)         # b,512,h/16,w/16
        d4 = torch.cat([d4, e4], dim=1)  # skip connection
        
        d3 = self.dec3(d4)        # b,256,h/8,w/8
        d3 = torch.cat([d3, e3], dim=1)
        
        d2 = self.dec2(d3)        # b,128,h/4,w/4
        d2 = torch.cat([d2, e2], dim=1)
        
        d1 = self.dec1(d2)        # b,64,h/2,w/2
        d1 = torch.cat([d1, e1], dim=1)
        
        out = self.final(d1)      # b,1,h,w
        return out
