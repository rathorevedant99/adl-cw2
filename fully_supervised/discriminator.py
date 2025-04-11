import torch.nn as nn

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3+1, base_channels=64):
        """
        in_channels = 3 (RGB image) + 1 (segmentation mask)
        """
        super().__init__()
        
        def conv_block(in_ch, out_ch, kernel_size, stride, padding, norm=True):
            layers = [nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.main = nn.Sequential(
            # (in_channels) -> (base_channels)
            *conv_block(in_channels, base_channels, 4, 2, 1, norm=False),
            # (base_channels) -> (base_channels*2)
            *conv_block(base_channels, base_channels*2, 4, 2, 1),
            # (base_channels*2) -> (base_channels*4)
            *conv_block(base_channels*2, base_channels*4, 4, 2, 1),
            # (base_channels*4) -> (base_channels*8)
            *conv_block(base_channels*4, base_channels*8, 4, 1, 1),
            nn.Conv2d(base_channels*8, 1, kernel_size=4, stride=1, padding=1)  # output patch
        )
        
    def forward(self, x):
        return self.main(x)
