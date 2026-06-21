import torch
import torch.nn as nn
import torch.nn.functional as F


class LightFPN(nn.Module):
    """Lightweight Feature Pyramid Network on top of DINOv2 patch tokens.
    
    Takes patch tokens [B, n_patches, embed_dim] from DINOv2 and produces
    multi-scale feature maps at 3 levels:
        P3: 16x16 (stride 14) — small objects
        P4: 8x8   (stride 28) — medium objects  
        P5: 4x4   (stride 56) — large objects
    
    All output channels are fpn_dim (default 256).
    """

    def __init__(self, embed_dim=384, fpn_dim=256, grid_size=16):
        super().__init__()
        self.grid_size = grid_size
        self.fpn_dim = fpn_dim

        # Lateral connections: project embed_dim -> fpn_dim
        self.lateral_p3 = nn.Conv2d(embed_dim, fpn_dim, 1)
        self.lateral_p4 = nn.Conv2d(embed_dim, fpn_dim, 1)
        self.lateral_p5 = nn.Conv2d(embed_dim, fpn_dim, 1)

        # Downsampling to create P4 and P5 from patch tokens
        self.downsample_p4 = nn.Conv2d(embed_dim, embed_dim, 3, stride=2, padding=1)
        self.downsample_p5 = nn.Conv2d(embed_dim, embed_dim, 3, stride=2, padding=1)

        # Smooth convolutions after merging
        self.smooth_p3 = nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1)
        self.smooth_p4 = nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1)
        self.smooth_p5 = nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1)

    def forward(self, patch_tokens):
        """Build feature pyramid from patch tokens.
        
        Args:
            patch_tokens: tensor [B, n_patches, embed_dim] from DINOv2
            
        Returns:
            dict with keys 'p3', 'p4', 'p5', each a tensor [B, fpn_dim, H, W]
        """
        B, N, C = patch_tokens.shape

        # Reshape to 2D feature map: [B, C, grid, grid]
        feat = patch_tokens.transpose(1, 2).reshape(B, C, self.grid_size, self.grid_size)

        # Create multi-scale features by downsampling
        c3 = feat                          # 16x16
        c4 = self.downsample_p4(c3)        # 8x8
        c5 = self.downsample_p5(c4)        # 4x4

        # Lateral connections
        p5 = self.lateral_p5(c5)           # 4x4, fpn_dim
        p4 = self.lateral_p4(c4)           # 8x8, fpn_dim
        p3 = self.lateral_p3(c3)           # 16x16, fpn_dim

        # Top-down pathway: upsample and add
        p4 = p4 + F.interpolate(p5, size=p4.shape[2:], mode='nearest')
        p3 = p3 + F.interpolate(p4, size=p3.shape[2:], mode='nearest')

        # Smooth
        p3 = self.smooth_p3(p3)
        p4 = self.smooth_p4(p4)
        p5 = self.smooth_p5(p5)

        return {'p3': p3, 'p4': p4, 'p5': p5}