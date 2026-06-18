import torch
import torch.nn as nn
from backbone.dinov2 import DINOv2Backbone


class MAMLClassifier(nn.Module):
    """Classifier head for MAML on top of frozen DINOv2.
    
    DINOv2 stays frozen 
    Only the classifier head gets adapted in the inner loop. This is the standard approach for MAML
    with large pretrained backbones
    """

    def __init__(self, embed_dim=384, n_way=5):
        super().__init__()
        self.backbone = DINOv2Backbone(frozen=True)
        self.embed_dim = embed_dim

        # Classifier head — this is what MAML adapts
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_way),
        )

    def forward_features(self, images):
        """Extract features from PIL images. No grad needed  """
        return self.backbone.extract_features(images)

    def forward(self, features):
        """Classify from pre-extracted features.
        
        Args:
            features: tensor [B, embed_dim]
            
        Returns:
            logits: tensor [B, n_way]
        """
        return self.head(features)