import torch
import torch.nn as nn
import torch.nn.functional as F

class Scale(nn.Module):

    
    def __init__(self, init_value=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_value))

    def forward(self, x):
        return x * self.scale

class FCOSHead(nn.Module):
    """FCOS detection head adapted for few-shot with prototype classification.
    
    For each spatial point in the FPN feature maps, predicts:
    - class scores (via distance to prototypes or linear layer)
    - bounding box (left, top, right, bottom distances)
    - centerness score
    """

    def __init__(self, fpn_dim=256, n_classes=None, num_convs=4):
        super().__init__()
        self.fpn_dim = fpn_dim
        self.n_classes = n_classes

        # Shared classification tower
        cls_layers = []
        for _ in range(num_convs):
            cls_layers.append(nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1))
            cls_layers.append(nn.GroupNorm(32, fpn_dim))
            cls_layers.append(nn.ReLU())
        self.cls_tower = nn.Sequential(*cls_layers)

        # Shared regression tower
        reg_layers = []
        for _ in range(num_convs):
            reg_layers.append(nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1))
            reg_layers.append(nn.GroupNorm(32, fpn_dim))
            reg_layers.append(nn.ReLU())
        self.reg_tower = nn.Sequential(*reg_layers)

        # Regression head: 4 distances (l, t, r, b)
        self.bbox_pred = nn.Conv2d(fpn_dim, 4, 3, padding=1)

        # Centerness head
        self.centerness = nn.Conv2d(fpn_dim, 1, 3, padding=1)

        # Classification head — only used in non-few-shot mode
        if n_classes is not None:
            self.cls_logits = nn.Conv2d(fpn_dim, n_classes, 3, padding=1)

        # Learnable scale per FPN level for bbox regression
        self.scales = nn.ModuleList([Scale() for _ in range(3)])

        self._init_weights()

    def _init_weights(self):
        """Initialize with proper defaults."""
        for modules in [self.cls_tower, self.reg_tower]:
            for layer in modules:
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, std=0.01)
                    nn.init.constant_(layer.bias, 0)

        nn.init.normal_(self.bbox_pred.weight, std=0.01)
        nn.init.constant_(self.bbox_pred.bias, 0)

        nn.init.normal_(self.centerness.weight, std=0.01)
        nn.init.constant_(self.centerness.bias, 0)

        if self.n_classes is not None:
            nn.init.normal_(self.cls_logits.weight, std=0.01)
            nn.init.constant_(self.cls_logits.bias, -4.0)  # low initial confidence

    def forward_single_level(self, feature, scale, prototypes=None):
        """Process one FPN level 
        
        Args:
            feature: tensor [B, fpn_dim, H, W]
            scale: Scale module for this level
            prototypes: optional [n_way, fpn_dim] for few-shot classification
            
        Returns:
            cls_score: [B, n_classes, H, W] or [B, n_way, H, W]
            bbox_pred: [B, 4, H, W]
            centerness: [B, 1, H, W]
        """
        # Classification branch
        cls_feat = self.cls_tower(feature)

        if prototypes is not None:
            # Few-shot mode: classify by distance to prototypes
            cls_score = self._prototype_classify(cls_feat, prototypes)
        else:
            # Standard mode: use linear classification layer
            cls_score = self.cls_logits(cls_feat)

        # Regression branch
        reg_feat = self.reg_tower(feature)
        bbox_pred = scale(self.bbox_pred(reg_feat)).exp()
        centerness = self.centerness(reg_feat).sigmoid()

        return cls_score, bbox_pred, centerness

    def _prototype_classify(self, cls_feat, prototypes):
        """Classify each spatial location by cosine similarity to prototypes.
        
        Args:
            cls_feat: [B, fpn_dim, H, W] features from cls tower
            prototypes: [n_way, fpn_dim] class prototypes
            
        Returns:
            scores: [B, n_way, H, W] similarity scores
        """
        B, C, H, W = cls_feat.shape
        n_way = prototypes.shape[0]

        # Reshape features to [B, H*W, C]
        feat_flat = cls_feat.permute(0, 2, 3, 1).reshape(B * H * W, C)

        # Cosine similarity
        feat_norm = F.normalize(feat_flat, dim=1)
        proto_norm = F.normalize(prototypes, dim=1)
        similarity = feat_norm @ proto_norm.t()  # [B*H*W, n_way]

        # Reshape back to spatial
        scores = similarity.reshape(B, H, W, n_way).permute(0, 3, 1, 2)

        return scores

    def forward(self, fpn_features, prototypes=None):
        levels = ['p3', 'p4', 'p5']
        all_cls = []
        all_bbox = []
        all_center = []

        for i, level in enumerate(levels):
            cls_score, bbox_pred, centerness = self.forward_single_level(
                fpn_features[level], self.scales[i], prototypes
            )
            all_cls.append(cls_score)
            all_bbox.append(bbox_pred)
            all_center.append(centerness)

        return {
            'cls_scores': all_cls,
            'bbox_preds': all_bbox,
            'centerness': all_center,
        }