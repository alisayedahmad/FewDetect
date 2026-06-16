import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor


class DINOv2Backbone(nn.Module):
    """DINOv2 ViT-S/14 feature extractor.
    
    Extracts CLS token or patch tokens from images.
    Shared across all three approaches for fair comparison.
    """

    def __init__(self, model_name="facebook/dinov2-small", frozen=True):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.embed_dim = self.model.config.hidden_size  # 384 for ViT-S

        if frozen:
            self.freeze()

    def freeze(self):
        """Freeze all backbone parameters."""
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze all backbone parameters."""
        for param in self.model.parameters():
            param.requires_grad = True

    def process_images(self, images):
        """Preprocess PIL images for the model.
        
        Args:
            images: single PIL image or list of PIL images
            
        Returns:
            dict with pixel_values tensor
        """
        if not isinstance(images, list):
            images = [images]
        return self.processor(images=images, return_tensors="pt")

    def forward(self, pixel_values, return_patch_tokens=False):
        """Extract features from preprocessed images.
        
        Args:
            pixel_values: tensor [B, 3, H, W] from processor
            return_patch_tokens: if True, return patch tokens instead of CLS
            
        Returns:
            CLS token [B, embed_dim] or patch tokens [B, n_patches, embed_dim]
        """
        output = self.model(pixel_values=pixel_values)
        
        if return_patch_tokens:
            return output.last_hidden_state[:, 1:, :]  # skip CLS token
        
        return output.last_hidden_state[:, 0, :]  # CLS token only

    @torch.no_grad()
    def extract_features(self, images, return_patch_tokens=False):
        """End-to-end: PIL images -> feature vectors.
        
        Args:
            images: single PIL image or list of PIL images
            return_patch_tokens: if True, return patch tokens
            
        Returns:
            features tensor
        """
        inputs = self.process_images(images)
        pixel_values = inputs["pixel_values"]
        
        if next(self.model.parameters()).is_cuda:
            pixel_values = pixel_values.cuda()
            
        return self.forward(pixel_values, return_patch_tokens)