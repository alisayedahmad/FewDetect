import torch
from backbone.dinov2 import DINOv2Backbone
from approaches.prototypical.prototype_builder import PrototypeBuilder


class PrototypicalDetector:
    """Full prototypical network pipeline: images in, predictions out.
    
    Connects DINOv2 backbone with prototype-based classification.
    """

    def __init__(self, distance="euclidean", frozen_backbone=True):
        self.backbone = DINOv2Backbone(frozen=frozen_backbone)
        self.builder = PrototypeBuilder(distance=distance)

    def _extract_support_features(self, support_images):
        """Extract features for each class in the support set.
        
        Args:
            support_images: list of lists of PIL images
                support_images[i] = list of K PIL images for class i
                
        Returns:
            list of tensors, each [k_shot, embed_dim]
        """
        support_features = []
        for class_images in support_images:
            features = self.backbone.extract_features(class_images)
            support_features.append(features)
        return support_features

    def _extract_query_features(self, query_images):
        """Extract features for all query images.
        
        Args:
            query_images: list of PIL images
            
        Returns:
            tensor [n_queries, embed_dim]
        """
        return self.backbone.extract_features(query_images)

    @torch.no_grad()
    def predict(self, support_images, query_images):
        """End-to-end prediction: PIL images -> class predictions.
        
        Args:
            support_images: list of lists of PIL images
                support_images[i] = K images for class i
            query_images: list of PIL images to classify
            
        Returns:
            predictions: tensor [n_queries] with class indices
            probabilities: tensor [n_queries, n_way]
            prototypes: tensor [n_way, embed_dim]
        """
        support_features = self._extract_support_features(support_images)
        prototypes = self.builder.build_prototypes(support_features)
        query_features = self._extract_query_features(query_images)
        predictions, probabilities = self.builder.classify(query_features, prototypes)

        return predictions, probabilities, prototypes

    def evaluate_episode(self, support_images, query_images, query_labels):
        """Run one episode and return accuracy.
        
        Args:
            support_images: list of lists of PIL images
            query_images: list of PIL images
            query_labels: tensor [n_queries] with ground truth class indices
            
        Returns:
            accuracy: float between 0 and 1
            predictions: tensor [n_queries]
        """
        predictions, _, _ = self.predict(support_images, query_images)
        correct = (predictions == query_labels).sum().item()
        accuracy = correct / len(query_labels)
        return accuracy, predictions