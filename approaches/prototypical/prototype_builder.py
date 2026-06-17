import torch
import torch.nn.functional as F




class PrototypeBuilder:
    """Builds class prototypes and classifies queries by distance.
    
    This is the core of Prototypical Networks (Snell et al., 2017).
    """

    def __init__(self, distance="euclidean"):
        if distance not in ("euclidean", "cosine"):
            raise ValueError(f"Unknown distance: {distance}")
        self.distance = distance

    def build_prototypes(self, support_features):
        """Compute one prototype per class as the mean of support embeddings.
        
        Args:
            support_features: list of tensors, one per class
                each tensor has shape [k_shot, embed_dim]
                
        Returns:
            prototypes: tensor [n_way, embed_dim]
        """
        prototypes = torch.stack([f.mean(dim=0) for f in support_features])
        return prototypes

    def compute_distances(self, queries, prototypes):
        """Compute distances between each query and each prototype.
        
        Args:
            queries: tensor [n_queries, embed_dim]
            prototypes: tensor [n_way, embed_dim]
            
        Returns:
            distances: tensor [n_queries, n_way] (lower = closer)
        """
        if self.distance == "euclidean":
            return torch.cdist(queries, prototypes)
        
        # Cosine: convert similarity to distance
        queries_norm = F.normalize(queries, dim=1)
        protos_norm = F.normalize(prototypes, dim=1)
        similarity = queries_norm @ protos_norm.t()
        return 1 - similarity

    def classify(self, queries, prototypes):
        """Predict class for each query based on nearest prototype.
        
        Args:
            queries: tensor [n_queries, embed_dim]
            prototypes: tensor [n_way, embed_dim]
            
        Returns:
            predictions: tensor [n_queries] with class indices
            probabilities: tensor [n_queries, n_way] softmax probabilities
        """
        distances = self.compute_distances(queries, prototypes)
        probabilities = F.softmax(-distances, dim=1)
        predictions = distances.argmin(dim=1)
        return predictions, probabilities