import random
import torch
from torch.utils.data import Dataset


class EpisodeSampler:
    """Génère des épisodes N-way K-shot depuis un dataset."""

    def __init__(self, dataset, n_way=5, k_shot=5, n_query=15, seed=None):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.rng = random.Random(seed)

        # Organiser les indices par classe
        self.class_to_indices = {}
        for i in range(len(dataset)):
            _, label = dataset[i]
            if label not in self.class_to_indices:
                self.class_to_indices[label] = []
            self.class_to_indices[label].append(i)

        # Garder que les classes avec assez d'images
        min_images = k_shot + n_query
        self.valid_classes = [
            cls for cls, indices in self.class_to_indices.items()
            if len(indices) >= min_images
        ]

        if len(self.valid_classes) < n_way:
            raise ValueError(
                f"Pas assez de classes valides: {len(self.valid_classes)} < {n_way}"
            )

    def sample_episode(self):
        """Retourne un épisode: (support_indices, query_indices, chosen_classes)"""
        chosen_classes = self.rng.sample(self.valid_classes, self.n_way)

        support_indices = []
        query_indices = []

        for cls in chosen_classes:
            indices = self.rng.sample(
                self.class_to_indices[cls],
                self.k_shot + self.n_query
            )
            support_indices.append(indices[:self.k_shot])
            query_indices.append(indices[self.k_shot:])

        return support_indices, query_indices, chosen_classes

    def __repr__(self):
        return (
            f"EpisodeSampler({self.n_way}-way {self.k_shot}-shot, "
            f"{self.n_query} queries, {len(self.valid_classes)} classes)"
        )