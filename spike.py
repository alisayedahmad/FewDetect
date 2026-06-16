import torch
from transformers import AutoModel, AutoImageProcessor
from torchvision import datasets



import random
import torch.nn.functional as F


processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
backbone = AutoModel.from_pretrained("facebook/dinov2-small")
backbone.eval()

cifar = datasets.CIFAR100(root="./data_cifar", train=True, download=True)
img, label = cifar[0]

inputs = processor(images=img, return_tensors="pt")
with torch.no_grad():
    output = backbone(**inputs)
cls_token = output.last_hidden_state[:, 0, :]
print(f"Feature shape: {cls_token.shape}")
print(f"Label: {label}")

# --- Étape 2 : construire un épisode ---

# Organiser CIFAR-100 par classe
class_to_images = {}
for i in range(len(cifar)):
    img, label = cifar[i]
    if label not in class_to_images:
        class_to_images[label] = []
    class_to_images[label].append(i)

# Choisir 5 classes random
N_WAY = 5
K_SHOT = 5
N_QUERY = 15

chosen_classes = random.sample(list(class_to_images.keys()), N_WAY)
print(f"\nClasses choisies: {chosen_classes}")

# Pour chaque classe, prendre K support + N query
support_indices = []
query_indices = []
for cls in chosen_classes:
    indices = random.sample(class_to_images[cls], K_SHOT + N_QUERY)
    support_indices.append(indices[:K_SHOT])
    query_indices.append(indices[K_SHOT:])

# Extraire les features de toutes les images
def extract_features(indices_list):
    all_features = []
    for cls_indices in indices_list:
        cls_features = []
        for idx in cls_indices:
            img, _ = cifar[idx]
            inputs = processor(images=img, return_tensors="pt")
            with torch.no_grad():
                output = backbone(**inputs)
            cls_features.append(output.last_hidden_state[:, 0, :])
        all_features.append(torch.cat(cls_features, dim=0))
    return all_features

print("Extraction features support set...")
support_features = extract_features(support_indices)

print("Extraction features query set...")
query_features = extract_features(query_indices)

# --- Étape 3 : calculer les prototypes ---
prototypes = torch.stack([f.mean(dim=0) for f in support_features])
print(f"\nPrototypes shape: {prototypes.shape}")  # [5, 384]

# --- Étape 4 : classifier les queries ---
correct = 0
total = 0

for cls_idx, cls_queries in enumerate(query_features):
    for query in cls_queries:
        distances = torch.cdist(query.unsqueeze(0), prototypes)
        predicted = distances.argmin(dim=1).item()
        if predicted == cls_idx:
            correct += 1
        total += 1

accuracy = correct / total * 100
print(f"\nAccuracy: {accuracy:.1f}% (random = 20%)")