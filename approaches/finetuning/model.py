import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from transformers import AutoModel, AutoImageProcessor
from peft import LoraConfig, get_peft_model


class LoRAFinetuner:
    """Few-shot classification by fine-tuning DINOv2 with LoRA.
    
    For each episode:
    1. Inject LoRA into DINOv2
    2. Fine-tune on support set with prototype loss
    3. Classify queries using adapted features
    4. Reset LoRA weights for next episode
    """

    def __init__(self, model_name="facebook/dinov2-small", lora_rank=4,
                 lr=1e-3, n_steps=20, distance="cosine"):
        self.model_name = model_name
        self.lora_rank = lora_rank
        self.lr = lr
        self.n_steps = n_steps
        self.distance = distance

        self.processor = AutoImageProcessor.from_pretrained(model_name)

        # Build LoRA model
        base_model = AutoModel.from_pretrained(model_name)
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
        )
        self.model = get_peft_model(base_model, lora_config)

        # Save initial LoRA weights for reset between episodes
        self._initial_lora_state = copy.deepcopy(
            {n: p.data.clone() for n, p in self.model.named_parameters() if p.requires_grad}
        )

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"LoRA params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    def _reset_lora(self):
        """Reset LoRA weights to initial values between episodes."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self._initial_lora_state:
                param.data.copy_(self._initial_lora_state[name])

    def _process_images(self, images):
        """PIL images -> pixel_values tensor."""
        if not isinstance(images, list):
            images = [images]
        inputs = self.processor(images=images, return_tensors="pt")
        return inputs["pixel_values"]

    def _extract_cls(self, pixel_values):
        """Extract CLS token from pixel values."""
        output = self.model(pixel_values=pixel_values)
        return output.last_hidden_state[:, 0, :]

    def _compute_prototypes(self, features_list):
        """Compute prototypes from list of per-class features."""
        return torch.stack([f.mean(dim=0) for f in features_list])

    def _classify(self, queries, prototypes):
        """Classify queries by distance to prototypes."""
        if self.distance == "cosine":
            q = F.normalize(queries, dim=1)
            p = F.normalize(prototypes, dim=1)
            distances = 1 - q @ p.t()
        else:
            distances = torch.cdist(queries, prototypes)

        predictions = distances.argmin(dim=1)
        return predictions, distances

    def _finetune_on_support(self, support_images):
        """Fine-tune LoRA on support set using prototype loss.
        
        Args:
            support_images: list of lists of PIL images
                support_images[i] = K images for class i
        """
        self.model.train()
        optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr
        )

        # Pre-process all support images
        all_pixels = []
        all_labels = []
        for cls_id, class_images in enumerate(support_images):
            pixels = self._process_images(class_images)
            all_pixels.append(pixels)
            all_labels.extend([cls_id] * len(class_images))

        all_pixels = torch.cat(all_pixels, dim=0)
        all_labels = torch.tensor(all_labels)

        for step in range(self.n_steps):
            features = self._extract_cls(all_pixels)

            # Split features by class for prototype computation
            features_by_class = []
            for cls_id in range(len(support_images)):
                mask = all_labels == cls_id
                features_by_class.append(features[mask])

            prototypes = self._compute_prototypes(features_by_class)

            # Prototype loss: each support image should be close to its class prototype
            if self.distance == "cosine":
                f_norm = F.normalize(features, dim=1)
                p_norm = F.normalize(prototypes, dim=1)
                distances = 1 - f_norm @ p_norm.t()
            else:
                distances = torch.cdist(features, prototypes)

            loss = F.cross_entropy(-distances, all_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    @torch.no_grad()
    def evaluate_episode(self, support_images, query_images, query_labels):
        """Full episode: fine-tune on support, evaluate on query, reset.
        
        Args:
            support_images: list of lists of PIL images
            query_images: list of PIL images
            query_labels: tensor with ground truth class indices
            
        Returns:
            accuracy: float
        """
        # Reset LoRA to initial weights
        self._reset_lora()

        # Fine-tune on support (needs grad)
        with torch.enable_grad():
            self._finetune_on_support(support_images)

        # Evaluate on queries
        self.model.eval()
        query_pixels = self._process_images(query_images)
        query_features = self._extract_cls(query_pixels)

        # Build prototypes from adapted features
        support_features = []
        for class_images in support_images:
            pixels = self._process_images(class_images)
            features = self._extract_cls(pixels)
            support_features.append(features)

        prototypes = self._compute_prototypes(support_features)
        predictions, _ = self._classify(query_features, prototypes)

        accuracy = (predictions == query_labels).float().mean().item()
        return accuracy