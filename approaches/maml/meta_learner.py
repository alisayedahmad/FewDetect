import torch
import torch.nn as nn
import torch.nn.functional as F
import higher
import copy


class MAMLMetaLearner:
    """MAML and FOMAML meta-learning.
    
    Inner loop: adapt classifier head on support set
    Outer loop: update initial weights based on query set performance
    """

    def __init__(self, model, inner_lr=0.01, outer_lr=0.001,
                 inner_steps=5, first_order=True):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.first_order = first_order  # True = FOMAML, False = full MAML

        # Only optimize the head, not the frozen backbone
        self.meta_optimizer = torch.optim.Adam(
            self.model.head.parameters(), lr=outer_lr
        )

    def inner_loop(self, support_features, support_labels):
        """Adapt the model on one task's support set
        
        Args:
            support_features: tensor [n_support, embed_dim]
            support_labels: tensor [n_support]
            
        Returns:
            adapted model (functional form via higher)
        """
        inner_opt = torch.optim.SGD(
            self.model.head.parameters(), lr=self.inner_lr
        )

        with higher.innerloop_ctx(
            self.model, inner_opt,
            copy_initial_weights=False,
            track_higher_grads=not self.first_order
        ) as (fmodel, diffopt):
            for step in range(self.inner_steps):
                logits = fmodel.head(support_features)
                loss = F.cross_entropy(logits, support_labels)
                diffopt.step(loss)

            return fmodel

    def meta_train_episode(self, support_features, support_labels,
                           query_features, query_labels):
        """One meta-training episode
        
        Inner loop adapts on support, outer loop updates on query
        
        Returns:
            query_loss: float
            query_accuracy: float
        """
        # Inner loop — adapt on support set
        adapted_model = self.inner_loop(support_features, support_labels)

        # Evaluate adapted model on query set
        query_logits = adapted_model.head(query_features)
        query_loss = F.cross_entropy(query_logits, query_labels)

        # Outer loop — update initial weights
        self.meta_optimizer.zero_grad()
        query_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.head.parameters(), max_norm=10.0)
        self.meta_optimizer.step()

        # Compute accuracy
        predictions = query_logits.argmax(dim=1)
        accuracy = (predictions == query_labels).float().mean().item()

        return query_loss.item(), accuracy

    def meta_test_episode(self, support_features, support_labels,
                          query_features, query_labels):
        """One meta-test episode. Adapts on support, evaluates on query
        
        No outer loop update — just measures performance.
        """
        # Save original weights
        original_state = copy.deepcopy(self.model.head.state_dict())

        # Inner loop — adapt with regular gradient descent
        inner_opt = torch.optim.SGD(
            self.model.head.parameters(), lr=self.inner_lr
        )

        for step in range(self.inner_steps):
            logits = self.model.head(support_features)
            loss = F.cross_entropy(logits, support_labels)
            inner_opt.zero_grad()
            loss.backward()
            inner_opt.step()

        # Evaluate on query set
        query_logits = self.model.head(query_features)
        predictions = query_logits.argmax(dim=1)
        accuracy = (predictions == query_labels).float().mean().item()

        # Restore original weights
        self.model.head.load_state_dict(original_state)

        return accuracy