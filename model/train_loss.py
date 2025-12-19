import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, Counter

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)  # p_t

        if self.alpha is not None:
            at = self.alpha.to(inputs.device).gather(0, targets)
        else:
            at = 1.0

        focal_loss = at * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

def get_criterion(train_dataset, num_classes, device, gamma=2.0):
    train_labels = [label for _,label,_ in train_dataset.samples]
    label_counts = Counter(train_labels)
    total_samples = len(train_dataset)
    class_weights = []
    for cls in range(num_classes):
        count = label_counts.get(cls, 0)
        weight = total_samples / (count + 1e-8)
        class_weights.append(weight)
    class_weights = torch.tensor(class_weights, device=device, dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * num_classes
    print(f"class weights: {class_weights.cpu().numpy()}")

    return FocalLoss(alpha=class_weights, gamma=gamma)