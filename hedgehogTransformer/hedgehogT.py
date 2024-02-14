import torch
import torch.nn as nn
import torch.nn.functional as F


class HedgehogFeatureMap(nn.Module):
    def __init__(self, head_dim: int):
        super(HedgehogFeatureMap, self).__init__()
        self.linear = nn.Linear(head_dim, head_dim)
        self.init_weights()

    def init_weights(self):
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor):
        # Apply the trainable linear transformation followed by an element-wise exp to x and its negation.
        return torch.cat(
            [torch.exp(self.linear(x)), torch.exp(-self.linear(x))], dim=-1
        )


class HedgehogAttention(nn.Module):
    def __init__(self, base_attn, head_dim: int):
        super(HedgehogAttention, self).__init__()
        self.base_attn = base_attn
        self.head_dim = head_dim
        self.mlp_q = HedgehogFeatureMap(self.head_dim)
        self.mlp_k = HedgehogFeatureMap(self.head_dim)

        # Freeze original attention parameters
        for p in self.base_attn.parameters():
            p.requires_grad = False

        self.training = True

    def forward(self, hidden_states: torch.Tensor, output_attentions: bool = True):
        # Apply base attention mechanism to get "true" attention scores
        true_attn = self.base_attn(hidden_states, output_attentions=output_attentions)

        # Project hidden states to query and key space
        q = self.mlp_q(hidden_states)
        k = self.mlp_k(hidden_states)

        # Compute attention scores
        # Adjusting the computation to ensure the correct shape for pred_attn
        attn_scores = torch.einsum("bhd,bhd->bh", q, k).unsqueeze(
            -1
        )  # This line needs correction

        # Correct approach to compute full attention matrix
        attn_scores = torch.einsum("bmd,bnd->bmn", q, k)
        # Apply softmax to get the predicted attention across all keys for each query
        pred_attn = F.softmax(attn_scores, dim=-1)

        if self.training:
            # Compute distillation loss if in training mode
            loss = compute_hedgehog_loss(q, k, true_attn, pred_attn)
            return loss, pred_attn, true_attn
        else:
            return pred_attn


def compute_hedgehog_loss(
    q: torch.Tensor, k: torch.Tensor, true_attn: torch.Tensor, pred_attn: torch.Tensor
):
    return soft_label_cross_entropy(pred_attn, true_attn)


def soft_label_cross_entropy(pred, target):
    log_pred = torch.log(pred + 1e-9)
    return torch.mean(torch.sum(-target * log_pred, dim=-1))


# Example of initializing the model:
# base_attn = ... # Your base attention module here
# hedgehog_attn = HedgehogAttention(base_attn, head_dim=64)
# hidden_states = torch.rand(2, 10, 64)  # (batch_size, seq_length, head_dim)
# outputs = hedgehog_attn(hidden_states, output_attentions=True)
# if isinstance(outputs, tuple) and len(outputs) == 3:
#     loss, pred_attn, true_attn = outputs
# else:
#     pred_attn = outputs
