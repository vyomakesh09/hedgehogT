import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class HedgehogFeatureMap(nn.Module):
    def __init__(self, head_dim: int, activation: str= 'exp'):
        super().__init__()
        # Trainable map
        self.layer = nn.Linear(head_dim, head_dim)
        self.init_weights()

    def _init_weights_(self):
        
        """" Initialize traninable map as identity """
        nn.init.eye_(self.layer.weight)
        nn.init.zeros_(self.layer.bias)

    def forward(self, x: torch.Tensor):
        x = self.layer(x) # shape b, h, l, d
        return torch.cat([torch.exp(x), torch.exp(-x)], dim=-1)

def softmax_attn(q: torch.Tensor, k: torch.Tensor):
    """" Get softmax attention weights -> Assume q, k, are both shape 
            (b, h, 1, d) """
    scale = q.shape[-1] ** 0.5
    qk = torch.einsum('bhmd, bhnd->bhmn', q, k) / scale
    return torch.softmax(qk, dim=-1)

def quadratic_linear_attn(q: torch.Tensor, k: torch.Tensor):
    """
        Get linear attention weights 
        -> Assume q, k are both shape (b, h, 1, d) and feature maps 
            already applied """
    qk = torch.einsum('bhmd, bhnd->bhmn', q, k)
    return qk / qk.sum(dim=-1, keepdim=True)

'''def compute_hedgehog_loss(q: torch.Tensor,
                            k: torch.Tensor,
                            hh_mlp_q: HedgehogFeatureMap,
                            hh_mlp_k: HedgehogFeatureMap):
        """
        Compute the attention distillation loss 
        -> Assume 'soft_label_cross_entropy' is implemented 
            (aleternatively use KL divergence)
        -> Assume q and k are the queries and keys of a 
            pretrained Transformer,
            via q = self.q_proj(hidden_states)
        """

        true_attn = softmax_attn(q, k)
        pred_attn = quadratic_linear_attn(hh_mlp_q(q), hh_mlp_k(k))
        return soft_label_cross_entropy(pred_attn, true_attn)

'''

class HedgehogAttention(nn.Module):
    """
    Sample code for HedgehogAttention, following HuggingFace API 
    """

    def __init__(self, base_attn, training = True):
        self.base_attn = base_attn 

        # Trainable feature maps 
        self.mlp_q = HedgehogFeatureMap(base_attn.head_dim)
        self.mlp_k = HedgehogFeatureMap(base_attn.head_dim)

        # Freeze original attention parameters 
        for p in self.base_attn.parameters():
            p.requires_grad = False

        self.q_proj = self.base_attn.q_proj
        self.k_proj = self.base_attn.k_proj


        # Whether we train attentions or not 
        self.training = training 

    def forward(self, 
                hidden_states: torch.Tensor, 
                output_attentions: bool = True,
                **base_kwargs: any):
        
        
        if self.training:
            # Compute ground-truth attention weights
            outputs, true_attns = self.base_attn(
                hidden_states=hidden_states,
                output_attentions=True,
                **base_kwargs
            )

            # Compute Hedgehig Featture Maps 
            q = self.mlp_q(self.q_proj(hidden_states))
            k = self.mlp_k(self.k_proj(hidden_states))

            pred_attns = quadratic_linear_attn(q, k)
            

            if output_attentions:
                # Hook for attentions 
                return outputs, (pred_attns, true_attns)
