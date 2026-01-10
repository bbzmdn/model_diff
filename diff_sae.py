import torch
import torch.nn as nn
from typing import Tuple
from jaxtyping import Float
from torch import Tensor


class DiffSAE(nn.Module):
    def __init__(self, input_dim: int, dict_size: int, k: int = 64, auxk: int = 256):
        super().__init__()
        self.input_dim: int = input_dim
        self.dict_size: int = dict_size
        self.k: int = k
        self.auxk: int = auxk

        self.encoder: nn.Linear = nn.Linear(input_dim, dict_size, bias=True)
        self.decoder: nn.Linear = nn.Linear(dict_size, input_dim, bias=True)
        self.decoder.weight.data = self.encoder.weight.data.t().clone()

        with torch.no_grad():
            self.decoder.weight.data = nn.functional.normalize(
                self.decoder.weight.data, dim=0
            )

    def encode(self, x: Float[Tensor, "batch input_dim"]) -> Float[Tensor, "batch dict_size"]:
       return self.encoder(x)

    def batchtopk(self, x: Float[Tensor, "batch dict_size"], k: int) -> Float[Tensor, "batch dict_size"]:
        topk_values, topk_indices = torch.topk(x, k, dim=-1)
        result: Float[Tensor, "batch dict_size"] = torch.zeros_like(x)
        result.scatter_(-1, topk_indices, topk_values)

        return result

    def decode(self, z: Float[Tensor, "batch dict_size"]) -> Float[Tensor, "batch input_dim"]:
        return self.decoder(z)

    def forward(self, x: Float[Tensor, "batch input_dim"], return_aux: bool = False):
        pre_activations = self.encode(x)
        pre_activations = torch.relu(pre_activations)
        latents = self.batchtopk(pre_activations, self.k)
        x_reconstructed = self.decode(latents)
        if return_aux:
            topk_indices = torch.topk(pre_activations, self.k, dim=-1)[1]

            mask = torch.ones_like(pre_activations, dtype=torch.bool)
            mask.scatter_(-1, topk_indices, False)

            dead_features = (pre_activations * mask.float())
            aux_latents = self.batchtopk(dead_features,min(self.auxk, dead_features.shape[-1]))
            aux_reconstructed = self.decode(aux_latents)

            aux_loss = torch.nn.functional.mse_loss(aux_reconstructed, x)
            return x_reconstructed, latents, aux_loss

        return x_reconstructed, latents
