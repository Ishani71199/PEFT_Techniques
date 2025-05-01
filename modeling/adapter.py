"""Input → Linear (down-project) → GELU(non-linearity) → Linear (up-project) → Add to Input (residual)

🔻Down-projection (to bottleneck):
- Forces the network to compress important task-specific information into fewer dimensions.
- Acts like a regularizer – avoids overfitting.

🔼 Up-projection:
- Brings it back to the original dimensionality so it can be added to the original flow (residual connection).

➕ Residual Addition:
- Maintains the original model behavior when the adapter is zero-initialized.
- Makes learning more stable and helps retain knowledge from the frozen base model.
"""
import torch
from .base_model import BaseModel
from functools import partial

class Adapter(torch.nn.Module):
    """Architecture: Linear (down-projection) -> GELU(non-linearity) -> Linear(up-projection)"""
    def __init__(self, linear_out_dim: int, bottleneck_dim: int):
        super().__init__()
        # Feedforward down-project
        self.linear1 = torch.nn.Linear(linear_out_dim, bottleneck_dim)
        # non-linearity
        self.gelu = torch.nn.GELU()
        # FeedForward up-project
        self.linear2 = torch.nn.Linear(bottleneck_dim, linear_out_dim)
    
    def forward(self, x):
        """
        Forward propogation of the Adapter layer
        """
        residual = x
        x = self.gelu(self.linear1(x)) # Applies the first linear transformation to reduce dimensionality (e.g. from 768 → 64).
        x = self.linear2(x) # Applies the second linear transformation to increase the dimensionality back (e.g. 64 → 768).
        return x + residual
    
class AdaptedLinear(torch.nn.Module):
    def __init__(self, linear, bottleneck_dim):
        super().__init__()
        self.linear = linear
        self.adapter = Adapter(linear.out_features, bottleneck_dim)
    def forward(self, x):
        """
        Forward propogation of the AdaptedLinear layer
        """
        x = self.linear(x)  # Normal linear layer propogation
        return self.adapter(x)  # Adapter layer propogation
    
class AdaptedModel(BaseModel):
    """Adds Adapter layers:
    After the output linear layer in self-attention (attention.out_lin)
    After the final feedforward projection (ffn.lin2)"""
    def __init__(self,
        model_uri: str = "distilbert/distilbert-base-uncased",
        num_classes: int = 2,
        freeze_all: bool = True,
        bottleneck_dim: int = 4, # bottleneck_dim: Defines the size to which hidden features are temporarily reduced (e.g. from 768 → 4 → 768).
    ):
        super().__init__(model_uri=model_uri, num_classes=num_classes, freeze_all=freeze_all)
        self.bottleneck_dim = bottleneck_dim
        self.__adapt()
    
    def __unfreeze_specific_layers(self) -> None:
        print("Unfreezing specific layers...")
        # Unfreeze (sa_layer_norm) and (output_layer_norm) in each TransformerBlock
        for block in self.model.distilbert.transformer.layer:
            block.sa_layer_norm.requires_grad = True
            block.output_layer_norm.requires_grad = True
        # Unfreeze final classifcation layer
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def __adapt(self) -> None:
        """
        Method that replaces specific Linear layers with AdaptedLinear layers
        Unfreezes specific layers as well
        """
        print("Adding Adapters...")
        adapted_linear = partial(AdaptedLinear, bottleneck_dim = self.bottleneck_dim)
        # Replace all Linear layers within the TransformerBlock with AdaptedLinear
        for block in self.model.distilbert.transformer.layer:
            ## Transformer Block: Multi-head Self-Attention block
            block.attention.out_lin = adapted_linear(block.attention.out_lin)
            ## Transformer Block: Feed-forward block
            block.ffn.lin2 = adapted_linear(block.ffn.lin2)
        # Unfreeze specific layers
        self.__unfreeze_specific_layers()

