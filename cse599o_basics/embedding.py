import torch

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """Construct an embedding module.
        This function should accept the following parameters:
        
        num_embeddings: int
            Size of the vocabulary
        embedding_dim: int
            Dimension of the embedding vectors, i.e., d_model
        device: torch.device | None = None
            Device to store the parameters on
        dtype: torch.dtype | None = None
            Data type of the parameters
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weights = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)))

    def forward(self, token_ids: torch.Tensor)-> torch.Tensor:
        """Lookup the embedding vectors for the given token IDs."""
        return self.weights[token_ids]