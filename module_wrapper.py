import torch

'''
Image (HxWx3)
      │
      ▼
Image Processor / Preprocess
(resize, crop, normalize, to tensor, batch)
      │              output: (B, 3, H', W')
      ▼
Vision Encoder (e.g., ViT / MobileCLIP)
(patchify + transformer blocks)
      │              output tokens: (B, N, Dv)
      │              N = #patch tokens (e.g., 256)
      │              Dv = vision hidden dim (e.g., 3072)
      ▼
[optional: Token Pooling]
 ├─ CLS token:      x_cls = tokens[:, 0, :]          → (B, Dv)
 └─ Mean pooling:   x_mean = tokens.mean(dim=1)      → (B, Dv)
      │
      ▼
Vision Projector (MLP inside nn.Sequential)
e.g., Linear(Dv→Dh) → GELU → Linear(Dh→Dh)
      │              applies per vector (CLS/mean) or per token
      │              Dh = projector (LM) hidden dim (e.g., 896)
      ▼
Projected Embedding
 ├─ per-image vector (if pooled before projector):  (B, Dh)
 └─ per-token vectors (if projector run on all tokens): (B, N, Dh)
      │
      ▼
[optional: Post-pooling + L2-norm]
 pooled = proj.mean(dim=1) or proj[:,0,:]   → (B, Dh)
 emb = pooled / (||pooled|| + 1e-9)         → cosine-ready
 '''

class VisionEncoderWrapper(torch.nn.Module):
    def __init__(self, vision_encoder):
        super().__init__()
        self.vision_encoder = vision_encoder
    def forward(self, x):
        return self.vision_encoder(x)

class ProjectorWrapper(torch.nn.Module):
    """
    Projects token embeddings with an MLP/Linear projector.
    Input:  (B, N, Dv)  or (B, Dv)
    Output: (B, N, Dh)  or (B, Dh)  (same rank as input)
    """
    def __init__(self, projector: torch.nn.Module):
        super().__init__()
        self.projector = projector

    def forward(self, x):
        return self.projector(x)

class VisionEncoderProjectorWrapper(torch.nn.Module):
    """
    Full extractor that returns BOTH:
      - vision tokens  (B, N, Dv)
      - pooled proj embedding (B, Dh)  (CLS or mean after projection)
    """
    def __init__(self, vision_encoder: torch.nn.Module, projector: torch.nn.Module, pool: str = "mean"):
        super().__init__()
        assert pool in ("mean", "cls")
        self.vision = VisionEncoderWrapper(vision_encoder)
        self.proj = ProjectorWrapper(projector)
        self.pool = pool

    def forward(self, pixel_values):
        tokens = self.vision(pixel_values)      # (B, N, Dv)
        proj_dtype = next(self.proj.parameters()).dtype
        tokens = tokens.to(proj_dtype)
        proj_tokens = self.proj(tokens.last_hidden_state if hasattr(tokens, "last_hidden_state") else tokens)         # (B, N, Dh)  (FastVLM projector is per-token)
        if self.pool == "mean":
            pooled = proj_tokens.mean(dim=1)    # (B, Dh)
        else:
            pooled = proj_tokens[:, 0, :]       # (B, Dh)  CLS
        return tokens, pooled                   # tokens before proj, pooled after proj
