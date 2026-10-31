"""
Hybrid multimodal transformer used by both the CL-Drive and DROZY models:
1D-CNN tokenizer stem per modality -> intra-modality self-attention ->
cross-modal fusion transformer over all tokens + [CLS] -> 3-class head and level regressor.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNTokenizerStem(nn.Module):
    """(B, C, 1000) raw waveform -> (B, num_tokens, d_model) tokens."""
    def __init__(self, in_channels: int, d_model: int = 64, num_tokens: int = 25, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, d_model // 2, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(d_model // 2)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(d_model // 2, d_model, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(num_tokens)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pool1(F.gelu(self.bn1(self.conv1(x))))
        out = self.pool2(F.gelu(self.bn2(self.conv2(out))))
        out = self.dropout(self.adaptive_pool(out))
        return out.transpose(1, 2)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 64, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class MultimodalDriverTransformer(nn.Module):
    """
    modalities: dict name -> (first channel, last channel + 1, embedding id), in fusion order.
    forward(x) takes the full calibrated (B, C, 1000) tensor and slices it per modality.
    """
    def __init__(self, modalities: dict, d_model: int = 64, nhead: int = 4, intra_layers: int = 1,
                 cross_layers: int = 2, dim_feedforward: int = 128, num_tokens: int = 25,
                 num_classes: int = 3, dropout: float = 0.2):
        super().__init__()
        self.modalities = {m: tuple(v) for m, v in modalities.items()}

        self.stems = nn.ModuleDict({
            m: CNNTokenizerStem(stop - start, d_model, num_tokens, dropout)
            for m, (start, stop, _) in self.modalities.items()
        })
        self.pos_encoder = PositionalEncoding(d_model, max_len=num_tokens + 5)
        self.modality_embed = nn.Embedding(len(self.modalities), d_model)

        def encoder(num_layers):
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, activation='gelu', batch_first=True)
            return nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)

        self.intra_encoders = nn.ModuleDict({m: encoder(intra_layers) for m in self.modalities})
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.cross_transformer = encoder(cross_layers)

        self.head_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, num_classes))
        self.regressor = nn.Sequential(nn.Linear(d_model, 16), nn.GELU(), nn.Linear(16, 1))

    def forward(self, x: torch.Tensor):
        tokens = []
        for m, (start, stop, mod_id) in self.modalities.items():
            tok = self.stems[m](x[:, start:stop])
            mod = self.modality_embed.weight[mod_id]
            tok = self.intra_encoders[m](self.pos_encoder(tok) + mod)
            tokens.append(tok)

        seq = torch.cat([self.cls_token.expand(x.size(0), -1, -1)] + tokens, dim=1)
        cls_repr = self.head_norm(self.cross_transformer(seq)[:, 0])
        return self.classifier(cls_repr), self.regressor(cls_repr).squeeze(-1)


def build_model(checkpoint_or_config: dict) -> MultimodalDriverTransformer:
    cfg = checkpoint_or_config.get('model_config', checkpoint_or_config)
    return MultimodalDriverTransformer(modalities=cfg['modalities'], **cfg['hparams'])


def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    model = build_model(ckpt)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device).eval()
    return model, ckpt
