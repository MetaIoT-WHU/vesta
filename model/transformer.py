import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(~mask, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, v)
        return output, attention_weights

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        attn_output, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(attn_output)
        return output, attention_weights


class SatelliteMultiLSTMWithAttention(nn.Module):
    """Multi-satellite LSTM + attention fusion network."""

    def __init__(
        self,
        input_dim=2,
        sequence_length=100,
        lstm_hidden_dim=128,
        lstm_num_layers=2,
        attention_dim=256,
        num_classes=2,
        dropout=0.1,
        use_position_encoding=True,
        attention_layers=1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.attention_dim = attention_dim
        self.num_classes = num_classes
        self.use_position_encoding = use_position_encoding
        self.attention_layers = max(1, int(attention_layers))

        self.satellite_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=True,
        )

        lstm_output_dim = lstm_hidden_dim * 2
        if self.use_position_encoding:
            self.position_encoder = nn.Sequential(
                nn.Linear(2, attention_dim // 4),
                nn.ReLU(),
                nn.Linear(attention_dim // 4, attention_dim),
            )
        else:
            self.position_encoder = None

        self.feature_projection = nn.Linear(lstm_output_dim, attention_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention_blocks = nn.ModuleList(
            [MultiHeadAttention(attention_dim, num_heads=4) for _ in range(self.attention_layers)]
        )
        self.attention_norms = nn.ModuleList([nn.LayerNorm(attention_dim) for _ in range(self.attention_layers)])
        self.feed_forward_norms = nn.ModuleList([nn.LayerNorm(attention_dim) for _ in range(self.attention_layers)])
        self.feed_forward_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(attention_dim, attention_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(attention_dim * 2, attention_dim),
                )
                for _ in range(self.attention_layers)
            ]
        )
        self.classifier1 = nn.Sequential(
            nn.Linear(attention_dim, attention_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(attention_dim // 2, num_classes),
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(attention_dim, attention_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(attention_dim // 2, num_classes),
        )

    def forward(self, satellite_signals, azimuth, elevation, mask=None):
        batch_size, num_satellites, seq_len, input_dim = satellite_signals.shape
        signals_reshaped = satellite_signals.view(-1, seq_len, input_dim)

        # Temporal encoding for each satellite sequence (LSTM)
        _, (hidden, _) = self.satellite_lstm(signals_reshaped)
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        satellite_features = torch.cat([forward_hidden, backward_hidden], dim=-1)
        satellite_features = satellite_features.view(batch_size, num_satellites, -1)
        satellite_features = self.feature_projection(satellite_features)

        # Direction-aware positional embedding (azimuth / elevation)
        if self.use_position_encoding and self.position_encoder is not None:
            position_input = torch.stack([azimuth, elevation], dim=-1)
            position_encoding = self.position_encoder(position_input)
            satellite_features = satellite_features + position_encoding

        if mask is not None:
            attn_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        else:
            attn_mask = None

        # Cross-satellite self-attention fusion
        for layer_idx in range(self.attention_layers):
            layer_input = self.attention_norms[layer_idx](satellite_features)
            attended_features, _ = self.attention_blocks[layer_idx](
                layer_input, layer_input, layer_input, attn_mask
            )
            satellite_features = satellite_features + self.dropout(attended_features)
            ff_input = self.feed_forward_norms[layer_idx](satellite_features)
            ff_output = self.feed_forward_blocks[layer_idx](ff_input)
            satellite_features = satellite_features + self.dropout(ff_output)

        if mask is not None:
            satellite_features = satellite_features * mask.unsqueeze(-1)
            valid_counts = mask.sum(dim=1, keepdim=True).float()
            pooled_features = satellite_features.sum(dim=1) / valid_counts.clamp(min=1)
        else:
            pooled_features = satellite_features.mean(dim=1)

        # Parallel heads for multi-target activity recognition
        output1 = self.classifier1(pooled_features)
        output2 = self.classifier2(pooled_features)
        return {
            "predictions1": output1,
            "predictions2": output2,
        }
