"""
Temporal Fusion Transformer (simplified, dual-branch) forecast model.
Mirrors forecasting/models/tcn_model.py's import contract:
    from forecasting.models.tft_model import TFTForecastModel

Architecture matches the training notebook (TFT.ipynb) exactly — same
class names, same default hyperparameters (hidden_size=16, nhead=2,
dropout=0.15) — so tft_model.pth's state_dict loads without shape
mismatches. If hidden_size/nhead are ever changed during retraining,
update the defaults here to match, or the saved weights won't load.

Static covariate encoding (tank-level features like catchment_area_km2)
is omitted — this pipeline doesn't feed those in as a separate static
context vector, only as per-timestep repeated features, same as the
other three models (LSTM/TCN/plain Transformer).
"""
import torch
import torch.nn as nn


class GatedLinearUnit(nn.Module):
    """GLU: splits a linear projection in half and uses one half as a
    sigmoid gate on the other. Lets the model learn to suppress a
    transformation entirely if it isn't useful."""
    def __init__(self, input_size, output_size=None):
        super().__init__()
        output_size = output_size or input_size
        self.fc = nn.Linear(input_size, output_size * 2)
        self.output_size = output_size

    def forward(self, x):
        x = self.fc(x)
        a, b = x[..., :self.output_size], x[..., self.output_size:]
        return a * torch.sigmoid(b)


class GatedResidualNetwork(nn.Module):
    """TFT's core building block: a small nonlinear transform (fc1 -> ELU
    -> fc2) wrapped in a gated residual connection, so information can
    skip the transform if the gate learns it's not useful."""
    def __init__(self, input_size, hidden_size, output_size=None, dropout=0.15):
        super().__init__()
        output_size = output_size or input_size
        self.skip = nn.Linear(input_size, output_size) if input_size != output_size else nn.Identity()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.gate = GatedLinearUnit(output_size, output_size)
        self.norm = nn.LayerNorm(output_size)

    def forward(self, x):
        residual = self.skip(x)
        h = self.elu(self.fc1(x))
        h = self.dropout(self.fc2(h))
        h = self.gate(h)
        return self.norm(residual + h)


class VariableSelectionNetwork(nn.Module):
    """Learns, at every timestep, how much weight to give each input
    variable. Returns both the selected/weighted embedding and the raw
    per-variable weights (useful for feature-importance analysis, not
    used by inference.py but kept for parity with the training notebook)."""
    def __init__(self, num_vars, hidden_size, dropout=0.15):
        super().__init__()
        self.num_vars = num_vars
        self.hidden_size = hidden_size
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(1, hidden_size, hidden_size, dropout) for _ in range(num_vars)
        ])
        self.weight_grn = GatedResidualNetwork(num_vars, hidden_size, num_vars, dropout)

    def forward(self, x):
        # x: (batch, seq_len, num_vars)
        b, t, v = x.shape
        flat = x.reshape(b * t, v)
        weights = torch.softmax(self.weight_grn(flat), dim=-1)            # (b*t, v)
        var_embeds = torch.stack(
            [self.var_grns[i](flat[:, i:i + 1]) for i in range(v)], dim=1
        )                                                                  # (b*t, v, hidden)
        selected = (weights.unsqueeze(-1) * var_embeds).sum(dim=1)         # (b*t, hidden)
        return selected.reshape(b, t, self.hidden_size), weights.reshape(b, t, v)


class TFTForecastModel(nn.Module):
    """Simplified single-entity TFT, dual-branch (regression + classification),
    same convention as the LSTM/TCN/Transformer models:
      - branch_reg sees ALL features (incl. storage) -> storage regression head
      - branch_cls sees storage-EXCLUDED features -> risk classification head
    """
    def __init__(self, input_size_full, input_size_cls, hidden_size=16,
                 nhead=2, dropout=0.15, forecast_steps=7, num_classes=3):
        super().__init__()

        def build_branch(num_vars):
            return nn.ModuleDict({
                "vsn": VariableSelectionNetwork(num_vars, hidden_size, dropout),
                "lstm": nn.LSTM(hidden_size, hidden_size, batch_first=True),
                "post_lstm_grn": GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout),
                "attn": nn.MultiheadAttention(hidden_size, nhead, dropout=dropout, batch_first=True),
                "post_attn_grn": GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout),
            })

        self.branch_reg = build_branch(input_size_full)
        self.branch_cls = build_branch(input_size_cls)

        self.fc_storage = nn.Linear(hidden_size, forecast_steps)
        self.fc_risk = nn.Linear(hidden_size, num_classes)

    def _encode(self, branch, x):
        selected, _weights = branch["vsn"](x)                      # (batch, seq_len, hidden)
        lstm_out, _ = branch["lstm"](selected)                     # local sequential processing
        lstm_out = branch["post_lstm_grn"](lstm_out)
        attn_out, _ = branch["attn"](lstm_out, lstm_out, lstm_out)  # self-attention over the window
        enriched = branch["post_attn_grn"](attn_out + lstm_out)    # gated skip around attention
        return enriched[:, -1, :]                                  # last timestep, same as other models

    def forward(self, x_full, x_cls):
        last_reg = self._encode(self.branch_reg, x_full)
        last_cls = self._encode(self.branch_cls, x_cls)
        return self.fc_storage(last_reg), self.fc_risk(last_cls)