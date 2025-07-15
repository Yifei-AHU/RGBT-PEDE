import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class FusionModule(nn.Module):
    def __init__(self, embed_dim=512):
        super(FusionModule, self).__init__()

        projection_dim = embed_dim * 2 # 4
        hidden_dim = embed_dim * 4
        self.rgb_projection_layer = nn.Linear(embed_dim, projection_dim) # projection_dim: embed_dim * 4
        self.t_projection_layer = nn.Linear(embed_dim, projection_dim)

        self.QuickGELU = QuickGELU()

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        self.combiner_layer = nn.Linear(projection_dim * 2, hidden_dim) # hidden_dim: embed_dim * 8
        self.output_layer = nn.Linear(hidden_dim, embed_dim)

        self.dropout3 = nn.Dropout(0.5)

        self.dynamic_scalar = nn.Sequential(nn.Linear(projection_dim * 2, hidden_dim), QuickGELU(), nn.Dropout(0.5),
                                            nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, rgb_features, t_features):

        rgb_features = rgb_features.permute(1,0,2)
        t_features = t_features.permute(1,0,2)

        rgb_projected_features = self.dropout1(self.QuickGELU(self.rgb_projection_layer(rgb_features)))
        t_projected_features = self.dropout2(self.QuickGELU(self.t_projection_layer(t_features)))

        raw_combined_features = torch.cat((rgb_projected_features, t_projected_features), -1)

        combined_features = self.dropout3(self.QuickGELU(self.combiner_layer(raw_combined_features)))

        dynamic_scalar = self.dynamic_scalar(raw_combined_features)
        output = self.output_layer(combined_features) + dynamic_scalar * rgb_features + (
                1 - dynamic_scalar) * t_features
        
        output = output.permute(1,0,2)

        return output.float(), output[:, 0, :].float()
