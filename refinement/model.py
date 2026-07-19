"""
Refinement CNN: learns to map attention heatmap -> precise click coordinate.

Input: 2D heatmap (1, rows, cols) + normalized predicted points (4,)
Output: refined click point as normalized (x, y) in [0, 1]
"""

import torch
import torch.nn as nn


class RefinementNet(nn.Module):
    """
    CNN that refines attention-based predictions.

    Input:
        - 2D heatmap (batch, 1, rows, cols) — preserves spatial structure
        - Predicted centroid (normalized x, y)
        - Predicted argmax point (normalized x, y)

    Output:
        - Refined click point (normalized x, y)
    """

    def __init__(self, heatmap_rows: int = 10, heatmap_cols: int = 16):
        super().__init__()

        self.heatmap_rows = heatmap_rows
        self.heatmap_cols = heatmap_cols

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        # 32 channels * 4 * 4 = 512 from conv + 4 from centroid/argmax
        self.head = nn.Sequential(
            nn.Linear(512 + 4, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Sigmoid(),
        )

    def forward(self, heatmap, centroid_norm, argmax_norm):
        """
        Args:
            heatmap: (batch, 1, rows, cols) normalized heatmap
            centroid_norm: (batch, 2) predicted centroid as (x/w, y/h)
            argmax_norm: (batch, 2) predicted argmax as (x/w, y/h)

        Returns:
            (batch, 2) refined click point as (x/w, y/h)
        """
        conv_out = self.conv(heatmap)
        conv_flat = conv_out.view(conv_out.size(0), -1)
        x = torch.cat([conv_flat, centroid_norm, argmax_norm], dim=1)
        return self.head(x)
