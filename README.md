# CBAM-from-scratch
```
This is a pytorch implementation of CBAM.
```

CBAM: Convolutional Block Attention Module implementation from scratch with detailed explanation


The code here explains detailed step by step implementation of Convolutional Block Attention Module(CBAM).

## Introduction
The Convolutional Block Attention Module (CBAM) enhances the interpretability and performance of convolutional neural networks by focusing on salient features through attention mechanisms. Implemented sequentially, it modulates features along both the channel and spatial dimensions.

## Dependencies
Python 3.x
PyTorch
torchvision

## Structure of CBAM
CBAM sequentially applies two attention mechanisms:

Channel Attention Module (CAM): Emphasizes informative features along the channel axes.
Spatial Attention Module (SAM): Highlights important spatial locations.

The code shown below is complete code for CBAM. You can find the complete step by step explanation in this [file](https://github.com/Marshajennifer/CBAM-from-scratch/blob/main/CBAM_attention.ipynb)

```
import torch
import torch.nn as nn

class channel_attention_module(nn.Module):
    def __init__(self, ch, ratio=8):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(ch, ch//ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch//ratio, ch, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.avg_pool(x).squeeze(-1).squeeze(-1)
        x1 = self.mlp(x1)

        x2 = self.max_pool(x).squeeze(-1).squeeze(-1)
        x2 = self.mlp(x2)

        feats = x1 + x2
        feats = self.sigmoid(feats).unsqueeze(-1).unsqueeze(-1)
        refined_feats = x * feats

        return refined_feats


class spatial_attention_module(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = torch.mean(x, dim=1, keepdim=True)
        x2, _ = torch.max(x, dim=1, keepdim=True)

        feats = torch.cat([x1, x2], dim=1)
        feats = self.conv(feats)
        feats = self.sigmoid(feats)
        refined_feats = x * feats

        return refined_feats

class cbam(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.ca = channel_attention_module(channel)
        self.sa = spatial_attention_module()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


if __name__ == "__main__":
    x = torch.randn((8, 32, 128, 128))
    module = cbam(32)
    y = module(x)
    print(y.shape)

```


## Conclusion
By integrating the CBAM module into your neural networks, you enhance the model's ability to focus on the most informative parts of an input, improving performance across various tasks.

## Citation
1. Woo, S., Park, J., Lee, J., & Kweon, I. S. (2018). CBAM: Convolutional Block Attention Module. In Lecture notes in computer science (pp. 3–19). https://doi.org/10.1007/978-3-030-01234-2_1
2. [CBAM Explanation.](https://sh-tsang.medium.com/reading-cbam-convolutional-block-attention-module-image-classification-ddbaf10f7430)

