from src.models.blocks.conv_block import ConvBlock
from src.settings.settings import Settings
import torch
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch import nn


class EfficientLRCN(nn.Module):
    """
    Model that combines efficientnet and lstm
    the embedding is directly passed from the so called "manifold of interest" to lstm
    see the paper
    """

    def __init__(self, settings: Settings):
        """
        Method to initialise the model
        Args:
            settings: the settings object
        """
        super(EfficientLRCN, self).__init__()
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT).features
        self.backbone[8] = nn.Identity()
        for p in self.backbone[:settings.frozen_layers].parameters():
            p.requires_grad = False
        self.point_wise = ConvBlock(320, settings.lstm_input_size, (1, 1), 1, 0, activation_function=nn.SiLU(inplace=True)) if settings.rnn_point_wise else None
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.rnn = settings.rnn_type(
            input_size = settings._lstm_input_size,
            hidden_size = settings.lstm_hidden_size,
            num_layers = settings.lstm_num_layers,
            bias = settings.lstm_bias,
            dropout = settings.lstm_dropout_prob,
            bidirectional = settings.lstm_bidirectional,
            batch_first=True
        )
        self.classifier = nn.Linear(
            in_features=settings.lstm_hidden_size if not settings.lstm_bidirectional else settings.lstm_hidden_size * 2,
            out_features=len(settings.dataset_labels),
            bias=True
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the model
        Args:
            x: the input clip in NLCHW
        Returns:
            Tensor: logits
        """
        bs, seq = x.shape[0], x.shape[1]
        x = self.backbone(x.view(-1, x.shape[2], x.shape[3], x.shape[4]))
        if self.point_wise:
            x = self.point_wise(x)
        x = self.gap(x)
        x = x.view(bs, seq, -1)
        x, _ = self.rnn(x)
        x = self.classifier(x[:,-1,:])
        return x
    

if __name__ == "__main__":
    raise NotImplementedError("Usage: python main.py args")