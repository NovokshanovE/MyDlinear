


class DLinear:
    """
    Linear layer with dropout.

    Args:
        in_features: number of input features
        out_features: number of output features
        bias: If False, then the layer does not use bias weights.
        dropout: dropout probability
    """

    def __init__(self, in_features, out_features, bias=True, dropout=0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear(x)
        x = self.dropout(x)
        return x

        
    

    