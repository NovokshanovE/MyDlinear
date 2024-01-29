from typing import Tuple

import torch
from torch import nn
from Scaler import *
from Input import *

def make_linear_layer(dim_in, dim_out):
    lin = nn.Linear(dim_in, dim_out)
    torch.nn.init.uniform_(lin.weight, -0.07, 0.07)
    torch.nn.init.zeros_(lin.bias)
    return lin

class MovingAvg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(
            kernel_size=kernel_size, stride=stride, padding=0
        )

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, ...].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, ...].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x



class SeriesDecomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean



class DLinearModel(nn.Module):
    """
    Module implementing a feed-forward model form the paper
    https://arxiv.org/pdf/2205.13504.pdf extended for probabilistic forecasting.

    Parameters
    ----------
    prediction_length
        Number of time points to predict.
    context_length
        Number of time steps prior to prediction time that the model.
    hidden_dimension
        Size of last hidden layers in the feed-forward network.
    distr_output
        Distribution to use to evaluate observations and sample predictions.
    """

    
    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        hidden_dimension: int,
        #distr_output=StudentTOutput(),
        kernel_size: int = 25,
        scaling: str = "mean",
    ) -> None:
        super().__init__()

        assert prediction_length > 0
        assert context_length > 0

        self.prediction_length = prediction_length
        self.context_length = context_length
        self.hidden_dimension = hidden_dimension
        self.decomposition = SeriesDecomp(kernel_size)

        #self.distr_output = distr_output
        if scaling == "mean":
            self.scaler = MeanScaler(keepdim=True)
        elif scaling == "std":
            self.scaler = StdScaler(keepdim=True)
        else:
            self.scaler = NOPScaler(keepdim=True)

        self.kernel_size = kernel_size

        self.linear_seasonal = make_linear_layer(
            context_length, prediction_length * hidden_dimension
        )
        self.linear_trend = make_linear_layer(
            context_length, prediction_length * hidden_dimension
        )

        self.args_proj = self.distr_output.get_args_proj(hidden_dimension)

    def describe_inputs(self, batch_size=1) -> InputSpec:
        return InputSpec(
            {
                "past_target": Input(
                    shape=(batch_size, self.context_length), dtype=torch.float
                ),
                "past_observed_values": Input(
                    shape=(batch_size, self.context_length), dtype=torch.float
                ),
            },
            torch.zeros,
        )


    def forward(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
        # scale the input
        past_target_scaled, loc, scale = self.scaler(
            past_target, past_observed_values
        )
        res, trend = self.decomposition(past_target_scaled.unsqueeze(-1))
        seasonal_output = self.linear_seasonal(res.squeeze(-1))
        trend_output = self.linear_trend(trend.squeeze(-1))
        nn_out = seasonal_output + trend_output

        distr_args = self.args_proj(
            nn_out.reshape(-1, self.prediction_length, self.hidden_dimension)
        )
        return distr_args, loc, scale


