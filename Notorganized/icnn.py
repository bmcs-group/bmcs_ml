from torch import nn
import torch

class ICNN(nn.Module):
    """ Creates a simple Input Convex Neural Network with 1 input and 1 output. """

    def __init__(self):
        super(ICNN, self).__init__()

        # First hidden layer (input size of 1 for scalar input)
        self.first_hidden_layer = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU()
        )

        # Second layer with non-negative weights
        self.second_layer_linear_prim = nn.Linear(64, 64)
        self.second_layer_linear_prim.weight.data = torch.abs(
            self.second_layer_linear_prim.weight.data)
        self.second_layer_linear_skip = nn.Linear(1, 64)
        self.second_layer_act = nn.ReLU()

        # Third layer with non-negative weights
        self.third_layer_linear_prim = nn.Linear(64, 64)
        self.third_layer_linear_prim.weight.data = torch.abs(
            self.third_layer_linear_prim.weight.data)
        self.third_layer_linear_skip = nn.Linear(1, 64)
        self.third_layer_act = nn.ReLU()

        # Fourth layer with non-negative weights
        self.fourth_layer_linear_prim = nn.Linear(64, 64)
        self.fourth_layer_linear_prim.weight.data = torch.abs(
            self.fourth_layer_linear_prim.weight.data)
        self.fourth_layer_linear_skip = nn.Linear(1, 64)
        self.fourth_layer_act = nn.ReLU()

        # Fifth layer with non-negative weights
        self.fifth_layer_linear_prim = nn.Linear(64, 64)
        self.fifth_layer_linear_prim.weight.data = torch.abs(
            self.fifth_layer_linear_prim.weight.data)
        self.fifth_layer_linear_skip = nn.Linear(1, 64)
        self.fifth_layer_act = nn.ReLU()

        # Final output layer with a single output
        self.output_layer_linear_prim = nn.Linear(64, 1)
        self.output_layer_linear_prim.weight.data = -1 * torch.abs(
            self.output_layer_linear_prim.weight.data)
        self.output_layer_linear_skip = nn.Linear(1, 1)

    def forward(self, x):
        # No need to flatten as input is scalar (1D)
        skip_x2 = x
        skip_x3 = x
        skip_x4 = x
        skip_x5 = x
        skip_x6 = x

        # Pass through layers with skip connections and clamping
        z1 = self.first_hidden_layer(x)
        z1 = self.second_layer_linear_prim(z1)
        z1 = torch.clamp(z1, min=0, max=None)
        y2 = self.second_layer_linear_skip(skip_x2)
        z2 = self.second_layer_act(z1 + y2)

        z2 = self.third_layer_linear_prim(z2)
        z2 = torch.clamp(z2, min=0, max=None)
        y3 = self.third_layer_linear_skip(skip_x3)
        z3 = self.third_layer_act(z2 + y3)

        z3 = self.fourth_layer_linear_prim(z3)
        z3 = torch.clamp(z3, min=0, max=None)
        y4 = self.fourth_layer_linear_skip(skip_x4)
        z4 = self.fourth_layer_act(z3 + y4)

        z4 = self.fifth_layer_linear_prim(z4)
        z4 = torch.clamp(z4, min=0, max=None)
        y5 = self.fifth_layer_linear_skip(skip_x5)
        z5 = self.fifth_layer_act(z4 + y5)

        # Final output layer, clamped to be non-positive
        z5 = self.output_layer_linear_prim(z5)
        z5 = torch.clamp(z5, min=None, max=0)
        y6 = self.output_layer_linear_skip(skip_x6)
        logits = z5 + y6

        return logits
