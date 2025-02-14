from torch import nn
import torch
import numpy as np
from sklearn.utils import shuffle
from torch.optim import lr_scheduler

class SoftplusSquared(nn.Module):
    def __init__(self, beta=3):
        super(SoftplusSquared, self).__init__()
        self.beta = beta

    def forward(self, x):
        return (1 / (2 * self.beta ** 4)) * (torch.log(1 + torch.exp(self.beta ** 2 * x))) ** 2


class ICNN(nn.Module):
    def __init__(self, architecture=None, activation_function="elu", seed=42):
        super(ICNN, self).__init__()
        torch.manual_seed(seed)
        self.architecture = architecture if architecture is not None else [1, 100, 50, 50, 1]
        self.layers = nn.ModuleDict()
        if activation_function == "elu":
            self.act_fun = nn.ELU()
        elif activation_function == "relu":
            self.act_fun = nn.ReLU()
        elif activation_function == "softplus_squared":
            self.act_fun = SoftplusSquared()
        else:
            raise ValueError("Unsupported activation function. Choose from 'elu', 'relu', or 'softplus_squared'.")
        # Construct layers based on architecture
        for layer_idx in range(len(self.architecture) - 1):
            self.layers[str(layer_idx)] = nn.Linear(self.architecture[layer_idx], self.architecture[layer_idx + 1])
        self.model = self._create_nn()
        self.input_min, self.input_max = 0.0, 0.0
        self.output_min, self.output_max = 0.0, 0.0

    def _create_nn(self):
        """
        Creates a sequential model based on the architecture.
        """
        model = nn.Sequential()
        for i in range(len(self.architecture) - 1):
            model.add_module(f"linear_{i}", self.layers[str(i)])
            if i < len(self.architecture) - 2:
                model.add_module(f"activation_{i}", self.act_fun)
        return model

    def forward(self, x, do_unscaling=True):
        """
        Forward pass of the model.
        :param x: Input tensor or array.
        :param do_unscaling: Boolean, to apply unscaling on output.
        """

        if self.quadratic_input:
            x = np.sqrt(x)

        if isinstance(x, torch.Tensor):
            x_frd = (x - self.input_min) / (self.input_max - self.input_min)
            out = x_frd.reshape(len(x), 1)

            out = self.model(out)

            if do_unscaling:
                out = out * (self.output_max - self.output_min) + self.output_min

            return out

        else:
            if isinstance(x, np.ndarray):
                x = x.tolist()

            if not isinstance(x, list):
                x_frd = (x - self.input_min) / (self.input_max - self.input_min)
                out = torch.tensor(x_frd, dtype=torch.float32).reshape(1, 1)
            else:
                x_frd = (np.array(x) - self.input_min) / (self.input_max - self.input_min)
                out = torch.tensor(x_frd, dtype=torch.float32).reshape(len(x), 1)

        out = self.model(out)

        if do_unscaling:
            out = out * (self.output_max - self.output_min) + self.output_min

        return out.detach().numpy().reshape((-1,))

    def scale_dataset(self, x, y, do_shuffle=True):
        """
        Scales the dataset and optionally shuffles it.
        :param x: Array, input features.
        :param y: Array, output targets.
        :param do_shuffle: Boolean, whether to shuffle data.
        """
        input_size = x.size

        self.input_min = np.min(x)
        self.input_max = np.max(x)

        self.output_min = np.min(y)
        self.output_max = np.max(y)

        x_res = (x - min(x)) / (max(x) - min(x))
        if max(y) - min(y) != 0:
            y_res = (y - min(y)) / (max(y) - min(y))
        else:
            y_res = np.zeros(len(y))

        if do_shuffle:
            x_res, y_res = shuffle(x_res, y_res, random_state=42)

        return torch.Tensor(x_res).reshape(input_size, 1), torch.Tensor(y_res).reshape(input_size, 1)

    def custom_loss_energy_to_stress_with_fixed_std(model, inputs, targets, std_train_stress):
        inputs.requires_grad = True
        energy_outputs = model(inputs)  # Predicted energy (W)
        # Compute stress as the derivative of energy with respect to strain
        stress_outputs = torch.autograd.grad(
            outputs=energy_outputs, 
            inputs=inputs,
            grad_outputs=torch.ones_like(energy_outputs),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]  
        # Compute the stress at reference input (epsilon = 0)
        reference_input = torch.zeros_like(inputs, requires_grad=True)  # Set requires_grad=True
        reference_energy_output = model(reference_input)
        reference_stress = torch.autograd.grad(
            outputs=reference_energy_output, 
            inputs=reference_input,
            grad_outputs=torch.ones_like(reference_energy_output),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        normalized_error = (stress_outputs - reference_stress - targets) / std_train_stress
        loss = torch.sum(normalized_error ** 2)  # Mean of squared normalized error
        return loss

    def convex_training(self, input_data, target_data, epochs=25000, epsilon=30, learning_rate=0.01, do_convex_training=True):
        """
        Trains the model using convex optimization.
        :param do_convex_training: apply convexity and monotonicity constraint. Default value is True
        :param input_data: Array, the input dataset.
        :param target_data: Array, the target dataset.
        :param epochs: Integer, number of training epochs.
        :param epsilon: Float, parameter for convexity constraint.
        :param learning_rate: Float, learning rate for the optimizer.
        """
        std_train_stress = torch.std(target_data).item()
        input_scaled, target_scaled = self.scale_dataset(input_data, target_data)

        # Split data into training and validation sets
        n_samples = len(input_scaled)
        train_indices = list(range(n_samples // 2))
        val_indices = [i for i in range(n_samples) if i not in train_indices]

        x_train, y_train = input_scaled[train_indices], target_scaled[train_indices]
        x_val, y_val = input_scaled[val_indices], target_scaled[val_indices]

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=1500)

        for epoch in range(epochs):
            # Forward pass
            train_output = self.model(x_train)
            val_output = self.model(x_val)

            # Compute loss
            train_loss = custom_loss_energy_to_stress_with_fixed_std(self.model, train_output, y_train, std_train_stress)
            val_loss = custom_loss_energy_to_stress_with_fixed_std(self.model, val_output, y_val, std_train_stress)

            # Log training process
            if (epoch + 1) % 500 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}] || "
                      f"Train Loss: {train_loss.item():.9f} || "
                      f"Val Loss: {val_loss.item():.9f} || "
                      f"LR: {optimizer.param_groups[0]['lr']:.0e}")

            # Learning rate scheduler step
            scheduler.step(val_loss)

            # Backward and optimize
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if do_convex_training:
                # Apply convexity constraint
                self._apply_convexity_constraint(epsilon)

                # Apply monotonic decreasing constraint
                # self._apply_monotonic_decreasing_constraint()

            if optimizer.param_groups[0]['lr'] <= 9e-08:
                print('LR lower than threshold')
                break

    def _apply_monotonic_decreasing_constraint(self):
        """
        Applies a monotonic decreasing constraint to the first layer weights.
        """
        first_layer = self.model[0]
        first_layer.weight.data[first_layer.weight.data > 0] = 0

    def _apply_convexity_constraint(self, epsilon):
        """
        Applies a convexity constraint to the model parameters.
        :param epsilon: Float, parameter for convexity constraint.
        """
        for name, param in self.model.named_parameters():
            if "weight" in name and "0.weight" not in name:
                param.data[param < 0] = torch.exp(param[param < 0] - epsilon)

    def __str__(self):
        print("#---- Neural Network architecture -------------------#")
        for key, layer in self.nn_architecture_dict.items():
            print(f'{key}: {layer}')
        print("#----------------------------------------------------#")
        return ""