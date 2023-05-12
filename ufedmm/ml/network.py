import torch
import torch.nn as nn


class FE_network(torch.nn.Module):

    def __init__(self, input_dim, width=2, depths=1, acti='Tanh', spectral_norm = False): #, mean=None, std=None, mean_force=None, std_force=None):
        super(FE_network, self).__init__()
        self.input_dim = input_dim * 2
        self.width = width
        self.output_size = 1
        self.depths = depths
        self.input_layer = torch.nn.Linear(self.input_dim, self.width)
        self.layers = [self.input_layer] + [torch.nn.Linear(self.width, self.width) for _ in range(self.depths-1)] + [torch.nn.Linear(self.width, 1, bias=False)]
        if spectral_norm:
            self.layers = [torch.nn.utils.spectral_norm(layer) for layer in self.layers]
        self.layers_params = torch.nn.ModuleList(self.layers)
        if acti == 'Tanh':
            self.acti = torch.nn.Tanh()
        elif acti == 'Softplus':
            self.acti = torch.nn.Softplus()
        elif acti == 'Silu':
            self.acti = torch.nn.SiLU()
        elif acti == 'ELU':
            self.acti = torch.nn.ELU()
        else:
            raise NotImplementedError
        # self.whitening_layer = Mean_std_layer(input_dim, mean, std)
        # self.whitening_layer_model = Mean_std_layer(input_dim)
        # self.whitening_layer_output = Mean_std_layer(input_dim, mean_force, std_force, mode_reverse=True)

        self.relu = nn.ReLU()

    def forward(self, features):
        features = features.requires_grad_(True)
        # feat_white = self.whitening_layer(features)
        x = torch.cat([torch.sin(features), torch.cos(features)], dim=1) # assuming dihedrals
        for layer in self.layers[:-1]:
            x = self.acti(layer(x))
        x = self.layers[-1](x)
        forces = torch.autograd.grad(x, features, grad_outputs=torch.ones_like(x), create_graph=True)[0]
        # forces_white = self.whitening_layer_model(forces_white)
        # forces = self.whitening_layer_output(forces_white)
        return forces
    
    def predict(self, features):
        # feat_white = self.whitening_layer(features)
        x = torch.cat([torch.sin(features), torch.cos(features)], dim=1)
        for layer in self.layers[:-1]:
            x = self.acti(layer(x))
        x = self.layers[-1](x)
        return x


class Mean_std_layer(torch.nn.Module):
    """ Custom Linear layer for substracting the mean and dividing by
        the std

        Parameters
        ----------
        size_in: int
            The input size of which mean should be subtracted
        mean: torch.Tensor
            The mean value of the input training values
        std: torch.Tensor
            The std value of the input training values
     """

    def __init__(self, size_in, mean=None, std=None, mode_reverse=False):
        super().__init__()
        self.size_in = size_in
        if mean is None:
            mean = torch.zeros((1,size_in))
        self.weights_mean = torch.nn.Parameter(mean, requires_grad=False)  # nn.Parameter is a Tensor that's a module parameter.
        if std is None:
            std = torch.ones((1,size_in))
        self.weights_std = torch.nn.Parameter(std, requires_grad=False)
        self.mode_reverse = mode_reverse

    def forward(self, x):
        if self.mode_reverse:
            y = x*self.weights_std+self.weights_mean
        else:
            y = (x-self.weights_mean)/self.weights_std
        return y  
    
    def set_both(self, mean, std):
        new_params = [mean, std]
        with torch.no_grad():
            for i, param in enumerate(self.parameters()):
                new_param = new_params[i]
                param.copy_(torch.Tensor(new_param))