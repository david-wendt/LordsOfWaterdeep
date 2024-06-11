import torch.nn as nn


ACTIVATION_FNS = {
    'ReLU': nn.ReLU(),
    'LeakyReLU': nn.LeakyReLU()
}

class DeepQNet(nn.Module):
    def __init__(self, 
            input_dim, 
            output_dim, 
            hidden_layer_sizes=[256, 128],
            layernorm='layernorm',
            activation='LeakyReLU'
        ):

        super(DeepQNet, self).__init__()

        print("INITIALIZING DEEP Q NETWORK")
        print("\tInput size (state dim):", input_dim)
        print("\tOutput size (action dim):", output_dim)
        print("\tHidden layer sizes:", hidden_layer_sizes)
        print() # Newline

        layer_sizes = hidden_layer_sizes + [output_dim]
        n_layers = len(layer_sizes)

        layers = [nn.Linear(input_dim, hidden_layer_sizes[0])]
        activation_fn = ACTIVATION_FNS[activation]

        for i in range(n_layers-1):
            if layernorm == 'layernorm' or layernorm is True:
                layers.append(nn.LayerNorm(layer_sizes[i]))
            elif layernorm == 'batchnorm':
                layers.append(nn.BatchNorm1d(layer_sizes[i]))
            else:
                if layernorm is not None:
                    raise ValueError(f'Unknown layer norm: {layernorm}')
                
            layers.append(activation_fn)
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)