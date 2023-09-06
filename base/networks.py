import torch
import torch.nn as nn
import numpy as np

def get_network(cfg, in_features, out_features):
    if cfg.network == 'siren':
        return MLP(in_features, out_features, cfg.num_hidden_layers,
            cfg.hidden_features,omega_0=cfg.omega_0, nonlinearity=cfg.nonlinearity)
    elif cfg.network == 'positional':
        return MLPPositional(in_features, out_features, cfg)
    else:
        raise NotImplementedError
    


class HarmonicEmbedding(torch.nn.Module):
    def __init__(self, n_harmonic_functions=60, omega_0=1):
        """
        Given an input tensor `x` of shape [minibatch, ... , dim],
        the harmonic embedding layer converts each feature
        in `x` into a series of harmonic features `embedding`
        as follows:
            embedding[..., i*dim:(i+1)*dim] = [
                sin(x[..., i]),
                sin(2*x[..., i]),
                sin(4*x[..., i]),
                ...
                sin(2**(self.n_harmonic_functions-1) * x[..., i]),
                cos(x[..., i]),
                cos(2*x[..., i]),
                cos(4*x[..., i]),
                ...
                cos(2**(self.n_harmonic_functions-1) * x[..., i])
            ]
            
        Note that `x` is also premultiplied by `omega_0` before
        evaluating the harmonic functions.
        """
        super().__init__()
        self.register_buffer(
            'frequencies',
            omega_0 * (2.0 ** torch.arange(n_harmonic_functions)),
        )
    def forward(self, x):
        """
        Args:
            x: tensor of shape [..., dim]
        Returns:
            embedding: a harmonic embedding of `x`
                of shape [..., n_harmonic_functions * dim * 2]
        """
        embed = (x[..., None] * self.frequencies).view(*x.shape[:-1], -1)
        return torch.cat((embed.sin(), embed.cos()), dim=-1)
    

class MLPPositional(nn.Module):
    def __init__(self, in_features, out_features, cfg):
        super().__init__()

        num_positional_encoding = cfg.n_harmonic_functions
        
        self.positional_encoding_layer = HarmonicEmbedding(n_harmonic_functions=num_positional_encoding, omega_0 = cfg.omega_0)

        n_embeding_dims = num_positional_encoding * 2 * in_features
        
        self.net = MLP(n_embeding_dims, out_features, cfg.num_hidden_layers,
            cfg.hidden_features)

    def forward(self, coords, weights=None):

        encoded_coords = self.positional_encoding_layer(coords)
        output = self.net(encoded_coords)
        if weights is not None:
            output = output * weights
        return output

############################### SIREN ################################
class Sine(nn.Module):
    def __init__(self, omega_0):
        super().__init__()
        self.omega_0 = omega_0

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(self.omega_0 * input)


class MLP(nn.Module):
    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,omega_0 = None,
                 outermost_linear=True, nonlinearity='relu', weight_init=None):
        super().__init__()

        self.first_layer_init = None
        self.omega_0 = omega_0

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'sine':(Sine(self.omega_0), lambda m: sine_init(m,self.omega_0), first_layer_sine_init),
                         'relu':(nn.ReLU(inplace=True), init_weights_normal, None),
                         'elu':(nn.ELU(inplace=True), init_weights_elu, None)}

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        self.net = []
        self.net.extend([nn.Linear(in_features, hidden_features), nl])

        for i in range(num_hidden_layers):
            self.net.extend([nn.Linear(hidden_features, hidden_features), nl])

        self.net.append(nn.Linear(hidden_features, out_features))
        if not outermost_linear:
            self.net.append(nl)

        self.net = nn.Sequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
            self.net[0].apply(first_layer_init)

    def forward(self, coords, weights=None):
        output = self.net(coords)
        if weights is not None:
            output = output * weights
        return output


def init_weights_normal(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def sine_init(m, omega_0):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / omega_0, np.sqrt(6 / num_input) / omega_0)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def init_weights_elu(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=np.sqrt(1.5505188080679277) / np.sqrt(num_input))
