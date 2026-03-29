## Adapted from TD-MPC2
import copy

import torch
import torch.nn as nn
from torch.utils._pytree import tree_flatten, tree_unflatten

class Ensemble(nn.Module):
    """
    Vectorized ensemble of modules. Adds 1 extra dimension.
    """

    def __init__(self, modules, device=None, **vmap_kwargs):
        super().__init__()
        self.modules_list = nn.ModuleList(modules)

        if device is None:
            try:
                device = next(self.modules_list[0].parameters()).device
            except StopIteration:
                device = torch.device("cpu")

        for m in self.modules_list:
            m.to(device)

        base_model = copy.deepcopy(self.modules_list[0]).to("meta")

        params_tree, _ = torch.func.stack_module_state(self.modules_list)
        flat_params, self._param_spec = tree_flatten(params_tree)

        # The actual trainable parameters of the ensemble
        self._ens_params = nn.ParameterList([nn.Parameter(p) for p in flat_params])
        flat_params_registered = list(self._ens_params)
        self._params_tree = tree_unflatten(flat_params_registered, self._param_spec)

        # single-model functional call (captures base_model but doesn't register it)
        def _call_single_model(params, buffers, x):
            return torch.func.functional_call(base_model, (params, buffers), (x,))

        # vectorize across the ensemble dimension
        self._vmapped = torch.vmap(
            _call_single_model,
            in_dims=(0, 0, None),
            randomness="different",
            **vmap_kwargs,
        )

    def forward(self, x):
        return self._vmapped(self._params_tree, {}, x)

    def __repr__(self):
        t = self.modules_list[0].__class__.__name__
        return f"VectorizedEnsemble({len(self.modules_list)} x {t})"

    def __len__(self):
        return len(self.modules_list)

    def __getitem__(self, idx):
        return self.modules_list[idx]


class NormedLinear(nn.Linear):
    """
    Linear layer with LayerNorm, activation, and optionally dropout. Don't care batch
    """

    def __init__(self, *args, dropout=0., act=nn.Mish(inplace=True), **kwargs):
        super().__init__(*args, **kwargs)
        self.ln = nn.LayerNorm(self.out_features)
        self.act = act
        self.dropout = nn.Dropout(dropout, inplace=True) if dropout else None

    def forward(self, x):
        x = super().forward(x)
        if self.dropout:
            x = self.dropout(x)
        return self.act(self.ln(x))
    
    def __repr__(self):
        repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
        return f"NormedLinear(in_features={self.in_features}, "\
            f"out_features={self.out_features}, "\
            f"bias={self.bias is not None}{repr_dropout}, "\
            f"act={self.act.__class__.__name__})"


def mlp(in_dim, mlp_dims, out_dim, final_act=None, dropout=0.):
    """
    Basic building block borrowed from TD-MPC2.
    MLP with LayerNorm, Mish activations, and optionally dropout.
    """
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]
    dims = [in_dim] + mlp_dims + [out_dim]
    mlp = nn.ModuleList()
    for i in range(len(dims) - 2):
        mlp.append(NormedLinear(dims[i], dims[i+1], dropout=dropout*(i==0)))
    mlp.append(NormedLinear(dims[-2], dims[-1], act=final_act) if final_act else nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*mlp)

def weight_init(m):
    """Custom weight initialization"""
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.ParameterList):
        for i,p in enumerate(m):
            if p.dim() == 3: # Linear
                nn.init.trunc_normal_(p, std=0.02) # Weight
                nn.init.constant_(m[i+1], 0) # Bias



