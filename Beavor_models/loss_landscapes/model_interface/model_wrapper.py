""" Class used to define interface to complex models """

import abc
import itertools
import torch.nn
from loss_landscapes.model_interface.model_parameters import ModelParameters
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,2,3,4,5,7'

class ModelWrapper(abc.ABC):
    def __init__(self, modules: list):
        self.modules = modules

    def get_modules(self) -> list:
        return self.modules

    def get_module_parameters(self) -> ModelParameters:
        return ModelParameters([p for module in self.modules for p in module.parameters()])
    
    def back_module_parameters(self, start_point, model) ->ModelParameters:
        for module in self.modules:
            for i in range(32):
            # module[0].self_attn.q_proj.weight
                model.model.layers[i].self_attn.q_proj.weight = torch.nn.Parameter(start_point[i*9])
                model.model.layers[i].self_attn.k_proj.weight = torch.nn.Parameter(start_point[i*9+1])
                model.model.layers[i].self_attn.v_proj.weight = torch.nn.Parameter(start_point[i*9+2])
                model.model.layers[i].self_attn.o_proj.weight = torch.nn.Parameter(start_point[i*9+3])
                model.model.layers[i].mlp.gate_proj.weight = torch.nn.Parameter(start_point[i*9+4])
                model.model.layers[i].mlp.down_proj.weight = torch.nn.Parameter(start_point[i*9+5])
                model.model.layers[i].mlp.up_proj.weight = torch.nn.Parameter(start_point[i*9+6])
                model.model.layers[i].input_layernorm.weight = torch.nn.Parameter(start_point[i*9+7])
                model.model.layers[i].post_attention_layernorm.weight = torch.nn.Parameter(start_point[i*9+8])
        return model

    def train(self, mode=True) -> 'ModelWrapper':
        for module in self.modules:
            module.train(mode)
        return self

    def eval(self) -> 'ModelWrapper':
        return self.train(False)

    def requires_grad_(self, requires_grad=True) -> 'ModelWrapper':
        for module in self.modules:
            for p in module.parameters():
                p.requires_grad = requires_grad
        return self

    def zero_grad(self) -> 'ModelWrapper':
        for module in self.modules:
            for p in module.parameters():
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
        return self

    def parameters(self):
        return itertools.chain([module.parameters() for module in self.modules])

    def named_parameters(self):
        return itertools.chain([module.named_parameters() for module in self.modules])

    @abc.abstractmethod
    def forward(self, x):
        pass


class SimpleModelWrapper(ModelWrapper):
    def __init__(self, model: torch.nn.Module):
        super().__init__([model])

    def forward(self):
        # output_ids = self.modules[0](x, max_new_tokens = y.shape[1]-x.shape[1])[0]
        # attention_mask = torch.ones_like(output_ids)
        # attention_mask[:x.shape[1]] = 0
        # return self.modules[0](output_ids.unsqueeze(0), attention_mask = attention_mask.unsqueeze(0)).logits.squeeze(0), output_ids
        # import pdb;pdb.set_trace()
        output_ids = self.modules[0]
        return output_ids


class GeneralModelWrapper(ModelWrapper):
    def __init__(self, model, modules: list, forward_fn):
        super().__init__(modules)
        self.model = model
        self.forward_fn = forward_fn

    def forward(self, x):
        return self.forward_fn(self.model, x)


def wrap_model(model):
    if isinstance(model, ModelWrapper):
        return model.requires_grad_(False)
    elif isinstance(model, torch.nn.Module):
        return SimpleModelWrapper(model).requires_grad_(False)
    else:
        raise ValueError('Only models of type torch.nn.modules.module.Module can be passed without a wrapper.')
