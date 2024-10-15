import torch
from torch.autograd import grad
from collections import OrderedDict
from torchmeta.modules import MetaModule
import torch
from torch.autograd import grad
from collections import OrderedDict
from torchmeta.modules import MetaModule

def stiefelmanifold_update_parameters(model, loss, params=None, step_size=0.5, first_order=False):
    if not isinstance(model, MetaModule):
        raise ValueError('The model must be an instance of `torchmeta.modules.MetaModule`, got `{0}`'.format(type(model)))

    if params is None:
        params = OrderedDict(model.meta_named_parameters())

    grads = torch.autograd.grad(loss, params.values(), create_graph=not first_order)

    updated_params = OrderedDict()

    for (name, param), grad in zip(params.items(), grads):
        original_shape = param.shape

        if len(param.shape) == 4:
            batch_size, in_channels, height, width = param.shape
            param_reshaped = param.view(batch_size * in_channels, height * width)
            grad_reshaped = grad.view(batch_size * in_channels, height * width)
        elif len(param.shape) == 2:
            param_reshaped = param
            grad_reshaped = grad
        elif len(param.shape) == 1:
            param_reshaped = param.unsqueeze(1)
            grad_reshaped = grad.unsqueeze(1)

        eye = torch.eye(param_reshaped.size(0), device=param.device)
        param_t_grad = param_reshaped.t() @ grad_reshaped
        riemann_grad = (eye - param_reshaped @ param_reshaped.t()) @ grad_reshaped
        riemann_grad += param_reshaped @ skew(param_t_grad)
        riemann_grad = riemann_grad / 2

        if name.startswith('classifier'):
            qf = param - step_size * grad
        else:
            z = -step_size * riemann_grad
            qf, _ = torch.linalg.qr(param_reshaped + z)

        if len(original_shape) == 4:
            qf = qf.view(batch_size, in_channels, height, width)
        elif len(original_shape) == 2:
            qf = qf.view(original_shape)
        elif len(original_shape) == 1:
            if qf.dim() == 2 and qf.shape[1] == 1:
                qf = qf.squeeze(1)

        updated_params[name] = qf

    return updated_params

def skew(A):

    return (A - A.t()) / 2

def kernel_function(X, Y, sigma=1.0):
    diff = X - Y
    distance_squared = torch.sum(diff * diff, dim=-1)
    return torch.exp(-distance_squared / (2 * sigma ** 2))

def kernel_loss(output, target, sigma=1.0):
    return 1 - kernel_function(output, F.one_hot(target, num_classes=output.size(-1)).float(), sigma)
