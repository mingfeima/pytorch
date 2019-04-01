import torch
import torch.nn as nn


mkldnn_supported_op = {
    nn.Conv2d,
    nn.ConvTranspose2d,
}

def is_available():
    r"""Returns whether PyTorch is built with MKL-DNN support."""
    return torch._C.has_mkldnn
