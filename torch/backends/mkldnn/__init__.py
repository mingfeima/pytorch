import os
import ctypes
import sys
import torch
import warnings

# Write:
#
#   torch.backends.mkldnn.enabled = False
#
# to globally disable MKLDNN

lib = None

def find_mkldnn_windows_lib():
    proc = Popen(['where', 'mkldnn*.dll'], stdout=PIPE, stderr=PIPE)
    out, err = proc.communicate()
    out = out.decode().strip()
    if len(out) > 0:
        if out.find('\r\n') != -1:
            out = out.split('\r\n')[0]
        mkldnn_lib_name = os.path.basename(out)
        mkldnn_lib = os.path.splitext(mkldnn_lib_name)[0]
        mkldnn_lib = str(mkldnn_lib)
        return ctypes.cdll.LoadLibrary(mkldnn_lib)
    else:
        return None

def _libmkldnn():
    global lib
    if lib is None:
        if sys.platform == "win32":
            lib = find_mkldnn_windows_lib()
        else:
            lib = ctypes.cdll.LoadLibrary(None)
        if not hasattr(lib, 'mkldnn_stream_create'):
            lib = None
    return lib

MKLDNN_TENSOR_TYPES = {
    'torch.FloatTensor',
}

def is_acceptable(tensor):
    if not torch._C._get_mkldnn_enabled():
        return False
    if tensor.type() not in MKLDNN_TENSOR_TYPES:
        return False
    if not torch._C.has_mkldnn:
        warnings.warn(
            "PyTorch was compiled without MKLDNN support. To use MKLDNN, rebuild"
            "PyTorch making sure the library is visible to the build system.")
        return False
    if _libmkldnn() is None:
        warnings.warn('MKLDNN library not found. Check your {libpath}'.format(
            libpath={
                'darwin': 'DYLD_LIBRARY_PATH',
                'win32': 'PATH'
            }.get(sys.platform, 'LD_LIBRARY_PATH')))
        return False
    return True

MKLDNN_RNN_RELU = 0
MKLDNN_RNN_TANH = 1
MKLDNN_LSTM = 2
MKLDNN_GRU = 3

def get_rnn_mode(mode):
    if mode == 'RNN_RELU':
        return MKLDNN_RNN_RELU
    elif mode == 'RNN_TANH':
        return MKLDNN_RNN_TANH
    elif mode == 'LSTM':
        return MKLDNN_LSTM
    elif mode == 'GRU':
        return MKLDNN_GRU
    else:
        raise Exception("Unknown mode: {}".format(mode))

# NB: We don't have  RNN_RELU or RNN_TANH at the moment
def is_rnn_acceptable(*args, **kwargs):
    _SUPPORTED_MODE = {'LSTM', 'GRU'}
    mode = args[0]
    if mode not in _SUPPORTED_MODE:
        return False
    return True
