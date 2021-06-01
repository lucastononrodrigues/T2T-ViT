import torch
from packaging import version
import numpy as np
import os
import torch.distributed as dist

if version.Version(torch.__version__) >= version.Version('1.0.0'):
  from torch import _softmax_backward_data as _softmax_backward_data
else:
  from torch import softmax_backward_data as _softmax_backward_data



class XSoftmax(torch.autograd.Function):
  """ Masked Softmax which is optimized for saving memory
  Args:
      
    input (:obj:`torch.tensor`): The input tensor that will apply softmax.
    mask (:obj:`torch.IntTensor`): The mask matrix where 0 indicate that element will be ignored in the softmax caculation.
    dim (int): The dimenssion that will apply softmax.
    
  Example::
    import torch
    from DeBERTa.deberta import XSoftmax
    # Make a tensor
    x = torch.randn([4,20,100])
    # Create a mask
    mask = (x>0).int()
    y = XSoftmax.apply(x, mask, dim=-1)
      
  """

  @staticmethod
  def forward(self, input, mask, dim):
    """
    """

    self.dim = dim
    if version.Version(torch.__version__) >= version.Version('1.2.0a'):
      rmask = ~(mask.bool())
    else:
      rmask = (1-mask).byte() # This line is not supported by Onnx tracing.

    output = input.masked_fill(rmask, float('-inf'))
    output = torch.softmax(output, self.dim)
    output.masked_fill_(rmask, 0)
    self.save_for_backward(output)
    return output

  @staticmethod
  def backward(self, grad_output):
    """
    """

    output, = self.saved_tensors
    inputGrad = _softmax_backward_data(grad_output, output, self.dim, output)
    return inputGrad, None, None



def build_relative_position(query_size, key_size, bucket_size=-1, max_position=-1):
  q_ids = np.arange(0, query_size)
  k_ids = np.arange(0, key_size)
  rel_pos_ids = q_ids[:, None] - np.tile(k_ids, (q_ids.shape[0],1))
  if bucket_size>0 and max_position > 0:
    rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size, max_position)
  rel_pos_ids = torch.tensor(rel_pos_ids, dtype=torch.long)
  rel_pos_ids = rel_pos_ids[:query_size, :]
  rel_pos_ids = rel_pos_ids.unsqueeze(0)
  return rel_pos_ids

def make_log_bucket_position(relative_pos, bucket_size, max_position):
  sign = np.sign(relative_pos)
  mid = bucket_size//2
  abs_pos = np.where((relative_pos<mid) & (relative_pos > -mid), mid-1, np.abs(relative_pos))
  log_pos = np.ceil(np.log(abs_pos/mid)/np.log((max_position-1)/mid) * (mid-1)) + mid
  bucket_pos = np.where(abs_pos<=mid, relative_pos, log_pos*sign).astype(np.int)
  return bucket_pos



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
    

def init_distributed_mode(args):
    
    print('Setting up distributed mode...')
    args.num_gpu = 1
    args.device = 'cuda:%d' % args.local_rank
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    args.world_size = torch.distributed.get_world_size()
    args.rank = torch.distributed.get_rank()
    print('Distributed mode set...')
    

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
    
    
    
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()
