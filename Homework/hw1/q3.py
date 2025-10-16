###Q3: allreduce###
###please implement ring_allreduce method, using  pytorch's dist method is not allowed###

from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch
import torch.distributed as dist
import torch.nn.functional as F

def reduce_scatter(chunks, tmp, world, rank, left, right):
    # your code here: follow slides instruction: do counter-clockwise iteration

    send_idx = rank
    for r in range(world - 1):
        recv_idx = (send_idx - 1) % world

        # send self and receive from left
        s_req = dist.isend(chunks[send_idx], dst=right)
        r_req = dist.irecv(tmp, src=left)

        # update self
        r_req.wait()
        chunks[recv_idx].add_(tmp)

        # wait for send to finish
        s_req.wait()

        # rotate to send the just-updated chunk
        send_idx = recv_idx
        
def all_gather(chunks, tmp, world, rank, left, right):
    # your code here: follow slides instruction: do counter-clockwise iteration

    send_idx = (rank + 1) % world
    for r in range(world - 1):
        recv_idx = (send_idx - 1) % world

        # send self and receive from left
        s_req = dist.isend(chunks[send_idx], dst=right)
        r_req = dist.irecv(tmp, src=left)

        # update self
        r_req.wait()
        chunks[recv_idx].copy_(tmp)

        # wait for send to finish
        s_req.wait()

        # rotate to send the just-updated chunk
        send_idx = recv_idx


def ring_allreduce_(tensor: torch.Tensor, world_size = None, rankid = None):
    """In-place ring all-reduce (SUM, optional average) using isend/irecv."""
    world = world_size
    if world == 1: return tensor
    rank = rankid
    left, right = (rank - 1) % world, (rank + 1) % world

    ##following steps try to fill blank to the tensor so that final tensor can be divided to 3 chunks evenly
    flat = tensor.contiguous().view(-1)
    n = flat.numel()
    chunk = (n + world - 1) // world
    # your code here: we cannot divide flat into 3 pieces evenly as the
    # flat lengh may not be able to divided exactly by 3....
    #So, fill zeros at the end of flat to generate padded_flat
    padded_flat = F.pad(flat, (0, chunk * world - n), mode='constant', value=0)

    chunks = [padded_flat[i*chunk:(i+1)*chunk] for i in range(world)]

    # your code here: call reduce_scatter and all_gather
    tmp = torch.empty_like(chunks[0])
    reduce_scatter(chunks, tmp, world, rank, left, right)
    all_gather(chunks, tmp, world, rank, left, right)

    # we provide the reduce_scatter and all_gather func prototype for you
    # You may adjust the function signature (input structure) of `reduce_scatter` and `all_gather` if needed.
    
    # stitch & unpad  
    flat = torch.cat(chunks, dim=0)[:n]
    flat /= world
    tensor.view(-1).copy_(flat)
    return tensor