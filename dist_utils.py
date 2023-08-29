import torch
import torch.distributed as dist


def cleanup() -> None:
    dist.destroy_process_group()


def initialized():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if initialized() else 0


def get_world_size():
    return dist.get_world_size() if initialized() else 1


def is_master():
    return get_rank() == 0


def synchronize() -> None:
    if get_world_size() == 1:
        return
    dist.barrier()


def broadcast(
    tensor: torch.Tensor, src, group=dist.group.WORLD, async_op: bool = False
) -> None:
    if get_world_size() == 1:
        return
    dist.broadcast(tensor, src, group, async_op)


def all_reduce(
    data, group=dist.group.WORLD, average: bool = False, device=None
):
    if get_world_size() == 1:
        return data
    tensor = data
    if not isinstance(data, torch.Tensor):
        tensor = torch.tensor(data)
    if device is not None:
        tensor = tensor.cuda(device)
    dist.all_reduce(tensor, group=group)
    if average:
        tensor /= get_world_size()
    if not isinstance(data, torch.Tensor):
        result = tensor.cpu().numpy() if tensor.numel() > 1 else tensor.item()
    else:
        result = tensor
    return result


def all_gather(data, group=dist.group.WORLD, device=None):
    if get_world_size() == 1:
        return data
    tensor = data
    if not isinstance(data, torch.Tensor):
        tensor = torch.tensor(data)
    if device is not None:
        tensor = tensor.cuda(device)
    tensor_list = [
        tensor.new_zeros(tensor.shape) for _ in range(get_world_size())
    ]
    dist.all_gather(tensor_list, tensor, group=group)
    if not isinstance(data, torch.Tensor):
        result = [tensor.cpu().numpy() for tensor in tensor_list]
    else:
        result = tensor_list
    return result
