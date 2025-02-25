import torch
# import nestedtensor as nt

def to_cuda_nestedtensor(ntensor, device):
    # 遍历嵌套张量中的每个张量并将其移动到指定设备
    if isinstance(ntensor, nt.NestedTensor):
        return ntensor.to(device)  # 不支持 non_blocking 参数
    elif isinstance(ntensor, torch.Tensor):
        return ntensor.to(device, non_blocking=True)
    elif isinstance(ntensor, list):
        return [to_cuda_nestedtensor(t, device) for t in ntensor]
    elif isinstance(ntensor, dict):
        return {k: to_cuda_nestedtensor(v, device) for k, v in ntensor.items()}
    else:
        return ntensor  # 其他类型直接返回
def to_cuda(samples, targets, device):
    # samples = samples.to(device, non_blocking=True)
    # samples = to_cuda_nestedtensor(samples, device)
    samples = samples.to(device)
    # targets = targets.to(device, non_blocking=True)
    targets=[{k: v.to(device) for k, v in t.items()} for t in targets]
    return samples, targets


class data_prefetcher:
    def __init__(self, loader, device, prefetch=True):
        """
        The purpose of this class is to preload data from a loader object
          and move it to the device (GPU) for faster processing.
        """
        self.loader = iter(loader)
        self.prefetch = prefetch
        self.device = device
        if prefetch:
            self.stream = torch.cuda.Stream()
            self.preload()

    def preload(self):
        try:
            self.next_samples, self.next_targets = next(self.loader)
        except (
            StopIteration
        ):  # If all the elements have been feteched, calling the next() method again will throw a StopIteration exception.
            self.next_samples = None
            self.next_targets = None
            return
        with torch.cuda.stream(self.stream):
            # print("self.next_samples: ", self.next_samples)
            # print("self.next_targets: ", self.next_targets)
            self.next_samples, self.next_targets = to_cuda(
                self.next_samples, self.next_targets, self.device
            )

    def next(self):
        if self.prefetch:
            torch.cuda.current_stream().wait_stream(self.stream)
            samples = self.next_samples
            targets = self.next_targets
            if samples is not None:
                samples.record_stream(torch.cuda.current_stream())
            if targets is not None:
                targets.record_stream(torch.cuda.current_stream())
            self.preload()
        else:
            try:
                samples, targets = next(self.loader)
                # print("samples: ", samples)
                # print("targets: ", targets)
                samples, targets = to_cuda(samples, targets, self.device)
            except StopIteration:
                samples = None
                targets = None
        return samples, targets
