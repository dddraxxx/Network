import torch
from torch.nn.modules import batchnorm

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        print('loading batch')
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            batch   = []
            for k in self.batch:
                if isinstance(k, torch.Tensor):
                    batch.append(k.cuda(non_blocking=True))
            self.batch  = batch
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            #     self.next_input = self.next_input.float()
        print('loading finished')

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch