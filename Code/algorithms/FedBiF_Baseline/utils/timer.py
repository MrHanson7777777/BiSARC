import time
import torch

class FLTimer:
    def __init__(self):
        self.pack_time = 0.0
        self.unpack_time = 0.0
        self.pack_calls = 0
        self.unpack_calls = 0

    def start(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def record_pack(self, start_time):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.pack_time += (time.perf_counter() - start_time)
        self.pack_calls += 1

    def record_unpack(self, start_time):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.unpack_time += (time.perf_counter() - start_time)
        self.unpack_calls += 1

    def print_stats(self, prefix=""):
        avg_pack = (self.pack_time / self.pack_calls) * 1000 if self.pack_calls > 0 else 0
        avg_unpack = (self.unpack_time / self.unpack_calls) * 1000 if self.unpack_calls > 0 else 0
        print(f"\n[{prefix}] Performance profiling report:")
        print(f"  - Average pack/compression time: {avg_pack:.2f} ms / call")
        print(f"  - Average unpack/decompression time: {avg_unpack:.2f} ms / call\n")

# Global singleton.
global_fl_timer = FLTimer()
