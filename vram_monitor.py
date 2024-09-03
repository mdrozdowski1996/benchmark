import threading
import time

import torch

POLL_RATE_SECONDS = 0.1


class VRamMonitor:
    _thread: threading.Thread
    _device: torch.device
    _vram_usage: set[int] = set()
    _stop_flag: threading.Event
    _lock: threading.Lock

    def __init__(self, device: torch.device):
        self._device = device
        self._stop_flag = threading.Event()
        self._lock = threading.Lock()

        self._thread = threading.Thread(target=self.monitor)
        self._thread.start()

    def monitor(self):
        while not self._stop_flag.is_set():
            vram = torch.cuda.memory_allocated(self._device)
            with self._lock:
                self._vram_usage.add(vram)
            time.sleep(POLL_RATE_SECONDS)

    def complete(self) -> float:
        self._stop_flag.set()
        self._thread.join()
        with self._lock:
            return sum(self._vram_usage) / len(self._vram_usage)
