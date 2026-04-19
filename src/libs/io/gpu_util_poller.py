"""GpuUtilPoller — NVML background thread sampling GPU utilization, memory, power."""

import threading
import time
from typing import List, Tuple

try:
    import pynvml
    _NVML_AVAILABLE = True
except ImportError:
    _NVML_AVAILABLE = False


class GpuUtilPoller:
    """Context manager that polls NVML in a background thread.

    Usage:
        with GpuUtilPoller(interval_ms=100) as p:
            run_simulation()
        print(p.summary())
    """

    def __init__(
        self,
        interval_ms: int = 100,
        device_idx: int = 0,
        live_print: bool = False,
        live_print_interval_ms: int = 500,
    ):
        if not _NVML_AVAILABLE:
            raise RuntimeError(
                "pynvml not installed. Install with: pip install nvidia-ml-py")
        self.interval_s = interval_ms / 1000.0
        self.device_idx = device_idx
        self.samples: List[Tuple[float, int, int, float]] = []
        self._stop = threading.Event()
        self._thread = None
        self._t0 = None
        self._handle = None
        self._live_print = live_print
        self._live_print_interval_s = live_print_interval_ms / 1000.0
        self._last_print_t = 0.0
        self._printed_anything = False

    def __enter__(self):
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_idx)
        self._t0 = time.perf_counter()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._printed_anything:
            print("", flush=True)
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    def _loop(self):
        while not self._stop.is_set():
            try:
                u = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                m = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                p_mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)
                now = time.perf_counter()
                t_ms = (now - self._t0) * 1000.0
                self.samples.append(
                    (t_ms, int(u.gpu), int(m.used // (1024 * 1024)), p_mw / 1000.0))
                if self._live_print and (now - self._last_print_t) >= self._live_print_interval_s:
                    self._emit_live(now, int(u.gpu), int(m.used // (1024 * 1024)), p_mw / 1000.0)
                    self._last_print_t = now
            except Exception:
                pass
            time.sleep(self.interval_s)

    def _emit_live(self, now: float, util: int, mem_mib: int, power_w: float) -> None:
        elapsed = now - self._t0
        if self.samples:
            window = [s[1] for s in self.samples[-20:]]
            util_mean = sum(window) / len(window)
        else:
            util_mean = util
        print(
            f"\r[gpu-lab t={elapsed:6.1f}s  util_now={util:3d}%  "
            f"util_avg20={util_mean:5.1f}%  mem={mem_mib:5d}MiB  pow={power_w:5.1f}W]",
            end="",
            flush=True,
        )
        self._printed_anything = True

    def summary(self) -> dict:
        if not self.samples:
            return dict(n_samples=0, mean_util=0.0, peak_util=0,
                        mean_mem_mib=0.0, peak_mem_mib=0,
                        mean_power_w=0.0, peak_power_w=0.0)
        utils = [s[1] for s in self.samples]
        mems = [s[2] for s in self.samples]
        pows = [s[3] for s in self.samples]
        return dict(
            n_samples=len(self.samples),
            mean_util=sum(utils) / len(utils),
            peak_util=max(utils),
            mean_mem_mib=sum(mems) / len(mems),
            peak_mem_mib=max(mems),
            mean_power_w=sum(pows) / len(pows),
            peak_power_w=max(pows),
        )

    def write_timeline_tsv(self, path: str) -> None:
        with open(path, "w") as f:
            f.write("t_ms\tgpu_util_pct\tmem_used_mib\tpower_w\n")
            for (t_ms, u, m, p) in self.samples:
                f.write(f"{t_ms:.1f}\t{u}\t{m}\t{p:.2f}\n")
