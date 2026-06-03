"""Lightweight on-device GPU utilization / power sampler for Apple Silicon (KLU-106).

Saturation evidence for the Mac-tier training runs. Two readers, in order of availability:

  * **GPU Device Utilization %** — ``ioreg -r -c IOAccelerator`` exposes ``Device Utilization %``
    (and ``In use System Memory``) with **no root** required. Sampled in a background thread during
    training, summarized as busy-sample mean / peak (the KLU-54 doc used the same source).
  * **Sustained package/GPU power (W)** — ``sudo powermetrics --samplers gpu_power`` is the only
    power reader on the standard box, and it needs root. When passwordless sudo is available we
    sample it; otherwise we record the exact command so a human can capture it on an interactive
    run (we never fabricate a power number).

The sampler is best-effort: any failure degrades to "not captured" rather than crashing a training
run. It is deliberately dependency-free (subprocess + threading only).
"""

from __future__ import annotations

import re
import subprocess
import threading

POWERMETRICS_CMD = "sudo powermetrics --samplers gpu_power -i 1000 -n 5"

_UTIL_RE = re.compile(r'"Device Utilization %"\s*=\s*(\d+)')


def _read_gpu_util_pct() -> int | None:
    """One ``Device Utilization %`` sample from IOAccelerator (no root). None if unavailable."""
    try:
        out = subprocess.run(
            ["ioreg", "-r", "-c", "IOAccelerator", "-d", "1"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except Exception:
        return None
    vals = [int(m) for m in _UTIL_RE.findall(out)]
    return max(vals) if vals else None


def _sudo_noninteractive() -> bool:
    try:
        return subprocess.run(["sudo", "-n", "true"], capture_output=True, timeout=5).returncode == 0
    except Exception:
        return False


def _sample_power_w() -> float | None:
    """Sustained GPU/package power (W) via powermetrics, only if passwordless sudo works."""
    if not _sudo_noninteractive():
        return None
    try:
        out = subprocess.run(
            ["sudo", "-n", "powermetrics", "--samplers", "gpu_power", "-i", "1000", "-n", "5"],
            capture_output=True, text=True, timeout=30,
        ).stdout
    except Exception:
        return None
    # powermetrics prints e.g. "GPU Power: 71 mW" or "Combined Power (CPU + GPU + ANE): 12345 mW".
    watts = []
    for m in re.finditer(r"GPU Power:\s*([\d.]+)\s*mW", out):
        watts.append(float(m.group(1)) / 1000.0)
    return (sum(watts) / len(watts)) if watts else None


class GpuUtilSampler:
    """Background thread sampling GPU ``Device Utilization %`` every ``interval`` seconds.

    Usage::

        s = GpuUtilSampler(); s.start()
        ...train...
        report = s.stop()   # {"available", "samples", "busy_mean", "peak", ...}

    "busy" samples are those above ``busy_floor`` (default 5%) — the active-training window — so the
    mean isn't dragged down by idle gaps between phases. Returns a JSON-able summary dict.
    """

    def __init__(self, interval: float = 2.0, busy_floor: int = 5) -> None:
        self.interval = interval
        self.busy_floor = busy_floor
        self._samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._available = _read_gpu_util_pct() is not None

    def _loop(self) -> None:
        while not self._stop.is_set():
            v = _read_gpu_util_pct()
            if v is not None:
                self._samples.append(v)
            self._stop.wait(self.interval)

    def start(self) -> GpuUtilSampler:
        if self._available:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval * 2)
        busy = [s for s in self._samples if s >= self.busy_floor]
        report: dict = {
            "source": "ioreg IOAccelerator 'Device Utilization %' (no root)",
            "available": self._available,
            "n_samples": len(self._samples),
            "n_busy_samples": len(busy),
            "busy_mean_pct": round(sum(busy) / len(busy), 1) if busy else None,
            "peak_pct": max(self._samples) if self._samples else None,
        }
        power = _sample_power_w()
        if power is not None:
            report["sustained_gpu_power_w"] = round(power, 1)
            report["power_source"] = "sudo powermetrics --samplers gpu_power"
        else:
            report["sustained_gpu_power_w"] = None
            report["power_note"] = (
                "not captured (needs passwordless sudo); run alongside training: " + POWERMETRICS_CMD
            )
        return report


def peak_mps_mem_gb() -> float | None:
    """Peak MPS driver-allocated unified memory in GB (the GPU working set), if on MPS."""
    try:
        import torch

        if torch.backends.mps.is_available():
            return torch.mps.driver_allocated_memory() / (1024**3)
    except Exception:
        pass
    return None


def total_unified_mem_gb() -> float | None:
    try:
        return int(
            subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip()
        ) / (1024**3)
    except Exception:
        return None


__all__ = ["GpuUtilSampler", "peak_mps_mem_gb", "total_unified_mem_gb", "POWERMETRICS_CMD"]
