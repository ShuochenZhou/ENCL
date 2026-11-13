import os
import time
import json
import threading
import pwd

class ResourceMeter:

    def __init__(
        self,
        gpu_index,
        out_path,
        sample_hz=20.0,
        idle_seconds=20.0,
        detailed_sharing=False,
        emit_clean_fields=False,
        energy_exclusivity_required=True,
    ):
        import pynvml
        self.nvml = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
        self.gpu_index = int(gpu_index)
        try:
            self.gpu_name = pynvml.nvmlDeviceGetName(self.handle)
            if isinstance(self.gpu_name, bytes):
                self.gpu_name = self.gpu_name.decode()
        except Exception:
            self.gpu_name = "unknown"
        self.pid = os.getpid()
        self.out_path = out_path
        self.idle_seconds = float(idle_seconds)
        self.detailed_sharing = bool(detailed_sharing)
        self.emit_clean_fields = bool(emit_clean_fields)
        self.energy_exclusivity_required = bool(energy_exclusivity_required)

        self._energy_supported = self._read_energy_mj() is not None

        self.train_energy_mj = 0.0
        self.train_time_s = 0.0
        self.peak_proc_mem_mb = 0.0
        self.peak_torch_alloc_mb = 0.0
        self.peak_train_power_w = 0.0
        self.in_train = False
        self._e0 = None
        self._t0 = None
        self.per_task = []

        self.concurrent_compute_pids_seen = set()
        self.concurrent_compute_processes_seen = {}
        self.gpu_sharing_events_path = os.path.join(
            os.path.dirname(self.out_path), "gpu_sharing_events.jsonl"
        )

        self._samples = []

        self.idle_power_w = self._measure_idle(self.idle_seconds)

        self._stop = False
        self._state_lock = threading.Lock()
        self._sample_dt = 1.0 / float(sample_hz)
        self.sample_hz = float(sample_hz)
        self._thr = threading.Thread(target=self._sample_loop, daemon=True)
        self._thr.start()

    def _read_energy_mj(self):
        try:
            return float(self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle))
        except Exception:
            return None

    def _read_power_w(self):
        try:
            return float(self.nvml.nvmlDeviceGetPowerUsage(self.handle)) / 1000.0
        except Exception:
            return None

    def _read_proc_mem_mb(self):
        mb = 0.0
        try:
            for p in self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle):
                if p.pid == self.pid and p.usedGpuMemory:
                    mb = float(p.usedGpuMemory) / 1e6
                elif self.in_train and p.pid != self.pid:
                    pid = int(p.pid)
                    self.concurrent_compute_pids_seen.add(pid)
                    used_mb = round(float(p.usedGpuMemory) / 1e6, 1) if p.usedGpuMemory else None
                    if self.detailed_sharing and pid not in self.concurrent_compute_processes_seen:
                        try:
                            owner = pwd.getpwuid(os.stat(f"/proc/{pid}").st_uid).pw_name
                        except Exception:
                            owner = "unknown"
                        try:
                            with open(f"/proc/{pid}/comm", encoding="utf-8", errors="replace") as f:
                                process_name = f.read().strip()
                        except Exception:
                            process_name = "unknown"
                        event = {
                            "first_seen_unix": time.time(),
                            "pid": pid,
                            "owner": owner,
                            "other_account": owner not in {pwd.getpwuid(os.getuid()).pw_name, "unknown"},
                            "process_name": process_name,
                            "used_gpu_memory_MB_first_seen": used_mb,
                        }
                        self.concurrent_compute_processes_seen[pid] = event
                        try:
                            with open(self.gpu_sharing_events_path, "a", encoding="utf-8") as f:
                                f.write(json.dumps(event, sort_keys=True) + "\n")
                        except Exception:
                            pass
        except Exception:
            pass
        return mb

    def _measure_idle(self, seconds):
        ps = []
        t = time.time()
        while time.time() - t < seconds:
            p = self._read_power_w()
            if p is not None:
                ps.append(p)
            time.sleep(0.2)
        return (sum(ps) / len(ps)) if ps else None

    def _sample_loop(self):
        while not self._stop:

            with self._state_lock:
                active = self.in_train
                p = self._read_power_w()
                mem = self._read_proc_mem_mb()
                self._samples.append((time.time(), p, mem, active))
                if active and mem > self.peak_proc_mem_mb:
                    self.peak_proc_mem_mb = mem
                if active and p is not None and p > self.peak_train_power_w:
                    self.peak_train_power_w = float(p)
            time.sleep(self._sample_dt)

    def mark_train_start(self, task=None):
        try:
            import torch
            if torch.cuda.is_available():

                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        with self._state_lock:
            self._e0 = self._read_energy_mj()
            self._t0 = time.time()
            self.in_train = True

    def mark_train_end(self, task=None):
        if self._t0 is None:
            return

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        with self._state_lock:
            e1 = self._read_energy_mj()
            t1 = time.time()

            p_end = self._read_power_w()
            if p_end is not None and p_end > self.peak_train_power_w:
                self.peak_train_power_w = float(p_end)
            self.in_train = False
        dt = t1 - self._t0
        de = None
        if self._energy_supported and self._e0 is not None and e1 is not None:
            de = e1 - self._e0
            self.train_energy_mj += de
        self.train_time_s += dt
        torch_mb = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                torch_mb = torch.cuda.max_memory_allocated() / 1e6
                self.peak_torch_alloc_mb = max(self.peak_torch_alloc_mb, torch_mb)
        except Exception:
            pass
        self.per_task.append({"task": task, "dt_s": dt, "energy_mj": de, "torch_alloc_mb": torch_mb})
        self._e0 = None
        self._t0 = None

    def _integrate_train_power(self):

        energy_j = 0.0
        prev = None
        for (t, p, mem, intr) in self._samples:
            if prev is not None:
                pt, pp, _, pintr = prev
                if pintr and intr and p is not None and pp is not None:
                    energy_j += 0.5 * (pp + p) * (t - pt)
            prev = (t, p, mem, intr)
        return energy_j

    def finalize(self, model_params=None, model_mb=None, extra=None):
        self._stop = True
        time.sleep(self._sample_dt * 3)
        if self._energy_supported and self.train_energy_mj > 0:
            energy_j = self.train_energy_mj / 1000.0
            energy_src = "nvml_total_energy_counter"
        else:
            energy_j = self._integrate_train_power()
            energy_src = "power_sampling_trapezoidal_fallback"
        avg_power_w = (energy_j / self.train_time_s) if self.train_time_s > 0 else None
        idle_energy_j = (self.idle_power_w * self.train_time_s) if self.idle_power_w else None
        marginal_energy_j = (energy_j - idle_energy_j) if idle_energy_j is not None else None
        clean_avg_power_w = (avg_power_w - self.idle_power_w) if avg_power_w is not None and self.idle_power_w else None
        clean_peak_power_w = (self.peak_train_power_w - self.idle_power_w) if self.idle_power_w else None
        out = {
            "gpu_index": self.gpu_index,
            "gpu_name": self.gpu_name,
            "pid": self.pid,
            "energy_source": energy_src,
            "energy_counter_supported": self._energy_supported,
            "TRAIN_ONLY_excludes_test_eval": True,
            "train_energy_J": round(energy_j, 2),
            "train_energy_MJ": round(energy_j / 1e6, 8),
            "train_time_s": round(self.train_time_s, 2),
            "avg_power_W": round(avg_power_w, 2) if avg_power_w is not None else None,
            "peak_train_power_W": round(self.peak_train_power_w, 2),
            "avg_power_definition": "TRAIN-window board energy divided by synchronized TRAIN-window wall time",
            "peak_train_power_definition": "maximum sampled NVML whole-board power during TRAIN windows",
            "peak_train_power_is_sampled": True,
            "power_sample_hz": self.sample_hz,
            "cuda_synchronized_at_train_boundaries": True,
            "idle_power_W": round(self.idle_power_w, 2) if self.idle_power_w else None,
            "idle_baseline_seconds": self.idle_seconds,
            "idle_energy_J_over_train_time": round(idle_energy_j, 2) if idle_energy_j is not None else None,
            "marginal_energy_J_above_idle": round(marginal_energy_j, 2) if marginal_energy_j is not None else None,
            "peak_train_proc_mem_MB": round(self.peak_proc_mem_mb, 1),
            "peak_train_torch_alloc_MB": round(self.peak_torch_alloc_mb, 1),
            "energy_exclusivity_required": self.energy_exclusivity_required,
            "energy_scope": "whole_gpu_board_during_train_windows",
            "energy_may_include_concurrent_workloads": len(self.concurrent_compute_pids_seen) > 0,
            "energy_isolation_clean": len(self.concurrent_compute_pids_seen) == 0,
            "concurrent_compute_pids_seen": sorted(self.concurrent_compute_pids_seen),
            "measurement_boundary": {
                "includes": [
                    "backbone optimization",
                    "training-set feature extraction required by independent head",
                    "independent-head parameter optimization",
                    "method consolidation"
                ],
                "excludes": [
                    "validation forward",
                    "test forward",
                    "accuracy/BWT evaluation"
                ]
            },
            "final_model_params": model_params,
            "final_model_MB": round(model_mb, 3) if model_mb is not None else None,
            "per_task_train": self.per_task,
        }
        if self.emit_clean_fields:
            out.update({
                "clean_train_energy_MJ": round(marginal_energy_j / 1e6, 8) if marginal_energy_j is not None else None,
                "clean_avg_train_power_W": round(clean_avg_power_w, 2) if clean_avg_power_w is not None else None,
                "clean_peak_train_power_W": round(clean_peak_power_w, 2) if clean_peak_power_w is not None else None,
            })
        if self.detailed_sharing:
            out["concurrent_compute_processes_seen"] = [
                self.concurrent_compute_processes_seen[pid]
                for pid in sorted(self.concurrent_compute_processes_seen)
            ]
        if extra:
            out.update(extra)
        try:
            json.dump(out, open(self.out_path, "w"), indent=2)
        except Exception as e:
            print("ResourceMeter dump failed:", e)
        try:
            self.nvml.nvmlShutdown()
        except Exception:
            pass
        return out
