# GPU detection: adapters, VRAM, and driver version

GPU discovery stays conservative by design (see `docs/architecture.md`): presence
always comes from a primary, OS-native source, and no detection result implies a
compute backend (CUDA, ROCm, DirectML, Metal) is installed.

## Primary sources

| Platform | Adapter presence |
|---|---|
| Windows | `Get-CimInstance Win32_VideoController`, falling back to the display-driver registry if CIM is restricted |
| Linux | `lspci` |
| macOS | `system_profiler SPDisplaysDataType -json` |

## VRAM and driver-version enrichment

`dedicated_memory_bytes` and `driver_version` are filled in, best-effort, from
additional sources layered on top of the primary source above. Every enrichment
source is optional: if it's missing, times out, or fails, the field it would have
filled stays `null` — it is never treated as a collection failure and never blocks
the rest of the profile.

`GPUProfile.source` records exactly which sources contributed, joined with `+`
(e.g. `linux_lspci+sysfs+nvidia_smi`), so a caller can tell a measured VRAM figure
from a missing one without guessing.

### Windows

`Get-CimInstance` already reports `AdapterRAM` and `DriverVersion` directly; no
additional enrichment is needed.

### Linux

Two independent, best-effort sources, both correlated to the `lspci` adapter by
**PCI address** (`BB:DD.F`), not by name — vendor ID strings from different tools
rarely match exactly:

1. **sysfs** (`_linux_sysfs_vram_by_pci_address`): reads
   `/sys/class/drm/cardN/device/mem_info_vram_total`, a file the amdgpu, nouveau,
   and nvidia kernel drivers expose directly in bytes. The matching PCI address
   comes from the adjacent `uevent` file's `PCI_SLOT_NAME` line. No subprocess, no
   elevated privileges.
2. **`nvidia-smi`** (`_nvidia_smi_info`): `--query-gpu=pci.bus_id,memory.total,driver_version`.
   The only source of `driver_version` on Linux — there is no general cross-vendor
   concept of "driver version" on Linux the way Windows exposes one (module version
   is tied to the kernel build, not a per-driver release number), so non-NVIDIA
   adapters intentionally report `driver_version: null`.

If both sources report VRAM for the same adapter, sysfs wins (it reads the number
the kernel driver itself computed, with no unit-conversion step); `nvidia-smi`'s
figure is only used when sysfs has nothing for that address.

### macOS

`system_profiler`'s JSON already carries a VRAM string on adapters that have one
(`sppci_vram` or `spdisplays_vram`, e.g. `"8 GB"`); the collector now parses it into
integer bytes. Two things stay `null` **by design**, not because parsing is
missing:

- **Apple Silicon VRAM.** Unified memory has no separate "VRAM" byte count for
  `system_profiler` to report — inventing one from total system RAM would be a
  guess dressed up as a measurement, which the project explicitly avoids.
- **Driver version, on every macOS adapter.** macOS does not expose a
  per-GPU driver version the way Windows does.

## Known remaining gaps

- No AMD-equivalent of `nvidia-smi` (`rocm-smi`) is wired up yet; AMD driver
  version on Linux stays `null` even when `rocm-smi` is installed.
- ARM Linux boards without a `Hardware:` line in `/proc/cpuinfo` can still end up
  with no CPU model at all (a CPU, not GPU, gap).
