"""Check GPU availability, runtime metadata, and kernel package provenance."""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

from max import driver


def _safe_getattr(obj, name: str) -> str:
    try:
        return str(getattr(obj, name))
    except Exception as exc:
        return f"<unavailable: {exc}>"


def _print_kernel_pkg_info() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    kernels_dir = repo_root / "mxframe" / "kernels_v261"
    pkg_path = repo_root / "mxframe" / "kernels.mojopkg"
    print(f"kernels_dir_path: {kernels_dir}")
    print(f"kernels_dir_exists: {str(kernels_dir.exists()).lower()}")
    print(f"kernels_pkg_path: {pkg_path}")
    if not pkg_path.exists():
        print("kernels_pkg_exists: false")
        return
    stat = pkg_path.stat()
    print("kernels_pkg_exists: true")
    print(f"kernels_pkg_size_bytes: {stat.st_size}")
    print(f"kernels_pkg_mtime_epoch: {int(stat.st_mtime)}")


def main() -> None:
    print(f"python: {sys.version.split()[0]}")
    print(f"platform: {platform.platform()}")
    print(f"MODULAR_DEVICE_CONTEXT_SYNC_MODE: {os.getenv('MODULAR_DEVICE_CONTEXT_SYNC_MODE', '<unset>')}")

    count = driver.accelerator_count()
    print(f"accelerator_count: {count}")
    if count <= 0:
        print("gpu_available: false")
        _print_kernel_pkg_info()
        return

    print("gpu_available: true")
    acc = driver.Accelerator()
    print(f"gpu_device: {acc}")
    print(f"gpu_architecture_name: {_safe_getattr(acc, 'architecture_name')}")
    print(f"gpu_id: {_safe_getattr(acc, 'id')}")
    print(f"gpu_driver_name: {_safe_getattr(acc, 'driver_name')}")
    _print_kernel_pkg_info()


if __name__ == "__main__":
    main()