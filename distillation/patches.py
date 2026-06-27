"""
Compatibility patches for LCM distillation.

- install_flash_attn_stub(): Must be called BEFORE importing wan_va modules.
  Installs a fake flash_attn in sys.modules so model.py import doesn't crash.

- SafeMultiLatentLeRobotDataset: Drop-in replacement for MultiLatentLeRobotDataset
  that skips incomplete sub-datasets instead of crashing.
"""

import importlib.machinery
import sys
import types


def install_flash_attn_stub():
    """Install FlashAttention compatibility aliases.

    Prefer the real flash-attn package when it is installed.  Some LingBot/
    Wan code first imports ``flash_attn_interface`` and only falls back to
    ``flash_attn`` on ImportError.  Creating a dummy ``flash_attn_interface``
    with ``flash_attn_func=None`` silently disables FlashAttention, so when
    real ``flash_attn`` exists we expose it under the interface name instead.
    Only install a stub when neither implementation is available.
    """
    try:
        import flash_attn as real_flash_attn
        if "flash_attn_interface" not in sys.modules:
            alias = types.ModuleType("flash_attn_interface")
            alias.__spec__ = importlib.machinery.ModuleSpec("flash_attn_interface", None)
            alias.__version__ = getattr(real_flash_attn, "__version__", "real-flash-attn")
            alias.flash_attn_func = real_flash_attn.flash_attn_func
            alias.flash_attn_varlen_func = getattr(real_flash_attn, "flash_attn_varlen_func", None)
            sys.modules["flash_attn_interface"] = alias
            print("INFO: flash_attn_interface aliased to real flash_attn")
        return
    except Exception as e:
        print(f"WARNING: real flash_attn unavailable ({e}); installing SDPA stubs")

    for mod_name in ("flash_attn_interface", "flash_attn"):
        if mod_name in sys.modules:
            continue
        stub = types.ModuleType(mod_name)
        stub.__spec__ = importlib.machinery.ModuleSpec(mod_name, None)
        stub.__version__ = "0.0.0"
        stub.flash_attn_func = None
        stub.flash_attn_varlen_func = None
        sys.modules[mod_name] = stub
        print(f"WARNING: {mod_name} not available, installed stub (torch SDPA will be used)")


class SafeMultiLatentLeRobotDataset:
    """
    Same as MultiLatentLeRobotDataset but skips sub-datasets that fail to load
    (e.g. incomplete downloads, missing parquet files).
    """

    def __init__(self, config, num_init_worker=128):
        import os
        from pathlib import Path
        from dataset.lerobot_latent_dataset import (
            recursive_find_file,
            LatentLeRobotDataset,
        )

        repo_list = recursive_find_file(config.dataset_path, "info.json")
        repo_list = [v.split("/meta/info.json")[0] for v in repo_list]

        self._datasets = []
        for repo_id in repo_list:
            try:
                ds = LatentLeRobotDataset(repo_id=repo_id, config=config)
                self._datasets.append(ds)
            except Exception as e:
                print(f"WARNING: Skipping incomplete dataset {os.path.basename(repo_id)}: {e}")

        total = len(repo_list)
        loaded = len(self._datasets)
        print(f"Loaded {loaded}/{total} sub-datasets successfully")
        if loaded == 0:
            raise RuntimeError(
                "No valid sub-datasets found. Dataset download may be incomplete."
            )

        self.item_id_to_dataset_id, self.acc_dset_num = self._get_item_id_to_dataset_id()

    def __len__(self):
        return sum(len(v) for v in self._datasets)

    def _get_item_id_to_dataset_id(self):
        item_id_to_dataset_id = {}
        acc_dset_num = {}
        acc_nums = [0]
        id = 0
        for dset_id, dset in enumerate(self._datasets):
            acc_nums.append(acc_nums[-1] + len(dset))
            for _ in range(len(dset)):
                item_id_to_dataset_id[id] = dset_id
                id += 1
        for did in range(len(self._datasets)):
            acc_dset_num[did] = acc_nums[did]
        return item_id_to_dataset_id, acc_dset_num

    def __getitem__(self, idx):
        assert idx < len(self)
        cur_dset = self._datasets[self.item_id_to_dataset_id[idx]]
        local_idx = idx - self.acc_dset_num[self.item_id_to_dataset_id[idx]]
        return cur_dset[local_idx]
