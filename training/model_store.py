#!/usr/bin/env python3
"""
training/model_store.py — Permanent model persistence for PotholeEye

Storage backend: HuggingFace Hub (private model repository)
  - Free tier: private repos with Git LFS — YOLOv8m ≈ 52 MB → 1 000+ versions for free
  - Every push is a git commit: fully auditable history, one-line rollback
  - Pull from any machine with just two env vars: HF_TOKEN + HF_MODEL_REPO
  - SHA-256 checksum verification on every download — never silently corrupt

Repository layout on HuggingFace Hub:
  {HF_MODEL_REPO}/
  ├── weights/
  │   ├── pothole_yolov8.pt           ← latest production model (overwritten each push)
  │   └── last.pt                     ← resume checkpoint (overwritten each push)
  ├── versions/
  │   └── pothole_yolov8_20240504_001700.pt  ← immutable timestamped archive
  ├── metrics/
  │   └── 20240504_001700.json        ← training results snapshot for that version
  └── model_manifest.json             ← metadata for the current latest model

Required environment variables (set in .env.gpu):
  HF_TOKEN       — HuggingFace user access token (starts with hf_...)
                   Get one at: https://huggingface.co/settings/tokens
  HF_MODEL_REPO  — Private repo ID, e.g. "yourname/potholeye-weights"
                   Create at: https://huggingface.co/new

CLI usage:
  #After training completes (called automatically by train_gpu.py):
  python training/model_store.py push models/pothole_yolov8.pt \\
      --last models/last.pt --run-dir runs/train/pothole_v1

  #On a new RunPod pod — restore the latest trained model:
  python training/model_store.py pull --dest models/

  #Check what's stored:
  python training/model_store.py status

  #Verify local file matches Hub:
  python training/model_store.py verify models/pothole_yolov8.pt
"""

import argparse
import csv
import hashlib
import io
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

#Repo layout constants────
MANIFEST_FILE  = "model_manifest.json"
WEIGHTS_DIR    = "weights"    #path inside HF repo for latest weights
VERSIONS_DIR   = "versions"   #path inside HF repo for versioned archives
METRICS_DIR    = "metrics"    #path inside HF repo for per-run metrics
LATEST_NAME    = "pothole_yolov8.pt"
RESUME_NAME    = "last.pt"
HF_CACHE_DIR   = ".hf_cache"  #local sub-dir inside dest_dir for HF cache files



#ModelStore


class ModelStore:
    """
    Manages permanent model weight storage backed by HuggingFace Hub.

    Instantiation reads HF_MODEL_REPO and HF_TOKEN from the environment
    (already loaded by start_app.sh from .env.gpu).  Both must be set.
    """

    def __init__(
        self,
        repo_id: Optional[str] = None,
        token:   Optional[str] = None,
    ):
        self.repo_id = (repo_id or os.environ.get("HF_MODEL_REPO", "")).strip()
        self.token   = (token   or os.environ.get("HF_TOKEN",       "")).strip()

        if not self.repo_id:
            raise EnvironmentError(
                "HF_MODEL_REPO is not set.\n"
                "  1. Create a private HuggingFace repo at https://huggingface.co/new\n"
                "  2. Add to .env.gpu:  HF_MODEL_REPO=yourname/potholeye-weights\n"
                "  3. Add to .env.gpu:  HF_TOKEN=hf_xxxxxxxxxxxx"
            )
        if not self.token:
            raise EnvironmentError(
                "HF_TOKEN is not set.\n"
                "  1. Get a Write token at https://huggingface.co/settings/tokens\n"
                "  2. Add to .env.gpu:  HF_TOKEN=hf_xxxxxxxxxxxx"
            )

        self._api = None  #lazy-initialised in self.api property

    #Lazy API─────────────

    @property
    def api(self):
        if self._api is None:
            try:
                from huggingface_hub import HfApi
            except ImportError:
                raise RuntimeError(
                    "huggingface_hub is not installed.\n"
                    "  pip install 'huggingface_hub>=0.20.0'"
                )
            self._api = HfApi(token=self.token)
        return self._api

   
    #PUSH — called automatically by train_gpu.py after every training run
   

    def push(
        self,
        model_path: Path,
        last_pt:    Optional[Path] = None,
        metrics:    Optional[dict] = None,
        run_dir:    Optional[Path] = None,
    ) -> str:
        """
        Upload trained model weights to HuggingFace Hub.

        What gets uploaded:
          weights/pothole_yolov8.pt          ← latest (overwrites previous)
          weights/last.pt                    ← resume checkpoint (if provided)
          versions/pothole_yolov8_<ts>.pt    ← immutable versioned archive
          metrics/<ts>.json                  ← training metrics snapshot
          model_manifest.json               ← metadata about the latest model

        Returns the HuggingFace commit URL (empty string on partial success).
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        ts       = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        sha256   = _sha256(model_path)
        size_mb  = model_path.stat().st_size / 1e6

        _ensure_repo_exists(self.api, self.repo_id)

        _banner("PUSH")
        print(f"  Repo    : {self.repo_id}")
        print(f"  Model   : {model_path.name}  ({size_mb:.1f} MB)")
        print(f"  SHA256  : {sha256[:24]}…")
        print(f"  Version : {ts}")
        print(_SEP)

        #Build upload list
        uploads: list[tuple] = []

        #1. Latest weights (overwrites previous)
        uploads.append((
            str(model_path),
            f"{WEIGHTS_DIR}/{LATEST_NAME}",
            f"train {ts}: update latest weights",
        ))

        #2. Versioned immutable archive
        archive_name = f"pothole_yolov8_{ts}.pt"
        uploads.append((
            str(model_path),
            f"{VERSIONS_DIR}/{archive_name}",
            f"train {ts}: archive {archive_name}",
        ))

        #3. Resume checkpoint
        if last_pt and Path(last_pt).exists():
            lp = Path(last_pt)
            uploads.append((
                str(lp),
                f"{WEIGHTS_DIR}/{RESUME_NAME}",
                f"train {ts}: update last.pt ({lp.stat().st_size/1e6:.1f} MB)",
            ))

        #4. Training metrics JSON
        metrics_data = _collect_metrics(metrics, run_dir, ts)
        uploads.append((
            json.dumps(metrics_data, indent=2).encode(),
            f"{METRICS_DIR}/{ts}.json",
            f"train {ts}: metrics snapshot",
        ))

        #5. Manifest
        manifest = {
            "version":        ts,
            "filename":       LATEST_NAME,
            "sha256":         sha256,
            "size_bytes":     model_path.stat().st_size,
            "pushed_at":      datetime.now(timezone.utc).isoformat(),
            "repo_id":        self.repo_id,
            "archive_path":   f"{VERSIONS_DIR}/{archive_name}",
            "metrics":        metrics_data.get("metrics", {}),
            "training":       metrics_data.get("training", {}),
        }
        uploads.append((
            json.dumps(manifest, indent=2).encode(),
            MANIFEST_FILE,
            f"train {ts}: update manifest",
        ))

        #Execute uploads──
        commit_url = ""
        for idx, (src, dst, msg) in enumerate(uploads, 1):
            label = dst.split("/")[-1]
            print(f"  [{idx}/{len(uploads)}] {label:<35}", end=" ", flush=True)
            t0 = time.time()
            try:
                if isinstance(src, bytes):
                    commit_info = self.api.upload_file(
                        path_or_fileobj=io.BytesIO(src),
                        path_in_repo=dst,
                        repo_id=self.repo_id,
                        repo_type="model",
                        commit_message=msg,
                    )
                else:
                    commit_info = self.api.upload_file(
                        path_or_fileobj=src,
                        path_in_repo=dst,
                        repo_id=self.repo_id,
                        repo_type="model",
                        commit_message=msg,
                    )
                elapsed = time.time() - t0
                if hasattr(commit_info, "commit_url"):
                    commit_url = commit_info.commit_url
                print(f"✓  ({elapsed:.1f}s)")
            except Exception as exc:
                print(f"✗  FAILED")
                raise RuntimeError(f"Upload failed for {dst}: {exc}") from exc

        print()
        print(f"  ✓ Model permanently saved!")
        print(f"  ▸ https://huggingface.co/{self.repo_id}")
        if commit_url:
            print(f"  ▸ {commit_url}")
        print()
        print("  To restore on any new pod:")
        print("    python training/model_store.py pull --dest models/")
        print(_SEP)
        return commit_url

   
    #PULL — called by start_app.sh on every new pod if model is missing
   

    def pull(
        self,
        dest_dir:  Path = Path("models"),
        filename:  str  = LATEST_NAME,
        also_last: bool = False,
        force:     bool = False,
    ) -> Optional[Path]:
        """
        Download the latest model weights from HuggingFace Hub.

        Skips download when the local file already matches the Hub checksum.
        Verifies SHA-256 after every download — deletes and fails if corrupt.

        Returns the local model Path on success, None on failure.
        """
        dest_dir  = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / filename
        cache_dir = dest_dir / HF_CACHE_DIR

        print(f"\n  Querying model store: {self.repo_id} …")

        manifest = self._pull_manifest()
        if manifest is None:
            print("  ✗ Cannot read manifest — repo may be empty or token invalid.")
            print("    Check HF_TOKEN and HF_MODEL_REPO in .env.gpu")
            return None

        remote_sha  = manifest.get("sha256", "")
        remote_size = manifest.get("size_bytes", 0)
        remote_ver  = manifest.get("version", "unknown")
        m           = manifest.get("metrics", {})

        print(f"  Latest version : {remote_ver}")
        print(f"  Size           : {remote_size / 1e6:.1f} MB")
        if m:
            map50 = m.get("mAP50", m.get("map50", "—"))
            prec  = m.get("precision", "—")
            rec   = m.get("recall", "—")
            print(f"  mAP50          : {map50}  |  Precision: {prec}  |  Recall: {rec}")

        #Skip if already up-to-date
        if dest_path.exists() and remote_sha and not force:
            local_sha = _sha256(dest_path)
            if local_sha == remote_sha:
                print(f"  ✓ Already up-to-date: {dest_path} (checksum matches)")
                return dest_path
            print(f"  ⚑ Local checksum differs — downloading fresh copy …")

        #Download
        result = self._download_file(
            repo_filename=f"{WEIGHTS_DIR}/{filename}",
            dest_path=dest_path,
            cache_dir=cache_dir,
            label=filename,
        )
        if result is None:
            return None

        #Verify checksum
        if remote_sha:
            print(f"  Verifying SHA-256 …", end=" ", flush=True)
            actual = _sha256(dest_path)
            if actual != remote_sha:
                print(f"✗  CORRUPTED\n  expected {remote_sha[:24]}…\n  got      {actual[:24]}…")
                dest_path.unlink(missing_ok=True)
                return None
            print("✓")

        #Optionally pull last.pt for training resume
        if also_last:
            self._download_file(
                repo_filename=f"{WEIGHTS_DIR}/{RESUME_NAME}",
                dest_path=dest_dir / RESUME_NAME,
                cache_dir=cache_dir,
                label=RESUME_NAME,
            )

        #Clean up HF cache (these are large .pt files — no need to double-store)
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)

        print(f"\n  ✓ Model restored → {dest_path}")
        return dest_path

   
    #STATUS
   

    def status(self) -> None:
        """Print a summary of what is stored in the Hub."""
        manifest = self._pull_manifest()
        if manifest is None:
            print(f"\n  No model stored in {self.repo_id} yet.")
            print("  Train first:  python training/train_gpu.py")
            return

        _banner("STATUS")
        print(f"  Repo      : https://huggingface.co/{self.repo_id}")
        print(f"  Version   : {manifest.get('version', '?')}")
        print(f"  Pushed at : {manifest.get('pushed_at', '?')}")
        print(f"  File      : {manifest.get('filename', '?')}")
        print(f"  Size      : {manifest.get('size_bytes', 0) / 1e6:.1f} MB")
        print(f"  SHA-256   : {manifest.get('sha256', '?')[:32]}…")

        m = manifest.get("metrics", {})
        if m:
            print(f"\n  Training Metrics")
            for k, v in m.items():
                print(f"    {k:<24}: {v}")

        t = manifest.get("training", {})
        if t:
            print(f"\n  Training Config─")
            for k, v in t.items():
                print(f"    {k:<24}: {v}")

        #Count archived versions
        try:
            tree = list(self.api.list_repo_tree(
                repo_id=self.repo_id, repo_type="model", path_in_repo=VERSIONS_DIR,
            ))
            n = sum(1 for f in tree if hasattr(f, "path") and f.path.endswith(".pt"))
            if n:
                print(f"\n  Archived Versions: {n}")
        except Exception:
            pass

        print(_SEP)

   
    #VERIFY
   

    def verify(self, local_path: Path) -> bool:
        """Return True if local_path checksum matches the Hub manifest."""
        local_path = Path(local_path)
        if not local_path.exists():
            print(f"  ✗ Local file not found: {local_path}")
            return False

        manifest = self._pull_manifest()
        if not manifest:
            print("  ✗ Cannot fetch manifest from Hub")
            return False

        remote_sha = manifest.get("sha256", "")
        if not remote_sha:
            print("  ✗ Manifest has no checksum")
            return False

        local_sha = _sha256(local_path)
        if local_sha == remote_sha:
            print(f"  ✓ {local_path.name} matches Hub ({remote_sha[:24]}…)")
            return True
        else:
            print(f"  ✗ Checksum mismatch!")
            print(f"    local : {local_sha[:32]}…")
            print(f"    hub   : {remote_sha[:32]}…")
            return False

    #────────
    #Private helpers
    #────────

    def _pull_manifest(self) -> Optional[dict]:
        """Download and parse model_manifest.json from the Hub."""
        try:
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(
                repo_id   = self.repo_id,
                filename  = MANIFEST_FILE,
                repo_type = "model",
                token     = self.token,
            )
            with open(p) as f:
                return json.load(f)
        except Exception:
            return None

    def _download_file(
        self,
        repo_filename: str,
        dest_path:     Path,
        cache_dir:     Path,
        label:         str,
    ) -> Optional[Path]:
        """Download one file from HF Hub to dest_path. Returns dest_path or None."""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise RuntimeError("pip install 'huggingface_hub>=0.20.0'")

        size_str = ""
        print(f"  Downloading {label:<30} {size_str}", end=" ", flush=True)
        t0 = time.time()
        try:
            local = hf_hub_download(
                repo_id   = self.repo_id,
                filename  = repo_filename,
                repo_type = "model",
                token     = self.token,
                cache_dir = str(cache_dir),
            )
            shutil.copy2(local, dest_path)
            elapsed = time.time() - t0
            mb = dest_path.stat().st_size / 1e6
            print(f"✓  ({mb:.1f} MB in {elapsed:.1f}s)")
            return dest_path
        except Exception as exc:
            print(f"✗  FAILED: {exc}")
            return None



#Module-level convenience wrappers (used by train_gpu.py)


def push_model(
    model_path: Path,
    last_pt:    Optional[Path] = None,
    metrics:    Optional[dict] = None,
    run_dir:    Optional[Path] = None,
) -> str:
    """Push model to Hub. Returns commit URL or '' on missing config."""
    try:
        return ModelStore().push(model_path, last_pt=last_pt, metrics=metrics, run_dir=run_dir)
    except EnvironmentError as e:
        print(f"\n  [model_store] Skipping push — {e}\n")
        return ""


def pull_model(
    dest_dir:  Path = Path("models"),
    also_last: bool = False,
    force:     bool = False,
) -> Optional[Path]:
    """Pull latest model from Hub. Returns local Path or None."""
    try:
        return ModelStore().pull(Path(dest_dir), also_last=also_last, force=force)
    except EnvironmentError as e:
        print(f"\n  [model_store] Cannot pull — {e}\n")
        return None



#Private utilities


_SEP = "─" * 60


def _banner(action: str) -> None:
    print(f"\n{'═'*60}")
    print(f"  PotholeEye Model Store — {action}")
    print(f"{'═'*60}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _ensure_repo_exists(api, repo_id: str) -> None:
    """Create private HF repo if it doesn't exist yet."""
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
    except Exception:
        print(f"  Creating private HF repo: {repo_id} …", end=" ", flush=True)
        try:
            api.create_repo(repo_id=repo_id, repo_type="model", private=True)
            print("✓")
        except Exception as exc:
            print(f"✗")
            raise RuntimeError(f"Failed to create repo {repo_id}: {exc}") from exc


def _collect_metrics(
    metrics: Optional[dict],
    run_dir: Optional[Path],
    ts:      str,
) -> dict:
    """
    Gather training metrics.  Priority:
      1. Caller-supplied metrics dict
      2. results.csv from the run directory (YOLO standard output)
      3. args.yaml from the run directory (training config)
    """
    data: dict = {"version": ts, "metrics": {}, "training": {}}

    if metrics:
        data["metrics"].update(metrics)

    if run_dir is None:
        return data

    run_dir = Path(run_dir)

    #Parse results.csv — last row = final epoch stats
    csv_path = run_dir / "results.csv"
    if csv_path.exists():
        try:
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            if rows:
                last = {k.strip(): v.strip() for k, v in rows[-1].items()}
                #YOLO column names vary by version — cover all known formats
                key_map = {
                    "metrics/mAP50(B)":    "mAP50",
                    "metrics/mAP50-95(B)": "mAP50-95",
                    "metrics/precision(B)":"precision",
                    "metrics/recall(B)":   "recall",
                    "train/box_loss":      "train_box_loss",
                    "val/box_loss":        "val_box_loss",
                    "val/cls_loss":        "val_cls_loss",
                    "val/dfl_loss":        "val_dfl_loss",
                }
                for src, dst in key_map.items():
                    if src in last:
                        try:
                            data["metrics"][dst] = round(float(last[src]), 4)
                        except (ValueError, TypeError):
                            pass
        except Exception:
            pass

    #Parse args.yaml — training hyper-parameters
    args_yaml = run_dir / "args.yaml"
    if args_yaml.exists():
        try:
            import yaml
            with open(args_yaml) as f:
                args = yaml.safe_load(f) or {}
            for k in ("model", "epochs", "imgsz", "batch", "device", "patience"):
                if k in args:
                    data["training"][k] = args[k]
        except Exception:
            pass

    return data



#CLI


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PotholeEye Model Store — push/pull weights to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    #push
    pp = sub.add_parser("push", help="Upload trained model weights to Hub")
    pp.add_argument("model", help="Path to pothole_yolov8.pt")
    pp.add_argument("--last",    default=None, help="Path to last.pt (training resume checkpoint)")
    pp.add_argument("--run-dir", default=None, help="Training run directory (used to read metrics)")

    #pull
    pl = sub.add_parser("pull", help="Download latest model from Hub to this machine")
    pl.add_argument("--dest",  default="models/", help="Local destination directory")
    pl.add_argument("--last",  action="store_true", help="Also download last.pt resume checkpoint")
    pl.add_argument("--force", action="store_true", help="Re-download even if checksum matches")

    #status
    sub.add_parser("status", help="Show what is stored in the Hub")

    #verify
    pv = sub.add_parser("verify", help="Verify local model checksum against Hub manifest")
    pv.add_argument("model", help="Path to local model file")

    return p


def _cli() -> None:
    args = _build_parser().parse_args()

    try:
        store = ModelStore()
    except EnvironmentError as e:
        print(f"\n✗  Configuration error:\n  {e}\n")
        sys.exit(1)

    if args.cmd == "push":
        last_pt = Path(args.last)    if args.last    else None
        run_dir = Path(args.run_dir) if args.run_dir else None
        store.push(Path(args.model), last_pt=last_pt, run_dir=run_dir)

    elif args.cmd == "pull":
        result = store.pull(
            dest_dir  = Path(args.dest),
            also_last = args.last,
            force     = args.force,
        )
        sys.exit(0 if result else 1)

    elif args.cmd == "status":
        store.status()

    elif args.cmd == "verify":
        ok = store.verify(Path(args.model))
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    _cli()
