"""End-to-end orchestrator for the AutoPharmaX pipeline.

Runs the five stages sequentially and prints a summary. Unlike `dvc repro`,
this orchestrator does not check input hashes - it just runs whatever you
ask, which is useful for fresh-box smoke tests and debug runs.

Use --from-stage to resume partway through:
  python src/pipeline/run_pipeline.py --from-stage train
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

log = logging.getLogger(__name__)

STAGES = [
    ("ingest",   "src/pipeline/stage1_ingest.py"),
    ("features", "src/pipeline/stage2_features.py"),
    ("train",    "src/pipeline/stage3_train.py"),
    ("evaluate", "src/pipeline/stage4_evaluate.py"),
    ("register", "src/pipeline/stage5_register.py"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-stage", choices=[s for s, _ in STAGES], default="ingest")
    ap.add_argument("--to-stage",   choices=[s for s, _ in STAGES], default="register")
    ap.add_argument("--debug",      action="store_true", help="pass --debug to each stage")
    ap.add_argument("--python",     default=str(ROOT / ".venv" / "Scripts" / "python.exe"),
                    help="python interpreter to use")
    ap.add_argument("--extra-ingest", nargs=argparse.REMAINDER, default=[],
                    help="extra flags for stage 1 (e.g. --skip-depmap --skip-drugcomb)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    names   = [s for s, _ in STAGES]
    start   = names.index(args.from_stage)
    end     = names.index(args.to_stage) + 1
    subset  = STAGES[start:end]

    log.info("running stages: %s", [s for s, _ in subset])
    summaries: list[tuple[str, int, float]] = []

    for name, script in subset:
        cmd = [args.python, script]
        if args.debug:
            cmd.append("--debug")
        if name == "ingest" and args.extra_ingest:
            cmd += args.extra_ingest
        log.info("========== stage: %s ==========", name)
        log.info("$ %s", " ".join(cmd))
        import time
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        dt = time.time() - t0
        summaries.append((name, rc, dt))
        if rc != 0:
            log.error("stage %s failed (exit %d); halting pipeline.", name, rc)
            break

    print()
    print("Pipeline summary")
    print("-" * 44)
    for name, rc, dt in summaries:
        status = "OK " if rc == 0 else "FAIL"
        print(f"  {status}  {name:<10s}  {dt:7.1f}s  (exit {rc})")
    print("-" * 44)

    metrics_path = ROOT / "metrics.json"
    if metrics_path.exists():
        try:
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
            ldo = m.get("ldo", {}).get("r2", None)
            std = m.get("standard", {}).get("r2", None)
            print(f"  headline LDO R^2 = {ldo}")
            print(f"  standard  R^2    = {std}")
        except Exception:
            pass

    return 0 if all(rc == 0 for _, rc, _ in summaries) else 1


if __name__ == "__main__":
    sys.exit(main())
