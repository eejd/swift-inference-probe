#!/usr/bin/env python3
"""Which device does each op actually run on? — ANE residency from MLComputePlan.

Every other experiment here infers dispatch from behaviour. F-007 noticed that
`.all` was bit-identical to `.cpuAndNE` and concluded Core ML must be sending
this model to the Neural Engine, then flagged confirming it against the compute
plan as the M3 task. This is that confirmation, and it reads the assignment
directly rather than inferring it.

Residency is reported two ways, because they answer different questions:

  BY OP COUNT — what fraction of operations are assigned to each device.
                Treats a 1x1 convolution and a full-resolution one alike.
  BY COST     — the same fractions weighted by Core ML's own estimated cost.
                This is the one that predicts latency, and the one the §5.3
                threshold should be set from.

`MLComputePlan.load_from_path` needs a **compiled** `.mlmodelc`, not the
`.mlpackage` on disk, and passing the wrong one aborts the process with an
uncaught C++ exception rather than a Python error. So the model is compiled to
a temporary directory first.

Usage:
    python compute_plan.py                         # fp16 + fp32, all units
    python compute_plan.py --precisions fp16
    python compute_plan.py --units all cpuOnly
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan

# Repository-relative, not cwd-relative — see export_model.py and F-011.
FIXTURES = Path(__file__).resolve().parent.parent / "Fixtures" / "models"

UNITS = {
    "cpuOnly": ct.ComputeUnit.CPU_ONLY,
    "cpuAndGPU": ct.ComputeUnit.CPU_AND_GPU,
    "cpuAndNE": ct.ComputeUnit.CPU_AND_NE,
    "all": ct.ComputeUnit.ALL,
}

DEVICE_NAMES = {
    "MLCPUComputeDevice": "cpu",
    "MLGPUComputeDevice": "gpu",
    "MLNeuralEngineComputeDevice": "ane",
}


def device_name(device) -> str:
    return DEVICE_NAMES.get(type(device).__name__, type(device).__name__)


def walk_ops(block):
    """Yield every operation, descending into nested blocks.

    MobileNetV3 has no control flow, so this is currently a flat walk — but a
    model that does would silently under-report residency without the recursion.
    """
    for op in block.operations:
        yield op
        for inner in getattr(op, "blocks", None) or []:
            yield from walk_ops(inner)


def analyse(mlmodelc: str, unit: str, ane_permitted: bool) -> dict:
    plan = MLComputePlan.load_from_path(mlmodelc, compute_units=UNITS[unit])
    fn = plan.model_structure.program.functions["main"]

    by_device: dict[str, dict] = {}
    total_cost = 0.0
    planned = 0
    unplanned = 0
    ane_capable = 0
    not_ane_capable: list[str] = []

    for op in walk_ops(fn.block):
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if usage is None:
            # const and other ops Core ML does not schedule as work.
            unplanned += 1
            continue
        planned += 1

        cost = plan.get_estimated_cost_for_mlprogram_operation(op)
        weight = float(cost.weight) if cost is not None else 0.0
        total_cost += weight

        name = device_name(usage.preferred_compute_device)
        entry = by_device.setdefault(name, {"op_count": 0, "cost": 0.0, "op_types": {}})
        entry["op_count"] += 1
        entry["cost"] += weight
        entry["op_types"][op.operator_name] = entry["op_types"].get(op.operator_name, 0) + 1

        # `supported_compute_devices` reports what is available *under the
        # requested configuration*, not a static property of the op. Under
        # .cpuOnly no op lists the ANE, which is trivially true and says nothing
        # about the op. Only a configuration that actually permits the ANE can
        # answer "could this op run there?", so the metric is gated on that.
        supported = {device_name(d) for d in usage.supported_compute_devices}
        if "ane" in supported:
            ane_capable += 1
        else:
            not_ane_capable.append(f"{op.operator_name}")

    # Core ML's per-op cost weights are not bit-stable across plan loads: the
    # device assignment is identical run to run, but the weights wobble in their
    # last ulp, so an unrounded sum lands on 0.9999999999999986 one run and
    # 0.9999999999999981 the next. Reporting 16 significant digits of an
    # *estimate* is false precision anyway, and rounding is what makes this
    # file reproducible enough to commit and diff (F-014).
    for entry in by_device.values():
        entry["op_share"] = round(entry["op_count"] / planned if planned else 0.0, 6)
        entry["cost_share"] = round(entry["cost"] / total_cost if total_cost else 0.0, 6)
        entry["cost"] = round(entry["cost"], 6)
        entry["op_types"] = dict(sorted(entry["op_types"].items(),
                                        key=lambda kv: -kv[1]))
    total_cost = round(total_cost, 6)

    ane = by_device.get("ane", {"op_count": 0, "cost": 0.0,
                                "op_share": 0.0, "cost_share": 0.0})
    result = {
        "planned_ops": planned,
        "unplanned_ops": unplanned,
        "total_estimated_cost": total_cost,
        "by_device": by_device,
        "ane_residency_by_op_count": round(ane["op_share"], 6),
        "ane_residency_by_cost": round(ane["cost_share"], 6),
    }
    if ane_permitted:
        result["ops_ane_capable"] = ane_capable
        result["ops_not_ane_capable"] = sorted(set(not_ane_capable))
    else:
        result["ane_capability"] = ("not measurable — this compute unit "
                                    "excludes the ANE, so no op reports it "
                                    "as supported regardless of the op")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--precisions", nargs="+", default=["fp16", "fp32"])
    ap.add_argument("--units", nargs="+", default=list(UNITS), choices=list(UNITS))
    ap.add_argument("--models-dir", default=str(FIXTURES))
    ap.add_argument("--out", default=str(FIXTURES / "compute_plan.json"))
    args = ap.parse_args()

    md = Path(args.models_dir)
    report: dict = {"runs": {}}
    workdir = Path(tempfile.mkdtemp(prefix="inferenceprobe-mlmodelc-"))

    try:
        for precision in args.precisions:
            package = str(md / f"mobilenet_v3_large.{precision}.mlpackage")
            print(f"compiling {precision} ...")
            mlmodelc = ct.utils.compile_model(package, str(workdir / f"{precision}.mlmodelc"))

            for unit in args.units:
                key = f"{precision}.{unit}"
                result = analyse(mlmodelc, unit, ane_permitted=unit in ("cpuAndNE", "all"))
                report["runs"][key] = result

                shares = "  ".join(
                    f"{d}={v['op_count']:3d} ({v['op_share']:5.1%} ops, {v['cost_share']:5.1%} cost)"
                    for d, v in sorted(result["by_device"].items())
                )
                print(f"  {key:<16s} {shares}")

        # F-007 inferred that `.all` dispatches to the ANE from the fact that
        # `.all` and `.cpuAndNE` produced bit-identical output. That inference
        # was made on the **fp16** model, so only the fp16 row bears on it — a
        # difference in the fp32 row is a separate fact, not a contradiction.
        print()
        for precision in args.precisions:
            a, b = f"{precision}.all", f"{precision}.cpuAndNE"
            if a not in report["runs"] or b not in report["runs"]:
                continue
            ra, rb = report["runs"][a], report["runs"][b]
            same = (ra["by_device"].keys() == rb["by_device"].keys()
                    and all(ra["by_device"][d]["op_count"] == rb["by_device"][d]["op_count"]
                            for d in ra["by_device"]))
            report[f"{precision}_all_matches_cpuAndNE"] = same
            dev_a = "+".join(sorted(ra["by_device"]))
            dev_b = "+".join(sorted(rb["by_device"]))
            if same:
                note = ("confirms the F-007 inference directly"
                        if precision == "fp16" else "plans agree")
            else:
                note = ("F-007's inference was made on the fp16 model and does "
                        "not extend here" if precision == "fp32"
                        else "unexpected — F-007 predicted these would match")
            print(f"{precision}: .all -> {dev_a}, .cpuAndNE -> {dev_b}  ({note})")

    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
