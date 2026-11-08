"""Run the trained QCNN circuit with Qiskit: exactly, with shot noise, with a device's noise, or on IBM hardware.

Needs results/quantum/qcnn_inputs.npz from 08_export_qcnn_inputs.py and the packages in
requirements-quantum.txt. Only the 8-qubit circuit runs here; its 8 input angles per test
image were computed by the trained backbone. The circuit is rebuilt gate by gate from
common.qcnn_layer, and `check` confirms it matches PennyLane exactly.

    python 09_quantum_hardware.py check                         # Qiskit circuit = PennyLane circuit?
    python 09_quantum_hardware.py ideal --shots 1000            # perfect device, finite shots
    python 09_quantum_hardware.py noisy --fake FakeFez          # simulated noise of an IBM device
    python 09_quantum_hardware.py ibm --backend ibm_fez --images 10   # real device: small trial first
    python 09_quantum_hardware.py ibm --backend ibm_fez               # all 450 test images
    python 09_quantum_hardware.py collect results/quantum/ibm_job_....json   # fetch a finished job
    python 09_quantum_hardware.py summary                       # table of all runs so far

The IBM modes need a saved IBM Quantum account (see the README). A real-device job is
saved as results/quantum/ibm_job_<backend>_<time>.json as soon as it is submitted, so the
results can be collected later even if this script is stopped while the job waits in
the queue.

Every run is compared with the exact circuit on the same images: accuracy, how many
predictions change, and an exact McNemar test.
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.stats import binomtest

QUANTUM_DIR = Path(__file__).resolve().parent / "results" / "quantum"
N_QUBITS = 8


# --- the circuit (same gates and order as common.qcnn_layer) ---------------------------------
def _conv(qc, w, a, b):
    qc.ry(w[0], a)
    qc.ry(w[1], b)
    qc.cx(a, b)
    qc.ry(w[2], a)
    qc.ry(w[3], b)
    qc.cx(b, a)
    qc.rz(w[4], a)
    qc.rz(w[5], b)


def _pool(qc, w, src, sink):
    qc.crz(w[0], src, sink)
    qc.x(src)
    qc.crx(w[1], src, sink)
    qc.x(src)


def qcnn_circuit(conv_w, pool_w, measure=True):
    """8-qubit QCNN with the trained weights fixed and the 8 input angles as parameters.

    Returns the circuit, its input parameters and the qubit whose <Z> is the output.
    """
    x = ParameterVector("x", N_QUBITS)
    qc = QuantumCircuit(N_QUBITS, 1 if measure else 0)
    for i in range(N_QUBITS):  # AngleEmbedding with RY
        qc.ry(x[i], i)
    active = list(range(N_QUBITS))
    for stage in range(3):
        n = len(active)
        pairs = [(active[i], active[i + 1]) for i in range(0, n, 2)]
        if n > 2:
            pairs += [(active[i], active[(i + 1) % n]) for i in range(1, n, 2)]
        for a, b in pairs:
            _conv(qc, conv_w[stage], a, b)
        for i in range(0, n, 2):
            _pool(qc, pool_w[stage], active[i], active[i + 1])
        active = active[1::2]
    if measure:
        qc.measure(active[0], 0)
    return qc, x, active[0]


# --- inputs, outputs and scoring --------------------------------------------------------------
def load_inputs(folder):
    f = folder / "qcnn_inputs.npz"
    if not f.exists():
        raise SystemExit(f"{f} not found: run 08_export_qcnn_inputs.py first")
    d = np.load(f)
    return {k: d[k] for k in d.files}


def z_from_bits(bitarray):
    """<Z> of the measured qubit for each input, from the measured bits (0 -> +1, 1 -> -1)."""
    ones = bitarray.array[..., -1] & 1  # shape (inputs, shots); one classical bit
    return 1.0 - 2.0 * ones.mean(axis=-1)


def score(inp, z, sel):
    """Accuracy and the paired comparison with the exact circuit on the same images."""
    y, z_exact = inp["y"][sel], inp["z_exact"][sel]
    s, s_exact = inp["k"] * z + inp["b"], inp["k"] * z_exact + inp["b"]
    pred, pred_exact = (s >= 0).astype(int), (s_exact >= 0).astype(int)
    ok, ok_exact = pred == y, pred_exact == y
    only_run, only_exact = int((ok & ~ok_exact).sum()), int((~ok & ok_exact).sum())
    tp, fn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 1)).sum())
    tn, fp = int(((pred == 0) & (y == 0)).sum()), int(((pred == 1) & (y == 0)).sum())
    return {"n_images": int(len(y)), "accuracy": float(ok.mean()), "exact_accuracy": float(ok_exact.mean()),
            "sensitivity": tp / (tp + fn) if tp + fn else None, "specificity": tn / (tn + fp) if tn + fp else None,
            "changed_predictions": int((pred != pred_exact).sum()),
            "only_this_run_right": only_run, "only_exact_right": only_exact,
            "mcnemar_p": float(binomtest(only_run, only_run + only_exact).pvalue) if only_run + only_exact else 1.0,
            "mean_abs_z_error": float(np.abs(z - z_exact).mean())}


def circuit_stats(isa):
    ops = isa.count_ops()
    two_q = sum(c for name, c in ops.items() if name in ("cz", "ecr", "cx", "rzz"))
    return {"two_qubit_gates": int(two_q), "depth": int(isa.depth()), "ops": {k: int(v) for k, v in ops.items()}}


def save_run(folder, name, meta, inp, z, sel):
    result = {**meta, **score(inp, z, sel), "saved": datetime.now().isoformat(timespec="seconds")}
    np.savez(folder / f"{name}.npz", index=inp["index"][sel], z=z, z_exact=inp["z_exact"][sel], y=inp["y"][sel])
    (folder / f"{name}.json").write_text(json.dumps(result, indent=2))
    print(f"\n{name}: accuracy {100 * result['accuracy']:.2f}% on {result['n_images']} images "
          f"(exact circuit {100 * result['exact_accuracy']:.2f}%); {result['changed_predictions']} predictions "
          f"changed; McNemar p = {result['mcnemar_p']:.3f}; mean |z error| {result['mean_abs_z_error']:.3f}")
    summary(folder)


def summary(folder):
    rows = [json.loads(f.read_text()) for f in sorted(folder.glob("*.json"))
            if f.name not in ("qcnn_inputs.json",) and not f.name.startswith("ibm_job_")]
    if not rows:
        print("no runs yet")
        return
    import csv
    keys = ["mode", "backend", "shots", "n_images", "accuracy", "exact_accuracy", "changed_predictions",
            "only_this_run_right", "only_exact_right", "mcnemar_p", "mean_abs_z_error", "two_qubit_gates", "depth"]
    with open(folder / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n{'run':34s} {'images':>6s} {'acc %':>7s} {'exact %':>8s} {'changed':>8s} {'McNemar p':>10s}")
    for r in rows:
        name = f"{r['mode']} {r.get('backend') or ''} {r.get('shots') or ''}".strip()
        print(f"{name:34s} {r['n_images']:6d} {100 * r['accuracy']:7.2f} {100 * r['exact_accuracy']:8.2f} "
              f"{r['changed_predictions']:8d} {r['mcnemar_p']:10.3f}")
    print(f"(saved to {folder / 'summary.csv'})")


# --- modes ------------------------------------------------------------------------------------
def run_check(inp):
    qc, x, out_q = qcnn_circuit(inp["conv_w"], inp["pool_w"], measure=False)
    obs = SparsePauliOp.from_sparse_list([("Z", [out_q], 1.0)], num_qubits=N_QUBITS)
    z = np.array([Statevector(qc.assign_parameters(a)).expectation_value(obs).real for a in inp["angles"]])
    diff = np.abs(z - inp["z_exact"]).max()
    print(f"Qiskit vs PennyLane, exact <Z> on {len(z)} test images: largest difference {diff:.1e} "
          f"(the trained model computes in float32, so about 1e-6 is expected; a wrong gate would give ~0.1)")
    if diff > 1e-5:
        raise SystemExit("the Qiskit circuit does not match the trained PennyLane circuit")
    print("OK: the Qiskit circuit is the trained QCNN")


def sample(backend, inp, sel, shots, seed):
    """Transpile once for `backend`, then run all selected inputs as one parameter sweep."""
    from qiskit_ibm_runtime import SamplerV2
    qc, x, _ = qcnn_circuit(inp["conv_w"], inp["pool_w"])
    isa = transpile(qc, backend=backend, optimization_level=3, seed_transpiler=seed)
    stats = circuit_stats(isa)
    print(f"circuit on {getattr(backend, 'name', backend)}: {stats['two_qubit_gates']} two-qubit gates, "
          f"depth {stats['depth']}; {len(sel)} images x {shots} shots", flush=True)
    sampler = SamplerV2(mode=backend)
    if hasattr(sampler.options, "simulator"):
        sampler.options.simulator.seed_simulator = seed
    params = inp["angles"][sel][:, [list(x).index(p) for p in isa.parameters]]  # columns in the circuit's order
    return sampler, isa, params, stats


def run_local(inp, folder, mode, shots, sel, fake, seed):
    from qiskit_aer import AerSimulator
    if mode == "ideal":
        backend, label = AerSimulator(seed_simulator=seed), "aer"
    else:
        from qiskit_ibm_runtime import fake_provider
        if not hasattr(fake_provider, fake):
            raise SystemExit(f"unknown --fake {fake}; e.g. FakeFez, FakeTorino, FakeMarrakesh, FakeBrisbane")
        backend, label = getattr(fake_provider, fake)(), fake
    sampler, isa, params, stats = sample(backend, inp, sel, shots, seed)
    t = time.time()
    res = sampler.run([(isa, params)], shots=shots).result()[0]
    z = z_from_bits(res.data.c)
    meta = {"mode": mode, "backend": label, "shots": shots, "seed": seed, "seconds": round(time.time() - t, 1), **stats}
    save_run(folder, f"{mode}_{label}_shots{shots}" + (f"_n{len(sel)}" if len(sel) < len(inp["y"]) else ""),
             meta, inp, z, sel)


def ibm_service():
    from qiskit_ibm_runtime import QiskitRuntimeService
    try:
        return QiskitRuntimeService()
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"could not open the saved IBM Quantum account ({e}). Save it once with\n"
                         "  python -c \"from qiskit_ibm_runtime import QiskitRuntimeService as S; "
                         "S.save_account(token='YOUR_API_KEY', instance='YOUR_INSTANCE_CRN', set_as_default=True)\"")


def run_ibm(inp, folder, args, sel):
    service = ibm_service()
    backend = (service.backend(args.backend) if args.backend
               else service.least_busy(operational=True, simulator=False, min_num_qubits=N_QUBITS))
    sampler, isa, params, stats = sample(backend, inp, sel, args.shots, args.seed)
    total = len(sel) * args.shots
    print(f"{total:,} shots in total. Check the usage of the trial job on the IBM dashboard before running "
          f"all images; free accounts have a monthly time limit.")
    if not args.yes and input("Submit to the real device? [y/N] ").strip().lower() != "y":
        raise SystemExit("not submitted")
    job = sampler.run([(isa, params)], shots=args.shots)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    record = {"job_id": job.job_id(), "backend": backend.name, "shots": args.shots, "seed": args.seed,
              "indices": [int(i) for i in sel], "submitted": stamp, **stats}
    rec_file = folder / f"ibm_job_{backend.name}_{stamp}.json"
    rec_file.write_text(json.dumps(record, indent=2))
    print(f"submitted job {job.job_id()} to {backend.name}; saved {rec_file}")
    if args.wait:
        print("waiting for the result (Ctrl+C stops waiting; the job keeps running; collect it later)", flush=True)
        collect(inp, folder, rec_file)
    else:
        print(f"collect it later with: python 09_quantum_hardware.py collect \"{rec_file}\"")


def collect(inp, folder, rec_file):
    rec = json.loads(Path(rec_file).read_text())
    job = ibm_service().job(rec["job_id"])
    print(f"job {rec['job_id']} on {rec['backend']}: {job.status()}", flush=True)
    res = job.result()  # waits if still queued or running; raises if the job failed or was cancelled
    z = z_from_bits(res[0].data.c)
    sel = np.array(rec["indices"])
    usage = job.usage() if hasattr(job, "usage") else None
    meta = {"mode": "ibm", "backend": rec["backend"], "shots": rec["shots"], "seed": rec["seed"],
            "job_id": rec["job_id"], "quantum_seconds": usage, **{k: rec[k] for k in ("two_qubit_gates", "depth", "ops")}}
    save_run(folder, f"ibm_{rec['backend']}_shots{rec['shots']}" + (f"_n{len(sel)}" if len(sel) < len(inp["y"]) else ""),
             meta, inp, z, sel)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["check", "ideal", "noisy", "ibm", "collect", "summary"])
    ap.add_argument("job_file", nargs="?", help="for collect: the ibm_job_*.json file")
    ap.add_argument("--folder", default=str(QUANTUM_DIR), help="where qcnn_inputs.npz is and results go")
    ap.add_argument("--shots", type=int, default=1000)
    ap.add_argument("--images", type=int, help="use only the first N test images (e.g. 10 for a trial)")
    ap.add_argument("--fake", default="FakeFez", help="noisy mode: local copy of an IBM device (default FakeFez)")
    ap.add_argument("--backend", help="ibm mode: device name, e.g. ibm_fez (default: the least busy one)")
    ap.add_argument("--seed", type=int, default=0, help="seed for transpiling and simulating")
    ap.add_argument("--wait", action="store_true", help="ibm mode: wait for the result instead of exiting")
    ap.add_argument("--yes", action="store_true", help="ibm mode: submit without asking")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    folder = Path(args.folder)
    if args.mode == "summary":
        summary(folder)
        return
    inp = load_inputs(folder)
    sel = np.arange(len(inp["y"]) if args.images is None else min(args.images, len(inp["y"])))
    if args.mode == "check":
        run_check(inp)
    elif args.mode in ("ideal", "noisy"):
        run_local(inp, folder, args.mode, args.shots, sel, args.fake, args.seed)
    elif args.mode == "ibm":
        run_ibm(inp, folder, args, sel)
    else:
        if not args.job_file:
            ap.error("collect needs the ibm_job_*.json file")
        collect(inp, folder, args.job_file)


if __name__ == "__main__":
    main()
