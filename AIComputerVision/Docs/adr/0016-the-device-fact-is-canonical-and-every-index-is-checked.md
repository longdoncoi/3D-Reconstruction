# ADR 0016 — The device fact is canonical, and every CUDA index is checked

- Status: Accepted
- Date: 2026-10-25
- Supersedes: none
- Amends: ADR 0009 (the resolution is written in the framework's spelling,
  not the operator's), ADR 0015 (the fact comes from the argument set before
  `trainer.device`, and the gate compares it canonically)

## Context

The audit that closed N15–N18 compared the *recorded* fact with the *next
run's* resolution using the real framework instead of the fake backend
(`ultralytics 8.4.41`, `torch 2.7.0+cu128`, RTX 3050). Two defects survived
the request/fact split of ADR 0015, both at the same seam:

| `--device` | `select_device(...)` | `resolve_device(...)` | recorded `device_resolved` | compared with | skip? |
|---|---|---|---|---|---|
| `"0"` | `cuda:0` | `"0"` | `cuda:0` | `"0"` | **never** |
| `"0,1"` | multi-GPU, primary `cuda:0` | `"0,1"` | `cuda:0` (collapsed) | `"0,1"` | **never** |
| `"cuda:0"` | `cuda:0` | `"cuda:0"` | `cuda:0` | `cuda:0` | yes |
| `"cpu"` | `cpu` | `"cpu"` | `cpu` | `cpu` | yes |

`RunFacts.device_resolved` was read from `trainer.device`, a single
`torch.device`; `resolve_device` handed an explicit request back verbatim. The
two agreed only when the operator happened to write the framework's own
spelling (`cpu`, `cuda:0`). The fake backend in the test suite returned the
request as the resolution, so every offline test agreed with itself and the
defect was invisible. The default-device path was already covered by N15/N18;
the explicit numeric paths were not.

The same probe found the preflight counterpart: `check_device` asked only
whether CUDA was *available*. `--device 5` on a one-GPU machine passed, the
operator confirmed the retrain, and only then did the framework raise
`Invalid CUDA 'device=5'`. `--device 0,cpu` — a list the framework cannot
honour — passed the grammar check for the same reason: every component was
individually valid.

## Decision

- **One canonical spelling, in the framework's domain.** `canonical_device()`
  in `cvtrain/config.py` maps `None`/blank/`auto` to `auto`, a bare index
  (`0`) and `cuda` to `cuda:0`, and a list to `cuda:0,cuda:1`. The *request*
  (`TrainConfig.device_class`) keeps its own normalized spelling, so the
  manifest still shows what the operator typed; the *resolution* and the fact
  use the canonical spelling. `ultralytics` strips `cuda:` before selecting, so
  the canonical value can be forwarded without changing what training does.
- **`resolve_device` canonicalizes an explicit request.** `--device 0`
  resolves to `cuda:0`, which is what the framework reports back, so an
  identical rerun compares equal. `train_kwargs` forwards the same spelling.
- **The fact comes from the argument set before `trainer.device`.**
  `Trainer.args.device` preserves a multi-GPU list (`0,1`); `trainer.device`
  is a single primary device and collapses it. `_effective_facts` therefore
  reads `trainer.args` first, falls back to `trainer.device`, then to the
  orchestrator's resolution. The value is canonicalized on the way in, because
  a checkpoint may carry the operator's spelling.
- **The gate compares both sides canonically.** An entry written by an older
  build with `device_resolved: "0"` describes the same device as `cuda:0`; the
  comparison must not turn syntax into a different experiment. `"0"` and
  `"cuda:0"` are not two devices.
- **`check_device` validates every requested index.** It parses the list,
  rejects a mix of CUDA and non-CUDA components, requires CUDA to be
  available, reads `torch.cuda.device_count()`, and rejects any index
  `>= count` with the available range in the message. A broken `torch`
  (import failure, `is_available()` raising, `device_count()` raising) is a
  failed preflight, not an escaping traceback. `mps`/`metal` remain the
  framework's answer; this check cannot prove them.
- **The mutation set pins the class, not one representative.** Twelve new
  mutants (ten net, after replacing two stale gate anchors with three) cover
  the canonical spelling, the explicit resolution, the argument-set priority,
  the canonical comparison, the mixed-list rejection, the index bound, the
  index grammar and the three exception barriers in `check_device`; the
  `train_kwargs` anchor is updated to the canonical call.

## Consequences

- An identical rerun with `--device 0` or `--device 0,1` now skips, like
  `auto`, `cpu` and `cuda:0` already did. The regression tests are
  `test_an_explicit_gpu_index_rerun_is_skipped` and
  `test_an_explicit_multi_gpu_rerun_is_skipped`.
- A multi-GPU run records `cuda:0,cuda:1`, not the primary `cuda:0`, so it
  cannot pass for a single-GPU run.
- An invalid index or a mixed list fails before the retrain prompt, with the
  visible range in the message. The operator can correct the command instead
  of reading a framework traceback after confirming.
- An artifact written before this ADR with the operator's spelling is still
  current; the comparison is about the device, not the string.
- The N17 retrain already records `device_resolved: "cuda:0"` because the
  framework reports the concrete device, and the rerun that proves the skip
  runs under this canonical spelling — so the proof does not depend on the
  operator's spelling.
- The suite grows from 516 to **530** tests (14 new device tests). The mutation
  set grows from 116 to **126**, and `MIN_MUTANTS` is raised in the same
  change; the two stale anchors are replaced rather than deleted.
- `Docs/SCORECARD.md` records the defect as N19 (closed) with the measured
  probe and the test/mutant counts.
