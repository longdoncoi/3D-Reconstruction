"""Publication policy: is the published artifact current, and is it good enough?

Two rules, one module — they answer the same question from the two directions
an operator can ask it about an artifact already sitting in ``Models/``:

* :func:`already_current` — "would this invocation build exactly what is
  published?" (the ``--skip-unchanged`` contract);
* :func:`enforce_quality_gate` — "is that artifact good enough to keep
  there?" (the ``--min-metric`` contract).

Both are *verdicts about an artifact*, not steps of a pipeline, so they live
here rather than in :mod:`cvtrain.trainer`: the orchestrator asks for a verdict
and obeys it, it does not decide what the verdict means. Both are documented as
their own sections in the README, and this module is the single spelling of
each. :mod:`cvtrain.manifest` keeps the *recording* half of the same contract;
this module is the *comparing* half.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import manifest as manifest_mod
from .config import AUTO_BATCH, ModelSpec, TrainConfig, canonical_device
from .dataset import dataset_content_fingerprint
from .errors import QualityGateError
from .fsio import sha256_file
from .metrics import coerce_metric

LOG = logging.getLogger("cvtrain.gate")

__all__ = [
    "DatasetSnapshot",
    "InputSnapshot",
    "already_current",
    "dataset_content_digest",
    "dataset_digest",
    "enforce_quality_gate",
    "snapshot_dataset",
    "snapshot_inputs",
    "weights_digest",
]


@dataclass(frozen=True)
class DatasetSnapshot:
    """What the dataset looked like when a run started training.

    ``definition_sha256`` is the digest of ``data.yaml`` itself (cheap, O(1)),
    ``content_sha256`` the fingerprint of everything the YAML references
    (O(dataset bytes)). Both are optional because the file can be absent or
    unreadable; ``None`` means "cannot prove", never "matches".
    """

    definition_sha256: str | None
    content_sha256: str | None


@dataclass(frozen=True)
class InputSnapshot:
    """The inputs one model was trained from, captured *before* training ran.

    Recording after training would describe whatever the filesystem looked like
    when the export finished: a dataset or checkpoint edited mid-run would be
    misattributed to the artifact, and ``already_current`` would then trust the
    wrong inputs. The trainer captures this immediately before ``backend.train``
    and hands it to the manifest recorder.
    """

    source_weights_sha256: str | None
    dataset_sha256: str | None
    dataset_content_sha256: str | None


def _usable_metrics(metrics: object) -> dict[str, float]:
    """Keep only metrics this gate can actually compare against a threshold.

    Accepting :class:`object` is deliberate: one caller hands back numbers read
    out of a JSON manifest that a hand (or an older build) may have reshaped,
    and the other hands back whatever the framework returned. A gate that
    cannot read its input must fail the threshold — *fail closed* — rather than
    raise ``AttributeError``/``TypeError`` halfway through the run, and a
    ``NaN`` must never silently satisfy ``actual < minimum`` (it does not, in
    IEEE-754 arithmetic, which would publish a model that met no bar at all).
    """

    if not isinstance(metrics, dict):
        return {}
    usable: dict[str, float] = {}
    for name, value in metrics.items():
        number = coerce_metric(value)
        if number is not None:
            usable[str(name)] = number
    return usable


def weights_digest(spec: ModelSpec, cfg: TrainConfig) -> str | None:
    """Digest of the base checkpoint ``spec`` would train from, if it exists.

    Provenance has to record what the model was *started from*: the file name
    ``yolo11x.pt`` says nothing about which weights are inside it, and a
    replaced or re-downloaded checkpoint is a different training run waiting to
    happen. ``None`` means "not on disk" — a state :func:`already_current` must
    read as *cannot prove*, never as *matches*.
    """

    path = cfg.weights_path(spec)
    return sha256_file(path) if path.exists() else None


def dataset_digest(cfg: TrainConfig) -> str | None:
    """Digest of ``data.yaml``, if it exists.

    The path alone is not provenance either: the YAML decides the classes, the
    splits and where the images live, so an edited ``data.yaml`` behind an
    unchanged file name is a different experiment. ``None`` (the file is gone)
    is again "cannot prove", and :func:`already_current` retracts the skip.

    Deliberately the *definition*, not the images: this stays O(1) so a
    re-run can cheaply prove the YAML it used. The bytes live in
    :func:`dataset_content_digest`, which :func:`already_current` now requires
    in addition — an entry without it is retrained, never assumed current.
    """

    path = cfg.data_yaml
    return sha256_file(path) if path.exists() else None


def dataset_content_digest(cfg: TrainConfig, on_file: Callable[[int, int], None] | None = None) -> str | None:
    """Digest of the dataset contents (paths and file bytes), if data.yaml exists.

    Complements :func:`dataset_digest` by hashing the actual files under
    ``train``/``val``. Because the bytes are hashed — not just names and
    sizes — an image replaced with different pixels under the same name and
    size is still a different dataset, which is exactly what
    ``--skip-unchanged`` must not miss.

    ``on_file`` is forwarded to the fingerprint so a caller can report
    progress while the dataset is read.
    """

    return dataset_content_fingerprint(cfg.data_yaml, on_file=on_file)


def snapshot_dataset(cfg: TrainConfig, on_file: Callable[[int, int], None] | None = None) -> DatasetSnapshot:
    """Capture the dataset's definition and content digests once per run.

    Computing the content fingerprint once (instead of once per model and again
    when recording provenance) keeps a multi-model run O(dataset) rather than
    O(models x dataset), and the single snapshot is what both the skip rule and
    the manifest recorder compare against.
    """

    return DatasetSnapshot(
        definition_sha256=dataset_digest(cfg),
        content_sha256=dataset_content_digest(cfg, on_file=on_file),
    )


def snapshot_inputs(
    spec: ModelSpec,
    cfg: TrainConfig,
    *,
    dataset: DatasetSnapshot | None = None,
    source_weights: Path | None = None,
) -> InputSnapshot:
    """Capture everything ``spec`` is about to be trained from.

    Called immediately before ``backend.train``. ``dataset`` lets a caller
    reuse the run-wide :func:`snapshot_dataset` result; when omitted (direct
    calls, tests) the dataset is looked at here, so the function is never able
    to return a snapshot that disagrees with its own observation.

    ``source_weights`` is the file the backend's plan says training will start
    from. A resumed run starts from the interrupted run's ``last.pt``, not from
    the base checkpoint the spec names, so hashing ``spec`` would record a
    digest that never entered the model.
    """

    ds = dataset if dataset is not None else snapshot_dataset(cfg)
    source = source_weights if source_weights is not None else cfg.weights_path(spec)
    return InputSnapshot(
        source_weights_sha256=sha256_file(source) if source.exists() else None,
        dataset_sha256=ds.definition_sha256,
        dataset_content_sha256=ds.content_sha256,
    )


def already_current(
    spec: ModelSpec,
    cfg: TrainConfig,
    manifest: manifest_mod.Manifest,
    backend_version: str,
    *,
    export_args: dict[str, object],
    dataset: DatasetSnapshot | None = None,
) -> bool:
    """Whether the published ONNX already *is* the artifact this run would build.

    The rule is deliberately strict, because "skip" must never hide work the
    operator asked for. Every one of these has to hold:

    * the artifact exists at the path the Qt app loads;
    * every recorded hyper-parameter matches this invocation — ``epochs``,
      ``imgsz``, ``seed``, ``opset``, ``batch`` and ``workers``. Omitting
      ``batch``/``workers`` would let a differently-trained run hide behind an
      identical file name. These fields are the **request**: the framework's
      resolved batch or rounded image size is recorded separately and is not
      what the next invocation asks for again (ADR 0015);
    * the export request matches too (``export_args``). The same weights
      exported with another opset, simplification or input size are another
      artifact, so the exporter options are part of "what this run would
      build". An entry that never recorded them is unprovable and retrained;
    * the dataset *definition* still hashes the same (``dataset_sha256``) and
      the ``device`` class matches — a re-pointed dataset or a CPU-vs-GPU run
      is a different artifact behind the same parameters. The dataset is
      compared by its **digests, not by its path**: the path is recorded as
      provenance, but a copy of the same dataset at another absolute path is
      the same experiment, and a verdict that depended on where the data
      happens to live would not survive a checkout on another machine;
    * the **resolved** device matches too (``device_resolved``). The requested
      device says what was asked for; the resolved one says what ran. Without
      the second, a GPU-trained artifact would pass for a CPU run whenever
      both were requested as ``auto``. Both sides must be known: an
      unresolved configuration is retrained too, because it cannot know what
      it would build. The comparison is made in the framework's spelling
      (``0`` ≡ ``cuda:0``), so two names for one device are not two
      experiments;
    * the requested batch is not ``AUTO_BATCH``. The automatic batch is sized
      by the framework from the machine's memory, so this run's batch is not
      known before training — and comparing only the request (``-1`` on both
      sides) would let a differently-sized run pass for the published one.
      A batch the operator named is comparable; one the framework guesses is
      not, so the model is retrained;
    * the dataset *content* still hashes the same
      (``dataset_content_sha256``). An entry without that digest is not
      comparable at all: "we did not record it" means "not vouchable", so the
      model is retrained rather than skipped on the strength of the cheaper
      definition digest alone;
    * the backend that produced it is the backend that would produce it now —
      a framework upgrade changes the export, so it invalidates the claim;
    * its digest matches what the manifest recorded for it (a tampered or
      half-replaced file is retrained, not trusted);
    * the base checkpoint on disk still has the digest that was recorded when
      the artifact was trained. Different starting weights are a different
      model, even with every other parameter identical;
    * the artifact was not produced by ``--resume``. A resumed run started
      from another run's checkpoint and ignored this invocation's arguments,
      so a fresh run would not reproduce it even when every recorded value
      happens to agree.

    An entry adopted from disk (``--write-manifest``) has no hyper-parameters,
    so it never satisfies the second condition and is retrained: provenance we
    do not have cannot be assumed to match. An entry written before a factor
    was recorded fails the same way, for the same reason.

    ``export_args`` is required and keyword-only: it is the backend's own
    answer to "what would :meth:`TrainingBackend.export` be told?", and a
    caller that cannot produce it cannot claim the published artifact is
    current.

    ``dataset`` reuses a run-wide :func:`snapshot_dataset` result — a
    multi-model run then hashes the data once instead of once per model. When
    omitted, this call reads the dataset itself.

    The checks run cheapest-first — the two digests (the published artifact and
    a checkpoint that can be hundreds of megabytes) are only computed once
    everything free already agrees.
    """

    target = cfg.output_path(spec)
    if not target.exists():
        return False
    entry = (manifest.get("entries") or {}).get(spec.output)
    if entry is None:
        return False
    # An artifact produced by a *resumed* run did not start from the base
    # checkpoint and did not follow this invocation's argument set: the
    # framework restored the interrupted run's experiment. Even if every
    # recorded value happens to agree, a non-resume run would not reproduce it,
    # so the artifact is not "what this invocation would build".
    if entry.get("resume_from"):
        return False
    # The automatic batch is a request the framework turns into a machine-sized
    # answer during training. "Cannot know what this run would build" is
    # retrained, exactly like an unresolved device.
    if cfg.batch == AUTO_BATCH:
        return False
    if entry.get("export_args") != export_args:
        return False
    recorded = (
        entry.get("epochs"),
        entry.get("imgsz"),
        entry.get("seed"),
        entry.get("opset"),
        entry.get("batch"),
        entry.get("workers"),
        entry.get("backend_version"),
        entry.get("device"),
    )
    requested = (
        cfg.epochs,
        cfg.imgsz,
        cfg.seed,
        cfg.opset,
        cfg.batch,
        cfg.workers,
        backend_version,
        cfg.device_class,
    )
    if recorded != requested:
        return False
    # The requested device is a *request* ("auto" means "let the framework
    # choose"); the resolved one is the fact. An artifact trained on ``cuda:0``
    # is not what a CPU run would build, and the request alone cannot tell the
    # two apart — so the resolved device is compared too. An entry that never
    # recorded one is unprovable about its device and is retrained.
    #
    # *Both* sides fail closed. A configuration that was never resolved cannot
    # know what this run would train on, so it cannot claim the published
    # artifact is the same one: "cannot prove" is retrained, exactly like
    # every other unknown in this function. Comparing only the request would
    # let two entries that agree on "auto" and disagree on the machine pass
    # for each other. (The orchestrator always resolves before it asks —
    # ``trainer._execute`` — so this guard protects the *next* caller rather
    # than the current path.)
    if cfg.device_resolved is None:
        return False
    # Compare in the framework's spelling (``0`` ≡ ``cuda:0``,
    # ``0,1`` ≡ ``cuda:0,cuda:1``). An entry written by an older build may
    # carry the operator's spelling; it describes the same device, so the
    # comparison must not turn syntax into a different experiment (ADR 0016).
    if canonical_device(entry.get("device_resolved")) != canonical_device(cfg.device_resolved):
        return False
    # The dataset *path* is recorded as provenance but deliberately **not**
    # compared: a path is not a fact about the data, it is a fact about one
    # machine. What proves "the same dataset" is the digest of ``data.yaml``
    # plus the fingerprint of every file it reaches — checked below. Comparing
    # the path as well would make the verdict machine-bound: the very same
    # dataset, copied to another directory (or checked out at another absolute
    # path), would be retrained for a difference that cannot change the
    # artifact.
    recorded_dataset = dataset.definition_sha256 if dataset is not None else dataset_digest(cfg)
    if recorded_dataset is None or entry.get("dataset_sha256") != recorded_dataset:
        return False
    entry_content = entry.get("dataset_content_sha256")
    recorded_content = dataset.content_sha256 if dataset is not None else dataset_content_digest(cfg)
    if not isinstance(entry_content, str) or not entry_content or entry_content != recorded_content:
        return False
    if entry.get("sha256") != sha256_file(target):
        return False
    weights_sha = weights_digest(spec, cfg)
    return weights_sha is not None and entry.get("source_weights_sha256") == weights_sha


def enforce_quality_gate(cfg: TrainConfig, spec: ModelSpec, metrics: object) -> None:
    """Refuse to publish (or keep) an artifact below the requested bar.

    Called right after training — *before* the export is published, so a model
    nobody would want never reaches the Qt app — and again against the metrics
    recorded for a model :func:`already_current` just decided to skip. Without
    the second call ``--skip-unchanged --min-metric ...`` would quietly pass on
    a model that never met the bar, which is precisely the "skip hides work"
    failure this pipeline refuses to allow elsewhere.

    A metric the backend did not report — or reported in a shape this gate
    cannot compare (see :func:`_usable_metrics`) — is a failure, not a pass:
    the operator asked to gate on it, so silently not checking would defeat the
    request. The message lists what *was* reported, which is usually a typo in
    the name.
    """

    if not cfg.min_metrics:
        return
    reported = _usable_metrics(metrics)
    for name, minimum in cfg.min_metrics:
        actual = reported.get(name)
        if actual is None:
            names = ", ".join(sorted(reported)) or "none"
            raise QualityGateError(
                f"{spec.output}: metric {name!r} was not reported as a usable number (reported: {names}); "
                "fix or drop that --min-metric threshold"
            )
        if actual < minimum:
            raise QualityGateError(
                f"{spec.output}: {name}={actual:.4f} is below the required minimum {minimum:.4f}; "
                "train longer, improve the dataset, or lower --min-metric"
            )
        LOG.info("Quality gate ok: %s %s=%.4f >= %.4f", spec.output, name, actual, minimum)
