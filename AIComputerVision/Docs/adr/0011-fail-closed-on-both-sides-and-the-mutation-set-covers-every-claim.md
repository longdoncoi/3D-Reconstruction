# ADR 0011 — A fail-closed rule fails closed on both sides; the provenance records what ran; the mutation set covers every claim

- Status: Accepted
- Date: 2026-10-21
- Supersedes: none
- Amends: ADR 0007 (its "the mutation set is the contract" rule, by widening
  the set) and ADR 0009 (the device comparison, by making it fail closed on
  the *configuration* side as well as the entry side)
- Amended by: [ADR 0013](0013-resume-is-provenance-effective-facts-and-a-lease-is-evidence-of-life.md)
  (the epoch decision, split into effective budget vs. completed count, and the
  mutation-set count, now 101), [ADR 0016](0016-the-device-fact-is-canonical-and-every-index-is-checked.md)
  (the device comparison is canonical on both sides, and every CUDA index is
  validated before the prompt)

## Context

A chief-architecture review of the package found two gaps, both of the same
shape: a rule that is stated as fail-closed, and one place where it is not.

**The device comparison was fail-closed on one side only.** ADR 0009 makes
the resolved device the comparable fact, and `gate.already_current` compares
it against the entry. But the comparison read:

```python
if cfg.device_resolved is not None and entry.get("device_resolved") != cfg.device_resolved:
    return False
```

The *entry* side failed closed — an entry that never recorded a resolution is
unprovable and is retrained. The *configuration* side did not: a configuration
that was never resolved skipped the check altogether. Every other unknown in
that function — a missing dataset digest, an absent checkpoint digest, an
artifact whose digest is not recorded — reads as "cannot prove" and is
retrained. The device was the single exception.

The production path was not affected: `trainer._execute` resolves the device
before the skip rule ever runs, and it is the only caller. The defect was a
trap for the *next* caller — a public function whose contract says "fails
closed" and does not.

**The provenance recorded the requested epochs, not the ones that ran.** The
manifest's `epochs` field was written from `cfg.epochs` — the number the
invocation *asked for*. But a resumed run does not use it: Ultralytics'
`check_resume` restores the whole argument set from the checkpoint, and the
list of keys an override may change does **not** include `epochs`. The
framework therefore owns the epoch count of a run it continues, and the
command line is ignored. Recording the request would describe a run that
never happened — and the field is compared by `gate.already_current`, so a
wrong value makes a later `--skip-unchanged` verdict trust a provenance
statement that is not true.

**The mutation set did not cover every claim the documentation makes.**
ADR 0007 states the rule the package lives by: *"the mutation set is the
contract ... a decision that is not listed there is a decision nobody has
pinned"*. Measured against the code, that sentence was stronger than the
truth. These load-bearing decisions were pinned by a test but carried no
mutant:

| Decision | Claimed in | Mutant |
|---|---|---|
| `TrainConfig.device_class` normalises the request (`""`/`"auto"`, `"0, 1"` → `"0,1"`, lower-cased) | `config.py`, ADR 0009 | none |
| `train_kwargs` forwards the *resolved* device, not the request | `backend.py`, ADR 0009 | none |
| `--min-metric` syntax is parsed here; semantics belong to preflight | `cli.py`, `preflight.py` | none |
| `--device` accepts exactly the grammar the backend forwards | `preflight.py` | none |
| an unknown model role is *rejected*, not guessed | `onnx_contract.py` | none |
| an unreadable dataset file contributes a stable marker | `dataset.py` | none |
| a lock that cannot be refreshed warns instead of raising | `fsio.py` | none |

Under ADR 0007's own rule those were unpinned decisions. They were not wrong,
and each had a test — but "a test exists" and "the mutation set is the
contract" are two different claims, and the package was making the second one
while only satisfying the first.

**Writing the missing mutants found a test that pinned nothing.** The mutant
for "an unreadable dataset file contributes a stable marker" *survived* — and
the reason was the test it was supposed to kill. `test_dataset.py` asserted
that the fingerprint of a dataset with an unreadable file is a 64-character
string, which is true of every fingerprint the package has ever produced. It
never compared that digest with the digest of the same dataset whose file was
readable — so removing the marker entirely left the suite green. The claim
was true, the test was named after it, and the test did not test it: exactly
the failure mode ADR 0007 exists to prevent, one level down.

## Decision

- **Both sides of the device comparison fail closed.** An unresolved
  configuration cannot know what this run would train on, so it cannot claim
  the published artifact is the same one: it is retrained, like every other
  unknown in the rule. The guard is explicit rather than folded into the
  comparison, because "cannot prove" deserves to be readable as its own rule.
- **The provenance records the epochs the framework ran.** The backend reads
  the trainer's own counter (`Trainer.epochs`) and hands it to the recorder;
  the request is used only when the framework cannot report one. "Cannot
  prove" stays *absent* rather than becoming a number nobody observed.
  [ADR 0013](0013-resume-is-provenance-effective-facts-and-a-lease-is-evidence-of-life.md)
  keeps the rule and separates the two facts it was carrying: `epochs` is the
  effective *budget*, and `epochs_completed` is the count of epochs actually
  completed.
- **The mutation set grows to cover every claim above.** Nine mutants were
  added — one per decision in the table, plus one for the new guard, plus one
  for the epoch provenance. The set is the contract; a claim that is not in it
  is a claim the package does not make.
- **ADR 0007's wording is kept but scoped.** The rule stays: *the mutation
  set is the contract*. What it covers is stated precisely — every
  load-bearing decision the package's documentation claims to protect. A
  decision with a test and no mutant is not "unpinned" (the test pins it);
  it is a decision the mutation set has not yet reached, and the amendment
  below records that the set has now reached it.

## Consequences

- `gate.already_current` has one behaviour instead of two, and the new
  behaviour is the one its docstring already promised.
- The manifest's `epochs` field is true again after a resume, and the
  `--skip-unchanged` verdict compares two facts about the same thing. [ADR
  0013](0013-resume-is-provenance-effective-facts-and-a-lease-is-evidence-of-life.md)
  adds `epochs_completed` and `resume_from`, so the budget, the work actually
  done and the checkpoint it continued are three separately recorded facts.
- Two tests that *claimed* to pin the unreadable-file marker now actually
  pin it: they compare the digest of an unreadable file with the digest of an
  empty one, which is the property the marker's existence guarantees. The
  mutant that removes the marker is killed by them.
- The mutation set is 67 → 76 decisions here; [ADR 0010](0010-retention-is-a-policy-not-an-accident.md)
  added four more for the retention rules, so the package reached **80**
  mutants. [ADR 0013](0013-resume-is-provenance-effective-facts-and-a-lease-is-evidence-of-life.md)
  added the resume, effective-fact, metric and lease decisions for a total of
  **101**. Every one of them is killed by the suite; the gate is the
  evidence, and the count in `Docs/SCORECARD.md` is re-measured rather than
  estimated.
- A new claim added to the README or an ADR now has an explicit home: the
  mutation set, and therefore a mutant in the same commit.
- ADR 0007's amendment records what changed and why, so the history of the
  rule stays readable instead of being quietly rewritten.
