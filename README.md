# BioGuard

BioGuard is a research prototype for studying drug–drug association prediction
from paired molecular structures. It is not a clinical decision-support system,
does not estimate interaction incidence or causality, and must not guide
medication decisions.

## Repository status

The repository is undergoing a scientific and engineering refactor.

- The bundled GAT checkpoint is a legacy artifact with unknown training
  provenance.
- Its architecture is verified before loading, but its operating threshold and
  stored calibrator are disabled because their validation provenance is unknown.
- The TWOSIDES loader now preserves drug–event–drug relations.
- Missing TWOSIDES associations are represented as **unlabeled**, never as
  confirmed non-interactions.
- A binary “any reported event” dataset remains available only as an explicitly
  derived positive-unlabeled (PU) compatibility task.
- An event-conditioned relation decoder is not implemented yet.
- The legacy enzyme table covers a closed drug set. Molecules absent from that
  table currently receive zero enzyme vectors because no deployable CYP
  predictor artifact is present.

## Data contracts

### Primary source representation

TWOSIDES is retained as positive association triples:

```text
(drug_a, event_id, drug_b)
```

Exact duplicate triples are counted, while different event identifiers for the
same drug pair remain separate records. Standardization, exclusions, source
counts, and record hashes are captured in a manifest.

```python
from bioguard.data_loader import load_twosides_relations

bundle = load_twosides_relations()
relations = bundle.records
manifest = bundle.manifest
```

These records are pharmacovigilance associations. They are not proof of
causality, severity, incidence, or a patient-specific outcome.

### Binary positive-unlabeled derivative

The current GAT can only emit one pair-level score. For compatibility
experiments, event triples can therefore be collapsed into “any reported event”
positives. Sampled missing pairs are explicitly annotated as unlabeled:

```python
from bioguard.data_loader import load_twosides_binary_pu

bundle = load_twosides_binary_pu(
    split_method="scaffold_disjoint",
    train_unlabeled_ratio=1.0,
    val_unlabeled_ratio=1.0,
    test_unlabeled_ratio=1.0,
)
```

Important columns include:

| Column | Meaning |
|---|---|
| `event_ids` | All retained TWOSIDES event identifiers for a positive pair |
| `observation` | `positive` or `unlabeled` |
| `target` | Optimization indicator; zero means unlabeled, not negative |
| `negative_evidence` | Explicit evidence status for the sampled pair |
| `split` | Auditable split assignment |

The binary training path uses non-negative PU risk. It requires a documented
class-prior assumption; the software deliberately supplies no universal
default.

## Split regimes

Three split contracts are implemented:

| Split | Contract |
|---|---|
| `pair_random` | Exact drug pairs are disjoint; individual drugs may overlap |
| `drug_disjoint` | Both drugs belong to disjoint train/validation/test drug pools |
| `scaffold_disjoint` | Drug pools are grouped by stereochemistry-aware Murcko scaffold; both drugs must belong to the same partition |

For drug- and scaffold-disjoint splits, cross-partition relations are recorded
as `excluded_cross_partition` instead of being silently reassigned. Manifests
report exclusions and overlap counts.

## Model

The legacy `BioGuardGAT` has two swap-invariant branches:

1. A shared GATv2 molecular encoder using 46-dimensional atom features and
   8-dimensional edge features.
2. A learned head over 60-dimensional legacy enzyme-table vectors.

A sigmoid gate combines the two logits. The gate value is telemetry, not a
causal explanation or calibrated pathway attribution.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The dependency file is not yet a complete lockfile. Reproducible release images
remain a refactor milestone.

## Data preparation

Build the event-preserving relation cache:

```bash
python -c "from bioguard.data_loader import load_twosides_relations; load_twosides_relations()"
```

Generate the LMDB molecular-graph cache:

```bash
python -m bioguard.build_lmdb
```

Molecules that fail graph featurization are written to
`data/lmdb_featurization_report.json`. Source labels are never overwritten.

Validate and export the split/leakage report:

```bash
python validate_data.py --split scaffold_disjoint
```

## Training

Training requires an explicit PU class-prior assumption:

```bash
python -m bioguard.main train \
  --split scaffold_disjoint \
  --class-prior <documented-value> \
  --unlabeled-ratio 1.0 \
  --epochs 50
```

The class prior must come from a prespecified analysis or sensitivity range; it
must not be selected on the final test set. New checkpoints embed their data and
split manifests and are checksummed with their feature artifacts.

## Evaluation and baselines

```bash
python -m bioguard.main eval
python -m bioguard.main baselines \
  --split scaffold_disjoint \
  --unlabeled-ratio 1.0
```

Reported PU ROC-AUC and PU PR-AUC distinguish observed associations from sampled
unlabeled pairs. They are not clinical sensitivity, specificity, precision, or
probability calibration metrics.

## Reproducible four-model benchmark

The new benchmark compares ECFP logistic regression, graph-only GAT,
enzyme-only, and fused GAT models on **identical effective train, validation,
and test cohorts**. It trains neural models using balanced binary
cross-entropy on the *observation indicator*: reported association versus
sampled unlabeled pair. It does not assert that missing associations are true
negatives or that the resulting score estimates latent interaction risk.
The separate nnPU training command remains experimental: its class prior and
sampling assumptions require scientific justification.

```bash
python -m bioguard.main benchmark \
  --output data/benchmark_runs/scaffold_001 \
  --split scaffold_disjoint \
  --split-seed 42 \
  --seeds 17 42 73 \
  --epochs 20
```

The default source is the versioned event-preserving TWOSIDES loader. An
optional `--source-csv` accepts raw columns `Drug1_ID`, `Drug1`, `Y`,
`Drug2_ID`, and `Drug2`; it passes through the same relation preparation and
split rules. A local fixture run checks execution only and cannot establish
predictive quality. Do not substitute the old binary Parquets for the raw
relation source.

Runs use CPU with two Torch threads by default. Existing output directories
are refused. Each run contains:

- The original candidate cohort, common graph-eligible cohort, excluded pairs,
  per-split graph coverage, and enzyme-table coverage counts.
- Source/data/split/feature manifests, training configuration, actual Python
  package versions, and snapshots with hashes of the implementation files.
- Separate seed/model directories containing weights or the fitted baseline,
  training histories for neural models, and held-out prediction rows.
- Per-model ranking metrics, seed means/spreads, and a completion status.

Validation AP selects regularization and neural checkpoints. Training seeds
change model initialization/batches while the split seed stays fixed. Seed
spread is not a confidence interval. The enzyme ablation currently measures
the legacy table with zero fallback; it does not yet implement per-endpoint
missingness masks or a validated learned CYP predictor. A benchmark refuses
an undeclared installed CYP predictor rather than silently mixing features.

The benchmark creates its own graph cache and does not replace the bundled
model. Its checkpoints declare the `observed_association_ranking` task;
the older nnPU evaluator intentionally does not accept them. Inspect their
run-local predictions and results instead. The API can load an individual
benchmark checkpoint using `BG_ARTIFACT_DIR`, but deployment requires separate
model selection and validation.

## Molecular and cohort contracts

New graphs use `charge_parent_isomeric_v1`, the same normalization used by the
relation loader: RDKit charge parent, canonical isomeric SMILES, preservation
of stereochemistry, and no added tautomer canonicalization. Inputs are bounded
before conformer generation. The API passes the checkpoint's declared policy
to the worker and returns the processed structures and per-drug enzyme
coverage. A legacy checkpoint without a policy explicitly uses `legacy_raw_v1`
to preserve its prior behavior; it does not gain validated normalization by
being loaded with newer software.

New model bundles and graph caches record the featurizer/normalizer source
hashes and RDKit/Torch/PyG versions. Incompatible caches or bundles fail closed.
The standard cache is now `data/lmdb_cache_v1`; `BG_GRAPH_CACHE` can select
another directory. Build a fresh cache when its contract changes. Existing
legacy cache contents are not overwritten or automatically migrated.

Standalone `baselines` defaults to `--cohort graph`, requiring the versioned
graph cache and using its eligible pairs. `--cohort all` explicitly evaluates
all pairs. Schema-3 comparisons reject differences in **any** effective
train/validation/test cohort, including changed labels or structures, even
when the upstream source hash is identical. Older results must be regenerated.

## API and dashboard

```bash
docker compose up --build
```

or:

```bash
python -m bioguard.main serve
streamlit run dashboard.py
```

The API exposes:

- `/health` for process liveness
- `/ready` for model, feature-schema, and artifact readiness
- `/predict` for research-use pair scoring

The current bundled checkpoint returns an unvalidated `screening_score` and a
null `threshold_decision`. It does not return a clinically validated high/low
classification.

## Tests

```bash
python -m unittest discover -v
ruff check .
```

The scientific contract tests cover event preservation, duplicate handling,
input-order determinism, positive-versus-unlabeled semantics, pair leakage,
drug-disjoint splits, scaffold-disjoint splits, artifact compatibility, and
drug-order invariance.

## Major remaining research work

- Implement an event-conditioned multi-relational decoder.
- Add relation ontology/version mapping and directionality where supported.
- Benchmark multiple PU class-prior assumptions and ranking objectives.
- Add temporal and external validation sources with overlap audits.
- Replace the closed-world enzyme table with experimentally defined auxiliary
  tasks and a versioned inference artifact.
- Add uncertainty, applicability-domain checks, abstention, confidence
  intervals, repeated seeds, and locked confirmatory splits.
- Publish model cards, data sheets, prediction files, and complete environment
  locks for every released result.

## License

Repository code is MIT licensed. Dataset and derived-artifact redistribution
rights must be evaluated separately for each upstream source.
