# Diverse-Planners-EvalToolkit — benchmark harness for diverse planners

Runs diverse planners over the classical benchmarks, **one slurm array index per
diversity task**, each under its own time and memory limit on the partition and
QOS you choose, and turns the results into coverage/timeout tables,
behaviour-diversity tables over the commonly solved instances, and survival
plots.

A **diversity task** is

| input | meaning |
|---|---|
| planning problem | a (domain, problem) pair from the benchmark repositories |
| `k` | the number of plans asked for |
| `q` | the quality bound: plans may cost up to `q` × the optimal cost (`1.0` = optimal plans only) |
| diversity model | the features the plans should be diverse over (`go`, `ru`, …) |

The structure follows [pyPMTEvalToolkit](https://github.com/pyPMT/pyPMTEvalToolkit):
same stages, same experiment-directory layout, same slurm job arrays and
in-process limits, but the unit of work is (planner, instance, k, q, model)
rather than (planner, instance).

Planners benchmarked out of the box:

| adapter | planner | source |
|---|---|---|
| `fi` | Forbid-Iterative top-quality planner (`extended_unordered_topq`), the baseline of the behaviour-planning paper experiments; with `"select"` it keeps the `k` most diverse of the plans it finds (B-Coverage or B-MaxSum, via BehaviourDiversityCounter) | [`forbiditerative`](https://pypi.org/project/forbiditerative/) |
| `bp-smt` | Behaviour planning over SMT (`ForbiddenBehaviorSMTPlanner`) | [pyBehaviourPlanningSMT](https://github.com/MFaisalZaki/pyBehaviourPlanningSMT) |
| `bp-asp` | Behaviour planning over ASP (`BCoveragePlanner` / `BMaxSumPlanner`) | [pyBehaviourPlanningASP](https://github.com/MFaisalZaki/pyBehaviourPlanningASP) |
| `up-anytime` | any Unified Planning engine with the `AnytimePlanner` interface (first `k` distinct plans) | — |
| `package.module:Class` | your own adapter (see [Adding a planner](#adding-a-planner)) | — |

## Quick start

```bash
./setup_benchmark.sh
```

needs Python 3.10 to 3.13 (it picks the newest one on `PATH`; `--python
python3.13` chooses explicitly, and 3.14 is refused because the dependencies
do not support it yet). It asks which configuration to run (`exp-configurations/default` or `demo`), where
everything should live, the per-task time and memory limit, the slurm
partition, QOS and account, the numbers of plans `k` and the quality bounds
`q`. It then creates a virtualenv, installs the toolkit with the planners the
configuration names, clones
[classical-domains](https://github.com/AI-Planning/classical-domains),
generates the slurm job arrays and prints the `sbatch` command. When the
planners' dependency pins cannot be satisfied in one environment (BP-SMT and
BP-ASP currently disagree on `up-fast-downward`), each planner is installed
into its own `venv-<planner>` and its configuration is pointed there;
`--separate-venvs` does that from the start. Every prompt has
a flag:

```bash
./setup_benchmark.sh --config default --work-dir /scratch/$USER/dp-sweep \
    --time-limit 30m --memory-limit 8GB --partition compute --qos normal \
    --k-values "5 10 100 1000" --quality-bounds "1.0 1.1 1.2 2.0" --yes
```

then

```bash
bash /scratch/$USER/dp-sweep/sandbox/default/slurm/submit_all.sh
```

and once the jobs are done:

```bash
dpevalcli analyze --sandbox-dir /scratch/$USER/dp-sweep/sandbox/default --per-domain
dpevalcli report  --sandbox-dir /scratch/$USER/dp-sweep/sandbox/default
```

## The stages

```
dpevalcli check     → can every planner of the experiment be imported here?
dpevalcli discover  → which tasks a benchmark repository holds
dpevalcli generate  → one solve command per diversity task, plus slurm arrays
dpevalcli solve     → run ONE diversity task under its limits, dump a JSON result   (slurm calls this)
dpevalcli analyze   → CSVs + text summary: coverage, timeouts, diversity on common solved
dpevalcli report    → LaTeX tables and figures (survival plots, …)
```

`generate`, `analyze` and `report` need nothing but the standard library
(`report`'s figures want matplotlib), so a sweep can be generated and
analysed on a laptop; only the compute nodes need the planners.

## Experiment configuration

```
experiment/
├── exp-details.json        limits, slurm, the diversity grid, task selection
└── planners/
    ├── fi-bcoverage.json           FI, k plans picked from its pool by B-Coverage
    ├── fi-bmaxsum.json             FI, picked by B-MaxSum
    ├── bp-asp-bcoverage.json       ASP behaviour planner optimising B-Coverage
    └── bp-asp-bmaxsum.json         ... optimising B-MaxSum
```

```jsonc
// exp-details.json
{
    "cfgs": {
        "timelimit": "00:30:00",            // per diversity task; also "30m" or 1800
        "memorylimit": "8GB",
        "slurm-time-headroom": "00:10:00",  // slurm gets limit + headroom, so the runner's
        "slurm-memory-headroom": "2GB",     // own limit fires first and records TIMEOUT/MEMOUT
        "validate-plans": true,
        "slurm": {
            "partition": "compute",
            "qos": "normal",
            "account": null,
            "cpus-per-task": 1,
            "max-parallel-jobs": 100,       // --array=...%100
            "max-array-size": 1000,         // longer arrays are split
            "extra-directives": []          // verbatim #SBATCH lines
        }
    },
    "diversity": {
        "k-values": [5, 10, 100, 1000],
        "quality-bounds": [1.0, 1.1, 1.2, 2.0],
        "models": {                         // name -> features
            "go":    [["go", {}]],
            "go-ru": [["go", {}], ["ru", {"resources": "builtin", "optional": true}]]
        }
    },
    "tasks": {
        "tracks": ["classical"],
        "max-instances-per-domain": 0,      // 0 = all
        "selection": "even",
        "include-domains": [], "exclude-domains": [], "ipc-years": [], "selected-tasks": [],
        "selected-instances": "builtin"     // the paper's instance list (below)
    }
}
```

`selected-instances` restricts a classical-domains style repository to a list
of `(ipc-year, domain, n)` references in the convention of the
behaviour-planning paper experiments
([`classical_instances`](https://github.com/MFaisalZaki/pyBehaviourPlanningSMT/blob/paper-refactor/paper_experiments/apptainer-generate-benchmark-slurm-tasks.py)):
`domain` and `ipc-year` are the `name` and `ipc` of a domain's `api.py`, and
`n` is the 1-based position of the instance once that `api.py`'s problems are
sorted by path. `"builtin"` is that very list (1076 instances, shipped as
`dp_eval_toolkit/data/classical-instances.txt`); a file path (one reference
per line, `#` comments) or an inline JSON list works too, as does
`--selected-instances FILE|builtin` on `discover` and `generate`. Where
classical-domains holds several directories for one (year, name) -- the `opt`
and `sat` tracks of an IPC, `strips` beside `full` or `adl` encodings -- the
paper's generator took whichever its directory walk met first; the toolkit
fixes the choice: `adl` directories are skipped, then `strips` is preferred,
then `opt` over `sat`, then the alphabetically first directory. The default
configuration uses the builtin list; `demo` does not.

Every (model, k, q) combination is crossed with every instance and every
planner: 4 k × 4 q × 2 models × N instances × 3 planners commands.

```jsonc
// planners/bp-asp.json
{
    "planner-tag": "BP-ASP",                 // names the results directory and the table rows
    "adapter": "bp-asp",
    "planner-params": {"indicator": "B-Coverage"},   // reaches the adapter verbatim
    "tracks": ["classical"],                 // optional: restrict to some tracks
    "models": ["go-ru"],                     // optional: restrict to some diversity models
    "venv-dir": "/path/to/venv"              // optional: run this planner from its own environment
}
```

`venv-dir` (or `conda-env`, or `apptainer-image`) overrides `generate`'s
`--venv-dir` for that one planner. That matters here: pyBehaviourPlanningSMT
pins `numpy<2`, which can break the scipy that other planners pull in, so FI
and BP-SMT are often happier in separate environments.

### Diversity-model features

Features are the dimensions of
[BehaviourDiversityCounter](https://github.com/MFaisalZaki/BehaviourDiversityCounter),
which scores every plan set (see below), named as the planners name them:

| key | feature | config | dissimilarity |
|---|---|---|---|
| `go` | goal ordering: the order in which the goal atoms are first achieved | `{}` (or `{"max-goals": m}`) | Hamming distance over the orderings / number of goals |
| `cb` | cost: the sum of the plan's action costs (its length under unit cost) | `{}` | \|c − c'\| / max(c, c') |
| `cbin` | cost bin: the bin of cost / optimal cost in a grid over [1, q] | `{"optimal-cost": c, "q": 1.2, "width": 0.1}` | bin distance / (bins − 1) |
| `ru` | resource usage: how many of the listed resource objects the plan uses (the planners' and the paper's definition; the counter's `rc` dimension) | resources (below) | \|a − b\| / (a + b) |
| `ru-set` | which of the listed resource objects the plan uses (the counter's `ru` dimension) | resources (below) | Jaccard distance over the used sets |
| `fn` | numeric function values: the bin each listed fluent ends in | functions file | mean bin distance / (bins − 1) over the fluents |
| `stability` | the plan's action set | `{}` | Jaccard distance over the action sets |

`ru` needs a list of resource objects per instance. `"builtin"` uses the data
shipped in `dp_eval_toolkit/data/ru-info` (from the behaviour-planning SMT paper
experiments: rovers, logistics, depot, driverlog, barman, floortile, childsnack,
assembly, agricola). A directory works too: either the same JSON files, or
`<dir>/<domain>/<instance>.txt`, `<dir>/<domain>/resources.txt` or
`<dir>/<domain>.txt` holding `(:resource NAME MIN MAX DELTA)` lines. `{"objects": [...]}`
lists the names inline. Names are matched to the task's objects case-insensitively,
with `-` and `_` interchangeable; names that match no object are dropped and noted. `generate` resolves the data once per instance and
writes it under `sandbox/feature-data/`. If an instance has no data,
`"optional": true` drops the feature for that instance; without it, the
instance is left out of that model's runs and listed under `not-applicable` in
`tasks.json`.

### How each planner gets k, q and the model

* **FI** takes `k` and `q` (`--number-of-plans`, `--quality-bound`) and ignores
  the model while searching. Its top-quality search usually leaves more than
  `k` plans; with `"select": "B-Coverage"` (or `B-MaxSum`, `B-MaxMin`,
  `B-Novelty`) the adapter keeps the `k` most diverse of them under the task's
  model, as the paper's `fi-bc` baseline did; without it, the first `k`.
  `"pool-size"` (a count, or `"3x"` for a multiple of `k`) asks FI for a larger
  pool to choose from; the shipped configurations leave it at `k`, as in the
  paper. The plans are then scored like everyone else's (see below). It is
  launched through a small shim that keeps its working files in a folder this
  run owns, so on a timeout the plans it has found are still recovered, and
  the folder is removed afterwards. `local-dir` (default `/tmp/planner_runs`)
  moves that folder, for example to a node-local `$TMPDIR`.
* **BP-SMT** gets the model as its behaviour-space dimensions. Its quality bound
  lives in the `cb` dimension, so when the model has no `cb` and `q ≠ 1.0`, one
  is added for the planner (and the addition logged). This changes what the
  planner is told, not what is measured. Every z3 check gets the remaining
  budget as its timeout.
* **BP-ASP** gets the model as feature specs and `q` as its `quality_bound`. It
  gets the remaining budget as its own timeout, so on a timeout it returns the
  plans it holds.

## What a run records

`results/<planner>/<model>__k<k>__q<q>/<instance>.json` holds the task, the
limits, the planning time (parse + plan, which is what the survival plots use),
peak memory, every plan as text, plan validation, the planner's own log and
info (optimal length, cost bound, …), and the **diversity scores**:

| status | meaning |
|---|---|
| `SOLVED` | `k` plans returned |
| `EXHAUSTED` | fewer than `k`, and the planner terminated saying that is every plan within `q` |
| `TIMEOUT` / `MEMOUT` | the task's own limit fired; **the plans found so far are kept and scored** |
| `UNSOLVABLE` | no plan at all |
| `UNSUPPORTED` | the planner cannot handle this task or model |
| `ERROR` | it crashed; traceback in `errors/` |
| `KILLED` | the scheduler killed the job before it could report |
| `MISSING` | the task never produced a result |

**Solved** (for coverage) means `SOLVED` or `EXHAUSTED`, with at least one plan
and no plan failing validation. `--strict-k` on `analyze`/`report` counts
`SOLVED` only.

### Scoring diversity: one evaluator for every planner

Every plan set is re-scored by
[BehaviourDiversityCounter](https://github.com/MFaisalZaki/BehaviourDiversityCounter)
(through `dp_eval_toolkit.behaviours`), which replays each plan on the original
problem and computes its behaviour under the task's diversity model. The
planners' self-reported behaviours cannot serve as the yardstick: FI has none,
and the two behaviour planners each compute theirs from their own encodings.
Those self-reports are still kept in the result, and `analyze` reports how often
a planner's own B-Coverage agrees with the evaluator's.

Per plan set, each indicator over the *distinct* behaviours:

* **B-Coverage**: the number of distinct behaviours;
* **B-MaxSum**: the sum over pairs of distinct behaviours of the model's
  dissimilarity (the mean of the features' dissimilarities);
* **B-MaxSum per pair**: B-MaxSum divided by the number of plan pairs, so it
  can be compared across `k`;
* **B-MaxMin**: the smallest pairwise dissimilarity;
* **B-Novelty**: the mean dissimilarity of each behaviour to its nearest
  neighbours;
* **stability** (mean pairwise Jaccard distance of action sets),
  **uniqueness** (fraction of plan pairs that are not reorderings of each
  other), distinct plans, plan costs.

A plan that cannot be replayed on the problem is listed under the result's
`errors` and left out of the indicators.

## Analysis

`dpevalcli analyze --sandbox-dir sandbox` writes to `sandbox/analysis/`:

| file | what |
|---|---|
| `coverage.csv` | per (model, k, q, planner) over **all** attempted instances: solved, coverage, `SOLVED` vs `EXHAUSTED`, timeouts, **timeouts that still returned plans** and the mean number of plans they returned, memouts, errors, unsupported, killed, missing, invalid plan sets, mean/median time of the solved ones |
| `coverage-per-track.csv` | the same per track |
| `diversity-common-solved.csv` | per (model, k, q, planner), over the instances **every** planner of the group solved: mean/median B-Coverage, B-Coverage / k, B-MaxSum, B-MaxSum per pair, B-MaxMin, B-Novelty, stability, uniqueness, plan count, cost, and how often the planner alone had the best B-Coverage (and how often it tied) |
| `per-domain.csv` | coverage, timeouts and mean B-Coverage per domain |
| `quality-bound-violations.csv` | solved plan sets whose costliest plan exceeds q × the cheapest plan any planner found for that instance |
| `results.csv` | one row per diversity task |
| `summary.txt`, `summary.json` | all of the above, plus the list of commonly solved instances per group |

## Paper tables and figures

`dpevalcli report --sandbox-dir sandbox` writes to `sandbox/report/`:

| file | what |
|---|---|
| `coverage.tex` | planners × (q, k) coverage grid, one table per diversity model |
| `timeouts.tex` | the same grid for timeouts (with how many returned plans) |
| `diversity.tex` | B-Coverage, B-MaxSum, B-MaxSum/pair, stability on the commonly solved instances per (k, q); the best B-Coverage in bold |
| `plots/survival/<model>__k<k>__q<q>.*` | **survival plot**: instances solved (y) within time t (x, log scale), a step curve per planner |
| `plots/survival-grid-<model>.*` | the same as small multiples, rows k, columns q |
| `plots/coverage-vs-k-<model>.*` | coverage as k grows, a panel per q |
| `plots/bcoverage-vs-k-<model>.*` | mean B-Coverage / k on the commonly solved instances as k grows |
| `plots/bcoverage-distribution-<model>.*` | per-instance B-Coverage / k on the commonly solved instances, a box (with its mean) per planner and (k, q) |
| `plots/outcomes-<model>.*` | stacked outcome bars (solved / exhausted / invalid / timeout / …) per (k, q) and planner |

Figures are PDF (fonts embedded as Type 42) and PNG. Each planner keeps one
colour, marker and line style across every figure, so they also read in
greyscale. `--no-plots` writes only the tables, without needing matplotlib.

## Running

```bash
bash sandbox/slurm/submit_all.sh       # one job array per planner (chunked to max-array-size)
bash sandbox/run_local.sh 8            # or locally, 8 at a time, no scheduler
```

Each array index runs one line of `sandbox/cmds/<planner>.txt`. To resume
after a partial sweep, or after adding a planner or widening the grid:

```bash
dpevalcli generate --exp-dir experiment --sandbox-dir sandbox \
    --tasks-dir classical-domains=benchmark-tasks/classical-domains --venv-dir venv --skip-existing
```

`tasks.json` keeps the full list of expected tasks, so a resumed sweep's
denominators do not shrink.

## Adding a planner

Write an adapter: a `DiversePlannerAdapter` subclass whose `plan(ctx)` returns
a `DiverseRun` of UP plans over `ctx.problem`.

```python
from dp_eval_toolkit.adapters.base import DiversePlannerAdapter, DiverseRun, COMPLETE, EXHAUSTED

class MyPlanner(DiversePlannerAdapter):
    name = 'mine'
    supported_features = ('go', 'ru')          # None = ignores the model

    def plan(self, ctx):
        # ctx.problem, ctx.domain_file, ctx.problem_file, ctx.k, ctx.q,
        # ctx.features [(key, config)], ctx.remaining() seconds, ctx.workdir
        planner = ...
        ctx.snapshot = lambda: planner.plans_so_far    # recovered on timeout
        plans = planner.plan(ctx.k, ctx.q, timeout=ctx.remaining(margin=10))
        return DiverseRun(plans=plans, outcome=COMPLETE if len(plans) >= ctx.k else EXHAUSTED)
```

and name it in a planner configuration: `"adapter": "my_package.my_module:MyPlanner"`.
Numeric (or other) tracks need no change to the harness: drop `tracks`
from the planner configuration, add the track to `exp-details.json`, and give
the model features that make sense there (for example `fn`).

## Tests

```bash
python3.13 -m venv venv && source venv/bin/activate    # Python 3.10-3.13
pip install -e ".[dev,plots]"
pytest
```

The tests cover configuration, resource lookup, command and slurm generation,
resuming, the analyzer (against a synthetic sandbox with known outcomes:
`tests/synthetic.py`), the report, and the behaviour evaluator against
hand-computed B-Coverage and B-MaxSum values.
