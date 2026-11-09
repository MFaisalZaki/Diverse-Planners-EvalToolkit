#!/usr/bin/env bash
#
# One-shot setup for a diverse-planning evaluation sweep:
#
#   1. ask for the per-task time and memory limits, the slurm partition/QOS,
#      the numbers of plans k and the quality bounds q,
#   2. create a virtualenv and install this toolkit plus the planners the
#      experiment's planner configurations name (fi, bp-smt, bp-asp), plus any
#      extra python packages passed with --packages,
#   3. fetch the benchmark repositories for the tracks you picked,
#   4. write the experiment configuration with those settings,
#   5. generate one run command per diversity task -- (planner, instance, k, q,
#      diversity model) -- and the slurm job arrays.
#
# Everything is prompted with a default, and every prompt has a matching flag,
# so the same script drives an interactive setup and a scripted one (--yes).
# Re-running it is safe: an existing venv, clone or experiment is reused.
#
# Usage:
#   ./setup_benchmark.sh                      # interactive
#   ./setup_benchmark.sh --yes                # all defaults, no prompts
#   ./setup_benchmark.sh --time-limit 30m --memory-limit 8GB --partition compute --qos normal \
#                        --k-values "5 10 100" --quality-bounds "1.0 1.1 1.2" --yes
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR"

# ---------------------------------------------------------------- defaults --
WORK_DIR="${REPO_DIR}/benchmark-run"
VENV_DIR=""
TASKS_DIR=""
SANDBOX_DIR=""
EXP_DIR=""
CONFIG_ROOT="${SCRIPT_DIR}/exp-configurations"
CONFIG=""                      # name under exp-configurations/, or a path
DEFAULT_CONFIG="default"
TIME_LIMIT=""                  # default: the configuration's own
MEMORY_LIMIT=""
MAX_INSTANCES=""                # default: the configuration's own (0 = all)
TRACKS="classical"
K_VALUES=""
Q_VALUES=""
PARTITION=""
ACCOUNT=""
QOS=""
MAX_PARALLEL="50"
EXTRAS=""
PACKAGES=""
SKIP_FETCH="no"
SKIP_INSTALL="no"
SEPARATE_VENVS="no"            # one virtualenv per planner, even when one would do
LIST_CONFIGS="no"
ASSUME_YES="no"
# The interpreter the venv is built with. unified-planning and the
# behaviour-diversity counter support Python 3.10 to 3.13; left blank, the
# newest supported one on PATH is used.
PYTHON_BIN="${PYTHON_BIN:-}"
PYTHON_MIN="3.10"
PYTHON_MAX_EXCLUSIVE="3.14"

CLASSICAL_REPO="https://github.com/AI-Planning/classical-domains.git"
NUMERIC_REPO="https://github.com/pyPMT/numeric-domains.git"
# The IPC archive is where the temporal benchmarks live: every `*-time*` and
# `*-temporal*` domain from IPC-2002 onwards, in one uniform layout. There is no
# equivalent of classical-domains/numeric-domains for the temporal track.
TEMPORAL_REPO="https://github.com/potassco/pddl-instances.git"
CLI_NAME="dpevalcli"

# "3.12" for an interpreter, or nothing when it cannot be run.
python_version() {
    "$1" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true
}

# Is a "major.minor" version within [PYTHON_MIN, PYTHON_MAX_EXCLUSIVE)?
python_supported() {
    [ -n "$1" ] || return 1
    printf '%s\n%s\n%s\n' "$PYTHON_MIN" "$1" "$PYTHON_MAX_EXCLUSIVE" | sort -V -c 2>/dev/null \
        && [ "$1" != "$PYTHON_MAX_EXCLUSIVE" ]
}

# Choose the interpreter for the venv: the one given with --python/PYTHON_BIN,
# checked, or else the newest supported python3.X on PATH.
pick_python() {
    if [ -n "$PYTHON_BIN" ]; then
        command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "python interpreter not found: ${PYTHON_BIN}"
        local version
        version="$(python_version "$PYTHON_BIN")"
        python_supported "$version" || die "${PYTHON_BIN} is Python ${version:-?}; the toolkit's dependencies need ${PYTHON_MIN} <= Python < ${PYTHON_MAX_EXCLUSIVE} (try --python python3.13)"
        return
    fi
    local candidate
    for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
        command -v "$candidate" >/dev/null 2>&1 || continue
        if python_supported "$(python_version "$candidate")"; then
            PYTHON_BIN="$candidate"
            return
        fi
    done
    die "no Python ${PYTHON_MIN}-3.13 interpreter on PATH (python3 is $(python_version python3)); install one and pass --python python3.13"
}

usage() {
    cat <<'EOF'
One-shot setup for a diverse-planning evaluation sweep.

Usage:
  ./setup_benchmark.sh                      # interactive; asks which configuration to run
  ./setup_benchmark.sh --yes                # all defaults, no prompts
  ./setup_benchmark.sh --config demo --time-limit 30m --partition compute --qos normal --yes

Options:
  --config NAME|DIR       which exp-configurations/ entry to run; a bare name is
                          looked up under exp-configurations/, a path is used as
                          given. Asked interactively when omitted (default: default)
  --list-configs          print the available configurations and exit
  --work-dir DIR          root for venv/tasks/sandbox   (default: <repo>/benchmark-run)
  --venv-dir DIR          virtualenv location           (default: <work-dir>/venv)
  --tasks-dir DIR         where benchmarks are cloned   (default: <work-dir>/benchmark-tasks)
  --sandbox-dir DIR       commands, results, logs       (default: <work-dir>/sandbox/<config>)
  --exp-dir DIR           the experiment to run, seeded from the configuration
                          (default: <work-dir>/experiments/<config>)
  --time-limit VALUE      per task, e.g. 00:30:00 or 30m
  --memory-limit VALUE    per task, e.g. 8GB
  --max-instances N       instances per domain, 0 for all (default: the configuration's own)
  --tracks "a b c"        subset of: classical numeric temporal (default: classical)
  --k-values "5 10 100"   numbers of plans (default: the configuration's)
  --quality-bounds "1.0 1.2"  quality bounds q >= 1.0 (default: the configuration's)
  --partition NAME        slurm partition
  --account NAME          slurm account
  --qos NAME              slurm QOS
  --max-parallel N        cap on concurrently running array jobs
  --extras "a,b"          extras to install on top of the ones the planner
                          configurations imply (e.g. "plots")
  --packages "a b"        extra python packages to pip install into the venv,
                          space or comma separated. Anything pip accepts works:
                          PyPI names ("forbiditerative==1.0"), and a local
                          checkout given as a directory is installed editable
                          (-e), so a planner under development stays live
                          (e.g. "~/dev/pyBehaviourPlanningASP")
  --skip-fetch            do not clone/update the benchmark repositories
  --skip-install          do not create the venv or install anything
  --separate-venvs        install every planner into its own virtualenv
                          (<work-dir>/venv-<planner>) and point its configuration
                          at it; done automatically when the planners' dependency
                          pins cannot be satisfied in one environment
  --python BIN            interpreter for the venv (default: the newest python3.10-3.13
                          on PATH; python3.14+ is not supported by the dependencies)
  -y, --yes               accept every default, never prompt
  -h, --help              this message
EOF
}

# ------------------------------------------------------------------- args ---
while [ $# -gt 0 ]; do
    case "$1" in
        --work-dir)         WORK_DIR="$2"; shift 2 ;;
        --venv-dir)         VENV_DIR="$2"; shift 2 ;;
        --tasks-dir)        TASKS_DIR="$2"; shift 2 ;;
        --sandbox-dir)      SANDBOX_DIR="$2"; shift 2 ;;
        --exp-dir)          EXP_DIR="$2"; shift 2 ;;
        --config)           CONFIG="$2"; shift 2 ;;
        --template-dir)     CONFIG="$2"; shift 2 ;;      # the older name for --config
        --list-configs)     LIST_CONFIGS="yes"; shift ;;
        --time-limit)       TIME_LIMIT="$2"; shift 2 ;;
        --memory-limit)     MEMORY_LIMIT="$2"; shift 2 ;;
        --max-instances)    MAX_INSTANCES="$2"; shift 2 ;;
        --tracks)           TRACKS="$2"; shift 2 ;;
        --k-values)         K_VALUES="$2"; shift 2 ;;
        --quality-bounds)   Q_VALUES="$2"; shift 2 ;;
        --partition)        PARTITION="$2"; shift 2 ;;
        --account)          ACCOUNT="$2"; shift 2 ;;
        --qos)              QOS="$2"; shift 2 ;;
        --max-parallel)     MAX_PARALLEL="$2"; shift 2 ;;
        --extras)           EXTRAS="$2"; shift 2 ;;
        --packages)         PACKAGES="$2"; shift 2 ;;
        --skip-fetch)       SKIP_FETCH="yes"; shift ;;
        --skip-install)     SKIP_INSTALL="yes"; shift ;;
        --separate-venvs)   SEPARATE_VENVS="yes"; shift ;;
        --python)           PYTHON_BIN="$2"; shift 2 ;;
        -y|--yes)           ASSUME_YES="yes"; shift ;;
        -h|--help)          usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

# VENV_DIR and TASKS_DIR default under WORK_DIR, which is itself prompted, so
# they are resolved after the prompts; SANDBOX_DIR and EXP_DIR additionally
# default per configuration, so they are resolved once that is known (below).

# ---------------------------------------------------------------- helpers ---
say()  { printf '\033[1m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33m warning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31m error:\033[0m %s\n' "$*" >&2; exit 1; }

# ask <prompt> <default> <variable-name>
ask() {
    local prompt="$1" default="$2" varname="$3" answer=""
    if [ "$ASSUME_YES" = "yes" ] || [ ! -t 0 ]; then
        answer="$default"
    else
        read -r -p "$(printf '%s [%s]: ' "$prompt" "$default")" answer || answer=""
        answer="${answer:-$default}"
    fi
    printf -v "$varname" '%s' "$answer"
}

has_track() {
    case " $TRACKS " in *" $1 "*) return 0 ;; *) return 1 ;; esac
}

# ------------------------------------------------------------ configurations --
# A configuration is a directory holding exp-details.json and planners/*.json --
# the same shape the toolkit runs. The ones shipped in exp-configurations/ are
# offered by name; any other directory works too, given as a path.

# Names of the shipped configurations, one per line.
config_names() {
    local dir
    for dir in "${CONFIG_ROOT}"/*/; do
        [ -f "${dir}exp-details.json" ] || continue
        basename "${dir%/}"
    done
}

# "3 planner(s): bp-asp, bp-smt, fi" for a configuration directory.
config_summary() {
    local dir="$1" count names
    count=$(find "${dir}/planners" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
    names=$(ls "${dir}/planners" 2>/dev/null | sed 's/\.json$//' | paste -sd, - | sed 's/,/, /g')
    printf '%s planner(s)%s' "$count" "${names:+: $names}"
}

# A name resolves under exp-configurations/, anything with a slash is a path.
config_dir() {
    case "$1" in
        */*|.|..) printf '%s' "$1" ;;
        *)        printf '%s' "${CONFIG_ROOT}/$1" ;;
    esac
}

list_configs() {
    local name
    echo "Configurations in ${CONFIG_ROOT}:"
    for name in $(config_names); do
        printf '  %-14s %s\n' "$name" "$(config_summary "${CONFIG_ROOT}/${name}")"
    done
}

# Ask which configuration to run and set CONFIG. Accepts its number, its name,
# or a path. Sets the global rather than printing it: the menu goes to stdout,
# so a $(...) capture would swallow it.
choose_config() {
    # `reply`, not `answer`: `ask` has a local of that name, and dynamic scoping
    # would have it assign to its own copy instead of this one.
    local names name count reply index=1
    names="$(config_names)"
    count=$(printf '%s\n' "$names" | sed '/^$/d' | wc -l | tr -d ' ')
    [ "$count" -gt 0 ] || die "no configuration in ${CONFIG_ROOT}"

    echo "Available experiment configurations:"
    for name in $names; do
        printf '  %d) %-14s %s\n' "$index" "$name" "$(config_summary "${CONFIG_ROOT}/${name}")"
        index=$((index + 1))
    done
    echo "  (or the path to a configuration directory of your own)"
    echo

    ask "Experiment configuration" "$DEFAULT_CONFIG" reply
    # A number picks from the list; anything else is a name or a path.
    case "$reply" in
        ''|*[!0-9]*) CONFIG="$reply" ;;
        *)           CONFIG="$(printf '%s\n' "$names" | sed -n "${reply}p")"
                     [ -n "$CONFIG" ] || die "no configuration number ${reply}" ;;
    esac
}

clone_or_update() {
    local url="$1" dest="$2"
    if [ -d "${dest}/.git" ]; then
        say "updating $(basename "$dest")"
        git -C "$dest" fetch --depth 1 origin HEAD --quiet || warn "could not update ${dest}"
        git -C "$dest" reset --hard FETCH_HEAD --quiet || warn "could not fast-forward ${dest}"
    elif [ -d "$dest" ]; then
        say "reusing $(basename "$dest") (not a git checkout)"
    else
        say "cloning $(basename "$dest")"
        git clone --depth 1 --quiet "$url" "$dest"
    fi
}

if [ "$LIST_CONFIGS" = "yes" ]; then
    list_configs
    exit 0
fi

# --------------------------------------------------------------- prompting --
if [ "$ASSUME_YES" != "yes" ] && [ -t 0 ]; then
    echo
    echo "Diverse planners evaluation setup -- press enter to accept a default."
    echo
fi

# The configuration comes first: it decides which planners run, and therefore
# which planners have to be installed. Asked only when --config was not given;
# --yes and a non-interactive shell take the default.
if [ -z "$CONFIG" ]; then
    if [ "$ASSUME_YES" = "yes" ] || [ ! -t 0 ]; then
        CONFIG="$DEFAULT_CONFIG"
    else
        choose_config
    fi
fi

TEMPLATE_DIR="$(config_dir "$CONFIG")"
CONFIG_NAME="$(basename "${TEMPLATE_DIR%/}")"
[ -d "$TEMPLATE_DIR" ] || die "no such configuration: ${CONFIG} (tried ${TEMPLATE_DIR}); \
run with --list-configs to see the shipped ones"
[ -f "${TEMPLATE_DIR}/exp-details.json" ] || \
    die "${TEMPLATE_DIR} holds no exp-details.json, so it is not a configuration"

# Everything the experiment produces -- venv, benchmark tasks, run commands,
# results, logs -- lives under this root unless a --*-dir flag says otherwise.
ask "Experiment root (venv, tasks, commands, results)" "$WORK_DIR" WORK_DIR
VENV_DIR="${VENV_DIR:-${WORK_DIR}/venv}"
TASKS_DIR="${TASKS_DIR:-${WORK_DIR}/benchmark-tasks}"

# One experiment and one sandbox per configuration: seeding a second
# configuration into a directory that already holds another would union their
# planner sets and mix their results.
SANDBOX_DIR="${SANDBOX_DIR:-${WORK_DIR}/sandbox/${CONFIG_NAME}}"
EXP_DIR="${EXP_DIR:-${WORK_DIR}/experiments/${CONFIG_NAME}}"

# The interpreter is needed from here on: to read the configuration, to work
# out which planners to install and to build the venv.
pick_python

# Limits default to the chosen configuration's own values: cfg_value SECTION KEY FALLBACK.
cfg_value() {
    "$PYTHON_BIN" -c 'import json,sys; v = json.load(open(sys.argv[1])).get(sys.argv[2], {}).get(sys.argv[3]); print(sys.argv[4] if v is None else v)' \
        "${TEMPLATE_DIR}/exp-details.json" "$1" "$2" "$3"
}
TIME_LIMIT="${TIME_LIMIT:-$(cfg_value cfgs timelimit 00:30:00)}"
MEMORY_LIMIT="${MEMORY_LIMIT:-$(cfg_value cfgs memorylimit 8GB)}"
MAX_INSTANCES="${MAX_INSTANCES:-$(cfg_value tasks max-instances-per-domain 0)}"
ask "Per-task time limit (HH:MM:SS or 30m)"      "$TIME_LIMIT"    TIME_LIMIT
ask "Per-task memory limit (e.g. 8GB)"           "$MEMORY_LIMIT"  MEMORY_LIMIT
ask "Tracks to benchmark"                        "$TRACKS"        TRACKS
# k and q default to the configuration's own grid; blank keeps it.
ask "Numbers of plans k (blank = configuration's)" "$K_VALUES"   K_VALUES
ask "Quality bounds q (blank = configuration's)"   "$Q_VALUES"   Q_VALUES
ask "Instances per domain (0 = all)"             "$MAX_INSTANCES" MAX_INSTANCES
ask "Slurm partition (blank = site default)"     "$PARTITION"     PARTITION
ask "Slurm account (blank = site default)"       "$ACCOUNT"       ACCOUNT
ask "Slurm QOS (blank = site default)"           "$QOS"           QOS
ask "Max array jobs running at once"             "$MAX_PARALLEL"  MAX_PARALLEL
ask "Extra python packages for the venv (blank = none)" "$PACKAGES" PACKAGES

# Commas and spaces both separate packages, so a requirements-style
# "a,b" and a shell-style "a b" both work.
PACKAGE_LIST="$(printf '%s' "$PACKAGES" | tr ',' ' ')"

# An entry that is a directory is a local checkout under development, so it is
# installed editable; everything else (PyPI names, pins, VCS URLs) goes to pip
# as given. An explicit -e is honoured either way.
PACKAGE_ARGS=()
NEXT_IS_EDITABLE="no"
for pkg in $PACKAGE_LIST; do
    case "$pkg" in
        -e|--editable) NEXT_IS_EDITABLE="yes"; continue ;;
        # A quoted --packages "~/dev/x" keeps its tilde from the shell.
        "~/"*) pkg="${HOME}/${pkg#\~/}" ;;
    esac
    if [ "$NEXT_IS_EDITABLE" = "yes" ] || [ -d "$pkg" ]; then
        PACKAGE_ARGS+=(-e "$pkg")
    else
        PACKAGE_ARGS+=("$pkg")
    fi
    NEXT_IS_EDITABLE="no"
done

for track in $TRACKS; do
    case "$track" in
        classical|numeric|temporal) ;;
        *) die "unknown track '${track}' (expected: classical, numeric, temporal)" ;;
    esac
done

echo
say "config       ${CONFIG_NAME} (${TEMPLATE_DIR})"
say "             $(config_summary "$TEMPLATE_DIR")"
say "work dir     ${WORK_DIR}"
say "venv         ${VENV_DIR}"
say "python       ${PYTHON_BIN} ($(python_version "$PYTHON_BIN"))"
say "tasks        ${TASKS_DIR}"
say "sandbox      ${SANDBOX_DIR}"
say "experiment   ${EXP_DIR}"
say "limits       ${TIME_LIMIT} / ${MEMORY_LIMIT} per task"
say "tracks       ${TRACKS}"
say "k values     ${K_VALUES:-from configuration}"
say "q values     ${Q_VALUES:-from configuration}"
say "slurm        partition=${PARTITION:-default} qos=${QOS:-default} account=${ACCOUNT:-default}"
[ ${#PACKAGE_ARGS[@]} -gt 0 ] && say "packages     ${PACKAGE_ARGS[*]}"
echo

mkdir -p "$WORK_DIR" "$TASKS_DIR"

# ------------------------------------------------------------- experiment --
# Set up before installing: the planner configurations decide which planners have
# to be installed, and every one of them is benchmarked.
mkdir -p "${EXP_DIR}/planners"
if [ ! -f "${EXP_DIR}/exp-details.json" ]; then
    say "creating experiment at ${EXP_DIR} from configuration ${CONFIG_NAME}"
    cp "${TEMPLATE_DIR}/exp-details.json" "${EXP_DIR}/"
else
    say "reusing experiment at ${EXP_DIR}"
    # Which configuration an experiment came from, so that pointing a second one
    # at it -- which would add that one's planners to this one's -- is noticed.
    if [ -f "${EXP_DIR}/.configuration" ] && \
       [ "$(cat "${EXP_DIR}/.configuration")" != "$CONFIG_NAME" ]; then
        warn "${EXP_DIR} was created from configuration '$(cat "${EXP_DIR}/.configuration")';"
        warn "seeding '${CONFIG_NAME}' into it adds its planners to the ones already there."
    fi
fi
printf '%s\n' "$CONFIG_NAME" > "${EXP_DIR}/.configuration"

# Copy in any template configuration the experiment does not have yet, without
# touching the ones it does: a config added to the template after the experiment
# was first created would otherwise never reach it, while local edits survive.
for template in "${TEMPLATE_DIR}"/planners/*.json; do
    [ -e "$template" ] || continue
    name="$(basename "$template")"
    if [ ! -f "${EXP_DIR}/planners/${name}" ]; then
        cp "$template" "${EXP_DIR}/planners/"
        say "added planner configuration ${name}"
    fi
done

PLANNER_COUNT=$(find "${EXP_DIR}/planners" -maxdepth 1 -name '*.json' | wc -l | tr -d ' ')
if [ "$PLANNER_COUNT" -eq 0 ]; then
    die "no planner configuration in ${EXP_DIR}/planners"
fi
say "planners (${PLANNER_COUNT}): $(ls "${EXP_DIR}/planners" | tr '\n' ' ')"

# --------------------------------------------------------------- install ---
# Which extras to install is read off the planner configurations: an engine that
# a configuration names should be installed, and one nobody asks for should not
# drag its (large, native) dependencies into the venv.
# Redirected to a file rather than captured with $(...): a heredoc inside a
# command substitution is mis-parsed by the bash 3.2 macOS still ships.
EXTRAS_FILE="$(mktemp "${TMPDIR:-/tmp}/dpeval-extras.XXXXXX")"
trap 'rm -f "$EXTRAS_FILE"' EXIT

"$PYTHON_BIN" - "${EXP_DIR}/planners" > "$EXTRAS_FILE" <<'PY'
import json, os, sys

# adapter -> extra of this package's pyproject
EXTRA_FOR = {'fi': 'fi', 'bp-smt': 'bp-smt', 'bp-asp': 'bp-asp'}

extras, unknown = {'plots'}, set()
directory = sys.argv[1]
for name in sorted(os.listdir(directory)):
    if not name.endswith('.json'):
        continue
    with open(os.path.join(directory, name)) as handle:
        cfg = json.load(handle)
    # A planner with its own environment is installed there, not here.
    if cfg.get('venv-dir') or cfg.get('conda-env') or cfg.get('apptainer-image'):
        continue
    adapter = str(cfg.get('adapter') or '')
    if adapter in EXTRA_FOR:
        extras.add(EXTRA_FOR[adapter])
    elif adapter:
        unknown.add(adapter)

if unknown:
    print('note: no extra known for adapter(s): ' + ', '.join(sorted(unknown)) +
          ' -- install their planners yourself', file=sys.stderr)
print(','.join(sorted(extras)))
PY

IMPLIED_EXTRAS="$(cat "$EXTRAS_FILE")"
ALL_EXTRAS="$(printf '%s,%s' "$IMPLIED_EXTRAS" "$EXTRAS" | tr ',' '\n' | sed '/^$/d' | sort -u | paste -sd, -)"

# Create a virtualenv with the chosen interpreter, or check that an existing one
# was built with a supported Python.
# Returns non-zero (after a warning) when the venv cannot be made usable.
ensure_venv() {
    local venv="$1" version
    if [ ! -d "$venv" ]; then
        say "creating virtualenv at ${venv} with ${PYTHON_BIN} ($(python_version "$PYTHON_BIN"))"
        "$PYTHON_BIN" -m venv "$venv" || { warn "could not create a virtualenv at ${venv}"; return 1; }
    else
        version="$(python_version "${venv}/bin/python")"
        if ! python_supported "$version"; then
            warn "${venv} was built with Python ${version:-?}, which the dependencies do not support; remove it (rm -rf ${venv}) for a fresh one"
            return 1
        fi
        say "reusing virtualenv at ${venv} (Python ${version})"
    fi
    # Only pip: builds run in isolated environments, and a planner may pin its
    # own setuptools (pyBehaviourPlanningSMT does).
    "${venv}/bin/python" -m pip install --quiet --upgrade pip || warn "could not upgrade pip in ${venv}; carrying on with the one it has"
    return 0
}

# Packages that could not be installed: "what|why", reported again at the end.
INSTALL_FAILURES=()
note_install_failure() {
    INSTALL_FAILURES+=("$1|$2")
    warn "$1: $2"
}

# pip install the toolkit (with the given extras) into a virtualenv; pip's output
# goes to a log that is named only when the install fails.
install_toolkit() {
    local venv="$1" extras="$2" log target
    target="${REPO_DIR}"
    [ -n "$extras" ] && target="${REPO_DIR}[${extras}]"
    log="$(mktemp "${TMPDIR:-/tmp}/dpeval-pip.XXXXXX")"
    if "${venv}/bin/python" -m pip install --quiet -e "$target" > "$log" 2>&1; then
        rm -f "$log"
        return 0
    fi
    PIP_LOG="$log"
    return 1
}

# The pyproject extra and the module that proves the planner is installed.
planner_probe() {
    case "$1" in
        fi)     echo forbiditerative ;;
        bp-smt) echo behaviour_planning_smt.fbi.planner ;;
        bp-asp) echo behaviour_planning_asp ;;
        *)      echo "" ;;
    esac
}

# Give every planner configuration of the experiment that uses adapter $1 (and
# names no environment of its own) the virtualenv $2.
assign_planner_venv() {
    "$PYTHON_BIN" - "${EXP_DIR}/planners" "$1" "$2" <<'PY'
import json, os, sys
directory, adapter, venv = sys.argv[1:4]
for name in sorted(os.listdir(directory)):
    if not name.endswith('.json'):
        continue
    path = os.path.join(directory, name)
    with open(path) as handle:
        cfg = json.load(handle)
    if cfg.get('adapter') != adapter or any(cfg.get(k) for k in ('venv-dir', 'conda-env', 'apptainer-image')):
        continue
    cfg['venv-dir'] = venv
    with open(path, 'w') as handle:
        json.dump(cfg, handle, indent=4)
        handle.write('\n')
    print(f'  {name}: venv-dir = {venv}')
PY
}

# One virtualenv per planner: the toolkit alone in the main venv, each planner's
# extra in <work-dir>/venv-<planner>, and the planner configurations pointed there.
# A planner that cannot be installed is reported, not fatal: the rest of the
# sweep is still set up, and the summary at the end says what is missing.
install_separately() {
    local base_extras planner_extras extra planner_venv probe
    base_extras="$(printf '%s,%s' "plots" "$EXTRAS" | tr ',' '\n' | sed '/^$/d' | sort -u | paste -sd, -)"
    say "installing the toolkit into ${VENV_DIR} (extras: ${base_extras})"
    install_toolkit "$VENV_DIR" "$base_extras" || die "could not install the toolkit; pip's output is in ${PIP_LOG}"
    planner_extras="$(printf '%s' "$IMPLIED_EXTRAS" | tr ',' '\n' | grep -v '^plots$' || true)"
    for extra in $planner_extras; do
        planner_venv="${WORK_DIR}/venv-${extra}"
        if ! ensure_venv "$planner_venv"; then
            note_install_failure "planner ${extra}" "no usable virtualenv at ${planner_venv}"
            continue
        fi
        say "installing planner ${extra} into ${planner_venv}"
        if ! install_toolkit "$planner_venv" "$extra"; then
            note_install_failure "planner ${extra}" "pip could not install it into ${planner_venv}; pip's output is in ${PIP_LOG}"
            continue
        fi
        probe="$(planner_probe "$extra")"
        if [ -n "$probe" ] && ! "${planner_venv}/bin/python" -c "import ${probe}" 2>/dev/null; then
            note_install_failure "planner ${extra}" "installed into ${planner_venv} but 'import ${probe}' fails there"
        fi
        assign_planner_venv "$extra" "$planner_venv"
    done
}

if [ "$SKIP_INSTALL" = "yes" ]; then
    say "skipping installation (--skip-install)"
    [ ${#PACKAGE_ARGS[@]} -gt 0 ] && warn "--skip-install also skips the extra packages: ${PACKAGE_ARGS[*]}"
    [ -x "${VENV_DIR}/bin/${CLI_NAME}" ] || warn "no ${CLI_NAME} in ${VENV_DIR}; generation will fail"
else
    ensure_venv "$VENV_DIR" || die "no usable virtualenv at ${VENV_DIR} (pass --venv-dir for another location)"
    if [ "$SEPARATE_VENVS" = "yes" ]; then
        install_separately
    else
        say "installing the toolkit with extras: ${ALL_EXTRAS:-none}"
        if ! install_toolkit "$VENV_DIR" "$ALL_EXTRAS"; then
            if grep -q 'ResolutionImpossible' "$PIP_LOG"; then
                warn "the planners' dependency pins cannot all be satisfied in one environment (pip's output: ${PIP_LOG})"
            else
                warn "installing everything into one virtualenv failed (pip's output: ${PIP_LOG})"
            fi
            say "installing each planner into its own virtualenv instead, so one failure does not take the others down"
            install_separately
        fi
    fi
    if [ ${#PACKAGE_ARGS[@]} -gt 0 ]; then
        say "installing extra packages: ${PACKAGE_ARGS[*]}"
        PIP_LOG="$(mktemp "${TMPDIR:-/tmp}/dpeval-pip.XXXXXX")"
        if "${VENV_DIR}/bin/python" -m pip install --quiet "${PACKAGE_ARGS[@]}" > "$PIP_LOG" 2>&1; then
            rm -f "$PIP_LOG"
        else
            note_install_failure "extra packages (${PACKAGE_ARGS[*]})" "pip could not install them into ${VENV_DIR}; pip's output is in ${PIP_LOG}"
        fi
    fi
fi

CLI="${VENV_DIR}/bin/${CLI_NAME}"
VENV_PYTHON="${VENV_DIR}/bin/python"
[ -x "$CLI" ] || die "${CLI_NAME} was not installed into ${VENV_DIR}"

# Fail here rather than on ten thousand compute nodes each discovering the same
# missing planner. A planner with its own venv-dir is probed in that venv.
say "checking that every configured planner is importable"
"$CLI" check --exp-dir "$EXP_DIR" || warn "some planners are not importable from the environment their configuration names (see above)"

# ------------------------------------------------------------- benchmarks --
TASKS_ARGS=()
SUITE_TRACK_ARGS=()
if [ "$SKIP_FETCH" != "yes" ]; then
    if has_track classical; then clone_or_update "$CLASSICAL_REPO" "${TASKS_DIR}/classical-domains"; fi
    if has_track numeric;   then clone_or_update "$NUMERIC_REPO"   "${TASKS_DIR}/numeric-domains"; fi
    if has_track temporal;  then clone_or_update "$TEMPORAL_REPO"  "${TASKS_DIR}/pddl-instances"; fi
fi

if has_track classical && [ -d "${TASKS_DIR}/classical-domains" ]; then
    TASKS_ARGS+=(--tasks-dir "classical-domains=${TASKS_DIR}/classical-domains")
fi
if has_track numeric && [ -d "${TASKS_DIR}/numeric-domains" ]; then
    TASKS_ARGS+=(--tasks-dir "numeric-domains=${TASKS_DIR}/numeric-domains")
fi
if has_track temporal && [ -d "${TASKS_DIR}/pddl-instances" ]; then
    TASKS_ARGS+=(--tasks-dir "pddl-instances=${TASKS_DIR}/pddl-instances")
    # The IPC archive holds all three tracks; take only its temporal domains so
    # its classical and numeric ones do not duplicate the two repos above.
    SUITE_TRACK_ARGS+=(--suite-tracks "pddl-instances=temporal")
fi
if [ ${#TASKS_ARGS[@]} -eq 0 ]; then
    die "no benchmark repository available for tracks: ${TRACKS}"
fi

# ------------------------------------------------------- experiment limits --
say "writing the limits into ${EXP_DIR}/exp-details.json"
DPEVAL_TIME="$TIME_LIMIT" DPEVAL_MEM="$MEMORY_LIMIT" DPEVAL_TRACKS="$TRACKS" \
DPEVAL_K="$K_VALUES" DPEVAL_Q="$Q_VALUES" \
DPEVAL_MAXINST="$MAX_INSTANCES" DPEVAL_PARTITION="$PARTITION" \
DPEVAL_ACCOUNT="$ACCOUNT" DPEVAL_QOS="$QOS" DPEVAL_PARALLEL="$MAX_PARALLEL" \
"$VENV_PYTHON" - "$EXP_DIR" <<'PY'
import json, os, sys

exp_dir = sys.argv[1]
path = os.path.join(exp_dir, 'exp-details.json')
with open(path) as handle:
    details = json.load(handle)

cfgs = details.setdefault('cfgs', {})
cfgs['timelimit'] = os.environ['DPEVAL_TIME']
cfgs['memorylimit'] = os.environ['DPEVAL_MEM']
slurm = cfgs.setdefault('slurm', {})
slurm['partition'] = os.environ['DPEVAL_PARTITION'] or None
slurm['account'] = os.environ['DPEVAL_ACCOUNT'] or None
slurm['qos'] = os.environ['DPEVAL_QOS'] or None
slurm['max-parallel-jobs'] = int(os.environ['DPEVAL_PARALLEL'] or 0)

diversity = details.setdefault('diversity', {})
if os.environ['DPEVAL_K'].strip():
    diversity['k-values'] = [int(v) for v in os.environ['DPEVAL_K'].replace(',', ' ').split()]
if os.environ['DPEVAL_Q'].strip():
    diversity['quality-bounds'] = [float(v) for v in os.environ['DPEVAL_Q'].replace(',', ' ').split()]

tasks = details.setdefault('tasks', {})
tasks['tracks'] = os.environ['DPEVAL_TRACKS'].split()
tasks['max-instances-per-domain'] = int(os.environ['DPEVAL_MAXINST'] or 0)
details['name'] = os.path.basename(os.path.abspath(exp_dir))

with open(path, 'w') as handle:
    json.dump(details, handle, indent=4)
    handle.write('\n')
print(f'  time={cfgs["timelimit"]} memory={cfgs["memorylimit"]} '
      f'k={diversity.get("k-values")} q={diversity.get("quality-bounds")} '
      f'tracks={tasks["tracks"]} instances/domain={tasks["max-instances-per-domain"]}')
PY

# --------------------------------------------------------------- generate --
GENERATE_ARGS=(
    generate
    --exp-dir "$EXP_DIR"
    --sandbox-dir "$SANDBOX_DIR"
    --venv-dir "$VENV_DIR"
    "${TASKS_ARGS[@]}"
)
if [ ${#SUITE_TRACK_ARGS[@]} -gt 0 ]; then
    GENERATE_ARGS+=("${SUITE_TRACK_ARGS[@]}")
fi

echo
say "generating run commands"
"$CLI" "${GENERATE_ARGS[@]}"

if [ ${#INSTALL_FAILURES[@]} -gt 0 ]; then
    echo
    warn "${#INSTALL_FAILURES[@]} package(s) could not be installed; the sweep is generated without them working:"
    for failure in "${INSTALL_FAILURES[@]}"; do
        printf '   - %s\n       %s\n' "${failure%%|*}" "${failure#*|}" >&2
    done
    printf '   A planner that is not installed records every one of its tasks as ERROR. Either install it\n' >&2
    printf '   by hand into the virtualenv named above (pip install -e "%s[<planner>]"), or delete its\n' "$REPO_DIR" >&2
    printf '   file under %s/planners/ and re-run this script.\n' "$EXP_DIR" >&2
fi

cat <<EOF

Next steps
  submit the sweep      bash ${SANDBOX_DIR}/slurm/submit_all.sh
  or run it locally     bash ${SANDBOX_DIR}/run_local.sh 8
  watch it              squeue -u \$USER
  coverage + diversity  ${CLI} analyze --sandbox-dir ${SANDBOX_DIR} --per-domain
  paper tables/figures  ${CLI} report  --sandbox-dir ${SANDBOX_DIR}

  re-generate after changing the experiment (skips finished tasks):
    ${CLI} generate --exp-dir ${EXP_DIR} --sandbox-dir ${SANDBOX_DIR} \\
        --venv-dir ${VENV_DIR} ${TASKS_ARGS[*]} --skip-existing
EOF
