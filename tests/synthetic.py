"""A synthetic sandbox: result files shaped exactly like the runner's, for
testing analyze/report without running any planner.

    python tests/synthetic.py /tmp/sandbox      # then dpevalcli report --sandbox-dir /tmp/sandbox
"""

from __future__ import annotations

import json
import os
import random
import sys

PLANNERS = ['FI', 'BP-SMT', 'BP-ASP']
K_VALUES = [5, 10, 100]
Q_VALUES = [1.0, 1.2]
MODEL = 'go-ru'
TIME_LIMIT = 1800


def label(k, q):
    return f'{MODEL}__k{k}__q{float(q)!r}'


def build(sandbox: str, instances: int = 30, seed: int = 7) -> dict:
    """Write the sandbox; return the ground truth the tests check against."""
    rng = random.Random(seed)
    tasks = [{'suite': 'classical-domains', 'domain': d, 'instance': f'p{i:02d}', 'track': 'classical',
              'ipc': None, 'domain_file': '/x/domain.pddl', 'problem_file': f'/x/p{i:02d}.pddl',
              'task_id': f'classical-domains:{d}:p{i:02d}'}
             for i in range(instances) for d in ['rovers' if i % 2 else 'logistics']]
    expected = []
    truth = {'statuses': {}}
    # Skill per planner: lower = solves more.
    hardness = {'FI': 0.8, 'BP-SMT': 1.6, 'BP-ASP': 1.1}
    for planner in PLANNERS:
        for k in K_VALUES:
            for q in Q_VALUES:
                for index, task in enumerate(tasks):
                    expected.append({'planner': planner, 'model': MODEL, 'k': k, 'q': q, **task})
                    slug = task['task_id'].replace(':', '_')
                    out_dir = os.path.join(sandbox, 'results', planner, label(k, q))
                    os.makedirs(out_dir, exist_ok=True)
                    # The last task is never run by BP-ASP (MISSING), and one is killed.
                    if planner == 'BP-ASP' and index == len(tasks) - 1:
                        truth['statuses'][(planner, k, q, task['task_id'])] = 'MISSING'
                        continue
                    difficulty = (index + 1) / instances * hardness[planner] * (1 + k / 200) * q
                    seconds = 0.3 * (10 ** (4 * difficulty))
                    if planner == 'BP-SMT' and index == 3 and k == 5 and q == 1.0:
                        with open(os.path.join(out_dir, f'{slug}.running'), 'w') as handle:
                            json.dump(_payload(planner, task, k, q, 'X', [], 0), handle)
                        truth['statuses'][(planner, k, q, task['task_id'])] = 'KILLED'
                        continue
                    if seconds < TIME_LIMIT:
                        status, count = 'SOLVED', k
                        if index % 7 == 0 and q == 1.0:
                            status, count = 'EXHAUSTED', max(1, k // 3)
                    else:
                        status, seconds = 'TIMEOUT', TIME_LIMIT
                        count = rng.randint(0, k - 1)
                    bcov = count if planner != 'FI' else max(1, int(count * rng.uniform(0.2, 0.7)))
                    bcov = min(bcov, count)
                    payload = _payload(planner, task, k, q, status, list(range(count)), seconds, bcov)
                    if planner == 'FI' and index == 1 and k == 5 and q == 1.2:
                        payload['metrics']['invalid-plans'] = 1          # an invalid plan set
                    with open(os.path.join(out_dir, f'{slug}.json'), 'w') as handle:
                        json.dump(payload, handle)
                    truth['statuses'][(planner, k, q, task['task_id'])] = status
    os.makedirs(sandbox, exist_ok=True)
    with open(os.path.join(sandbox, 'tasks.json'), 'w') as handle:
        json.dump({'expected': expected, 'limits': {'time-seconds': TIME_LIMIT, 'memory-mb': 8192},
                   'tasks': tasks}, handle)
    return truth


def _payload(planner, task, k, q, status, plans, seconds, bcov=0):
    n = len(plans)
    costs = [10 + (i % 3) for i in range(n)]
    return {
        'task': {'task-id': task['task_id'], 'suite': task['suite'], 'domain': task['domain'],
                 'instance': task['instance'], 'track': task['track'], 'ipc': None},
        'diversity': {'k': k, 'q': q, 'model': MODEL, 'features': [['go', {}], ['ru', 'x']]},
        'planner': {'tag': planner, 'adapter': planner.lower()},
        'status': status,
        'timings': {'planning-seconds': seconds, 'total-seconds': seconds + 1},
        'plans': [['(a)'] for _ in plans],
        'metrics': {'plan-count': n, 'valid-plans': n, 'invalid-plans': 0, 'peak-memory-mb': 100},
        'diversity-scores': {'b-coverage': bcov, 'b-maxsum': bcov * (bcov - 1) / 4,
                             'b-maxsum-normalised': 0.5 if n > 1 else None, 'stability': 0.4,
                             'uniqueness': 0.9, 'distinct-plans': n, 'costs': costs, 'errors': []},
        'info': {'native-b-coverage': bcov} if planner == 'BP-ASP' else {},
    }


if __name__ == '__main__':
    build(sys.argv[1])
    print(f'synthetic sandbox written to {sys.argv[1]}')
