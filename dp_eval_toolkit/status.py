"""Result statuses: what the runner records and what the analyzer counts.

Shared by every stage so that a status written by ``solve`` on a compute node
is spelled exactly the way ``analyze`` and ``report`` read it.
"""

from __future__ import annotations

SOLVED = 'SOLVED'              # k plans
EXHAUSTED = 'EXHAUSTED'        # fewer than k plans, and the planner says that is all within q
TIMEOUT = 'TIMEOUT'
MEMOUT = 'MEMOUT'
UNSOLVABLE = 'UNSOLVABLE'
UNSUPPORTED = 'UNSUPPORTED'
ERROR = 'ERROR'
KILLED = 'KILLED'              # the job was killed before the runner could report
MISSING = 'MISSING'            # an expected task that never produced a result file

STATUS_ORDER = [SOLVED, EXHAUSTED, TIMEOUT, MEMOUT, UNSOLVABLE, UNSUPPORTED, ERROR, KILLED, MISSING]
