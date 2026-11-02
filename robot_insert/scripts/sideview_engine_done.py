#!/usr/bin/env python3
"""Wait for the official engine's zero exit and complete scoring before teardown."""
import json
import errno
from pathlib import Path
import re
import sys
import time

import yaml

RETRYABLE_READ_ERRORS = {errno.ENODATA, errno.EAGAIN, errno.EINTR,
                         errno.ENOENT, errno.ESTALE, errno.EBUSY, errno.ETIMEDOUT}


def process_alive(pid):
    """Check only a PID supplied by this session; zombies have already exited."""
    if pid is None:
        return True
    try:
        stat = Path(f'/proc/{pid}/stat').read_text()
        return stat.rsplit(')', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def completed(output):
    output = Path(output)
    log = (output/'eval.log').read_text(errors='replace')
    if re.search(r'\[aic_engine-\d+\]: process has died', log):
        raise RuntimeError('Official engine exited with an error')
    if not re.search(r'\[aic_engine-\d+\]: process has finished cleanly', log):
        return False
    scoring = output/'scoring/scoring.yaml'
    if not scoring.exists():
        return False
    scores = yaml.safe_load(scoring.read_text())
    if not isinstance(scores, dict):
        return False  # A just-created score file may still be empty.
    expected = set(json.loads((output/'settings.json').read_text())['trial_ids'])
    return set(scores)-{'total'} == expected and all(isinstance(scores[name], dict) for name in expected)


def wait_for_completion(output, *, timeout=1800., read_error_timeout=30.,
                        evaluator_pid=None, model_pid=None, clock=time.monotonic,
                        sleep=time.sleep):
    """Retry transient DrvFS reads, while retaining strict engine/score checks.

    ENODATA can be returned when reading an actively written file through the
    D: bind mount. A single read failure is not an engine failure. Persistent
    errors, real engine failures, and vanished processes must still fail closed.
    """
    if timeout <= 0 or read_error_timeout <= 0:
        raise ValueError('Monitor timeouts must be positive')
    output = Path(output)
    started = clock()
    error_started = None
    retries = 0
    last_warning = float('-inf')
    while True:
        now = clock()
        if now-started >= timeout:
            raise TimeoutError(f'Official engine did not finish within {timeout:g} seconds; inspect {output}')
        try:
            done = completed(output)
            if done:
                report = {'engine_exit_zero': True, 'all_trials_scored': True,
                    'transient_read_retries': retries,
                    'teardown': 'SIGKILL only after official engine exits; avoids existing Gazebo destructor abort/WSL dump'}
                proof = output/'official-engine-complete.json'
                temporary = proof.with_suffix('.json.partial')
                temporary.write_text(json.dumps(report, indent=2)+'\n')
                temporary.replace(proof)
                return report
            error_started = None
        except (OSError, json.JSONDecodeError, yaml.YAMLError) as error:
            if isinstance(error, OSError) and error.errno not in RETRYABLE_READ_ERRORS:
                raise
            now = clock()
            if error_started is None:
                error_started = now
            retries += 1
            if now-error_started >= read_error_timeout:
                raise TimeoutError(f'Cannot read collection status for {read_error_timeout:g} seconds: {error}') from error
            if now-last_warning >= 5:
                print(f'Status read temporarily unavailable; retrying: {error}', file=sys.stderr, flush=True)
                last_warning = now
        for name, pid in (('evaluator', evaluator_pid), ('model', model_pid)):
            if not process_alive(pid):
                raise RuntimeError(f'{name} process {pid} exited before official completion was verified')
        sleep(.25)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--timeout-seconds', type=float, default=1800)
    parser.add_argument('--read-error-seconds', type=float, default=30)
    parser.add_argument('--evaluator-pid', type=int)
    parser.add_argument('--model-pid', type=int)
    args = parser.parse_args()
    if any(pid is not None and pid <= 0 for pid in (args.evaluator_pid, args.model_pid)):
        parser.error('Process IDs must be positive')
    wait_for_completion(args.output, timeout=args.timeout_seconds,
        read_error_timeout=args.read_error_seconds, evaluator_pid=args.evaluator_pid, model_pid=args.model_pid)


if __name__=='__main__':
    main()
