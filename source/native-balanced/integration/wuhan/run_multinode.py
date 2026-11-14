"""Coordinate explicit SSH node workers; release only after every pre-gate passes."""
import argparse
import base64
import json
from pathlib import Path
import shlex
import subprocess
import time


def ssh(node, command, **kwargs):
    return subprocess.run(['ssh', '-n', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                           node['ssh'], command], stdin=subprocess.DEVNULL, **kwargs)


def ssh_probe(node, command, timeout=15):
    """Return a probe result, treating a transient SSH timeout as unknown.

    A node worker owns the experiment lock and its process group.  The
    coordinator must not turn one slow control-plane query into an abort while
    the worker is still healthy.  Callers decide whether an unknown result is
    retryable or fatal.
    """
    try:
        return ssh(node, command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None


def best_effort_abort(node, path):
    """Ask a node worker to retire without masking the original failure."""
    result = ssh_probe(node, shlex.join(['touch', path]), timeout=5)
    return result is not None and result.returncode == 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    spec = json.loads(a.manifest.read_text(encoding='utf-8'))
    nodes, name = spec['nodes'], spec['name']
    assert len(nodes) >= 2 and len({n['ssh'] for n in nodes}) == len(nodes)
    assert name and all(c.isalnum() or c in '-_' for c in name)
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out/'manifest.json').write_text(json.dumps(spec, indent=2))
    jobs = []
    try:
        for node in nodes:
            payload = base64.b64encode(json.dumps(node['command']).encode()).decode()
            command = shlex.join(['python3', node['project']+'/snapshots/multinode_node.py',
                '--project', node['project'], '--name', name, '--command-base64', payload,
                '--timeout', str(spec.get('timeout', 1800))])
            log = (a.out/(node['ssh']+'.log')).open('w')
            proc = subprocess.Popen(['ssh', '-n', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                node['ssh'], command], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            jobs.append((node, proc, log))
        deadline = time.monotonic()+100
        pending = list(nodes)
        while pending:
            for node in pending[:]:
                path = node['project']+'/evidence/'+name+'/ready.json'
                r = ssh_probe(node, shlex.join(['cat', path]))
                if r is not None and r.returncode == 0:
                    record = json.loads(r.stdout)
                    if not record['ready']:
                        raise RuntimeError('node pre-gate rejected: '+node['ssh'])
                    pending.remove(node)
            if pending:
                if time.monotonic() >= deadline or any(p.poll() is not None for _, p, _ in jobs):
                    raise RuntimeError('node worker failed or readiness timeout')
                time.sleep(2)
        print('ALL_NODE_PRE_GATES_PASS', len(nodes), flush=True)
        for node in nodes:
            path = node['project']+'/evidence/'+name+'/release'
            r = ssh_probe(node, shlex.join(['touch', path]))
            if r is None or r.returncode:
                raise RuntimeError('release failed: '+node['ssh'])
        results = []
        for node, proc, log in jobs:
            rc = proc.wait()
            log.close()
            r = ssh(node, shlex.join(['cat', node['project']+'/evidence/'+name+'/acceptance.json']),
                    capture_output=True, text=True, timeout=15)
            receipt = json.loads(r.stdout) if r.returncode == 0 else None
            results.append(dict(node=node['ssh'], ssh_exit=rc, receipt=receipt))
            print('NODE_EXIT', node['ssh'], rc, flush=True)
        passed = all(x['ssh_exit'] == 0 and x['receipt'] and x['receipt']['eligible'] for x in results)
        (a.out/'result.json').write_text(json.dumps(dict(pass_=passed, nodes=results), indent=2))
        if not passed:
            raise RuntimeError('distributed job failed; preserve all node evidence')
    except BaseException:
        for node, _, _ in jobs:
            path = node['project']+'/evidence/'+name+'/abort'
            best_effort_abort(node, path)
        # Do not infer a terminal remote state from a dropped SSH connection.
        # Remote workers own their timeout, process group and post-gate.
        raise


if __name__ == '__main__':
    main()
