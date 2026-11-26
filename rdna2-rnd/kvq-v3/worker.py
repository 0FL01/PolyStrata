#!/usr/bin/env python3
"""Minimal strata-poly --serve driver for KVQ V3 research (private). Production args/env from the live config."""
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

CONFIG = Path.home() / '.config/strata/inkling2-mini/strata-inkling-2-mini.json'
KVQ = Path.home() / '.cache/opencode-tmp/opencode/polystrata-runs/kvq-v3'
RESEARCH_BIN = KVQ / 'src/build/strata-poly'
SAMPLING = 'temperature=1 top_p=1 top_k=0 min_p=0'
GREEDY = 'temperature=0'


def setarg(args, key, value=None):
    if key in args:
        i = args.index(key)
        if value is None:
            return
        args[i + 1] = str(value)
    else:
        args.extend([key] if value is None else [key, str(value)])


def dropflag(args, key, has_value=True):
    if key in args:
        i = args.index(key)
        del args[i:i + (2 if has_value else 1)]


def read_ids(path):
    return [int(x) for x in Path(path).read_text().strip().split(',') if x]


def csv(ids):
    return ','.join(str(int(i)) for i in ids)


class Worker:
    """One engine process. `home` is a private writable dir (profile copy, XDG dirs, logs)."""

    def __init__(self, home, binary=RESEARCH_BIN, env=None, stable=True, cap_mib=None, extra_args=(),
                 max_context=200000, profile=None, swaps=None):
        cfg = json.loads(CONFIG.read_text())
        self.home = Path(home)
        self.home.mkdir(parents=True, exist_ok=True)
        args = list(cfg['args'])
        prof = self.home / 'profile'
        src = Path(profile) if profile else Path(args[args.index('--expert-profile') + 1])
        if not prof.exists():
            shutil.copyfile(src, prof)
        setarg(args, '--expert-profile', prof)
        setarg(args, '--max-context', max_context)
        if swaps is not None:
            setarg(args, '--expert-swaps', swaps)
        setarg(args, '--window', 1)
        setarg(args, '--lookup', 0)
        args += ['--no-draft', '--eos-ids', '200006', '--serve']
        e = {k: v for k, v in os.environ.items() if not k.startswith(('STRATA_', 'DF2_', 'RECON_'))}
        e.update(cfg.get('env', {}))
        libs = ':'.join(cfg.get('lib_dirs', []))
        e['LD_LIBRARY_PATH'] = libs + (':' + e['LD_LIBRARY_PATH'] if e.get('LD_LIBRARY_PATH') else '')
        if stable:
            args += ['--cpu-share', '0', '--stage-ahead', '0']
            e['STRATA_INKLING_PREFETCH'] = '0'
        if cap_mib:
            e['STRATA_INKLING_EXPERT_KV_MIB'] = str(cap_mib)
        for k in ('HOME', 'XDG_CACHE_HOME', 'XDG_CONFIG_HOME', 'XDG_STATE_HOME'):
            pass
        e.update({'HOME': str(self.home), 'XDG_CACHE_HOME': str(self.home / '.cache'),
                  'XDG_CONFIG_HOME': str(self.home / '.config'), 'XDG_STATE_HOME': str(self.home / '.state')})
        if env:
            e.update({k: str(v) for k, v in env.items()})
        args += list(extra_args)
        self.cmd = [str(binary), *map(str, args)]
        (self.home / 'cmd.json').write_text(json.dumps(dict(cmd=self.cmd, env={k: v for k, v in e.items()
                                                       if k.startswith(('STRATA_', 'HIP', 'LD_'))}), indent=1))
        self.err = open(self.home / 'engine.err', 'ab')
        t0 = time.time()
        self.p = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err,
                                  env=e, cwd=cfg.get('cwd'), text=True, bufsize=1)
        self.info = []
        while True:
            line = self._line()
            self.info.append(line)
            if line.startswith('READY'):
                break
        self.startup_s = time.time() - t0
        self.log = open(self.home / 'wire.log', 'a')

    def _line(self):
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError(f'engine exited (rc={self.p.poll()}); see {self.home / "engine.err"}')
        return line.rstrip('\n')

    def send(self, s):
        self.p.stdin.write(s + '\n')
        self.p.stdin.flush()

    def gen(self, ids, max_new, sampling=GREEDY, seed=None, extra='', turn_at=None):
        keys = sampling + (f' seed={seed}' if seed is not None else '') + (f' turn_at={turn_at}' if turn_at else '')
        keys += (' ' + extra) if extra else ''
        t0 = time.time()
        self.send(f'GEN {max_new} {keys} {csv(ids)}')
        out, meta = [], dict(pp=[])
        while True:
            line = self._line()
            if line.startswith('T '):
                out.append(int(line[2:]))
            elif line.startswith('RESUME'):
                meta['resume'] = int(line.split()[1])
            elif line.startswith('PP '):
                meta['pp'].append(line)
            elif line.startswith('DONE'):
                f = line.split()
                meta.update(produced=int(f[1]), prompt=int(f[2]), prompt_ms=float(f[3]), decode_ms=float(f[4]),
                            finish=f[5], hits=int(f[9]), lookups=int(f[10]), read=int(f[14]), done=line)
                break
            elif line.startswith('ERR') or line.startswith('FATAL'):
                raise RuntimeError(line)
        meta['wall_s'] = time.time() - t0
        self.log.write(json.dumps(dict(req=len(ids), max_new=max_new, keys=keys, out=len(out), **{k: v for k, v in meta.items() if k != 'pp'})) + '\n')
        self.log.flush()
        return out, meta

    def session(self, verb, path):
        t0 = time.time()
        self.send(f'{verb} {path}')
        while True:
            line = self._line()
            if line.startswith(('SAVED', 'RESTORED')):
                f = line.split()
                return dict(pos=int(f[1]), bytes=int(f[2]), ms=float(f[3]), wall_s=time.time() - t0)
            if line.startswith(('SERR', 'FATAL', 'ERR')):
                raise RuntimeError(line)

    def save(self, path):
        return self.session('SAVE', path)

    def restore(self, path):
        return self.session('RESTORE', path)

    def close(self):
        try:
            self.send('QUIT')
            self.p.stdin.close()
            self.p.wait(timeout=120)
        except Exception:
            self.p.kill()
        self.err.close()
        return self.p.returncode
