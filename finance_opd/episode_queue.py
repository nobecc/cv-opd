"""Run native research on a network-capable CPU host, keeping models on GPUs.

The shared-filesystem queue carries requests/status, never substitute token
probabilities. The worker writes the same authoritative turns.jsonl.gz as local
EpisodeProxy; the optimizer reads those sampled IDs, masks and logprobs verbatim.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import fcntl
import gzip
import json
import os
from pathlib import Path
import socket
import sys
import time
from urllib.parse import urlsplit, urlunsplit


def atomic(path: Path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))
    temporary.chmod(0o600)
    temporary.replace(path)


def reachable_student_url(url: str, teacher_url: str) -> str:
    parsed = urlsplit(url)
    if parsed.hostname not in ('127.0.0.1', 'localhost', '0.0.0.0', '::1'):
        return url
    teacher = urlsplit(teacher_url)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.connect((teacher.hostname, teacher.port or 80))
        host = probe.getsockname()[0]
    return urlunsplit((parsed.scheme, f'{host}:{parsed.port}', parsed.path, '', ''))


async def queued_episode(config, task, student_url, directory, session_id, sampling_params):
    queue = Path(config['runtime']['episode_worker_queue'])
    job = queue / session_id
    job.mkdir(parents=True, exist_ok=False, mode=0o700)
    request = dict(config=config, task=task, student_url=reachable_student_url(student_url, config['teacher']['url']),
                   directory=str(directory), session_id=session_id, sampling_params=sampling_params)
    atomic(job / 'heartbeat.json', {'time': time.time()})
    atomic(job / 'request.json', request)
    deadline = min(config['job']['deadline_epoch'], time.time()+config['runtime']['max_runtime_seconds']+300)
    try:
        while time.time() < deadline:
            atomic(job / 'heartbeat.json', {'time': time.time()})
            if (job / 'error.json').exists():
                raise RuntimeError('CPU episode worker: '+json.loads((job/'error.json').read_text())['error'])
            if (job / 'result.json').exists():
                result = json.loads((job / 'result.json').read_text())
                with gzip.open(directory / 'turns.jsonl.gz', 'rt') as stream:
                    records = [json.loads(line) for line in stream]
                return result['status'], records
            await asyncio.sleep(2)
        raise TimeoutError('CPU episode worker deadline reached')
    finally:
        if not (job / 'result.json').exists():
            atomic(job / 'cancel.json', {'time': time.time()})


class EpisodeWorker:
    def __init__(self, config, run, execute=None):
        self.config, self.run = config, Path(run).resolve()
        self.concurrency = config['runtime'].get('episode_worker_concurrency', 2)
        self.memory_headroom_mib = config['runtime'].get('episode_worker_memory_headroom_mib', 0)
        if type(self.concurrency) is not int or not 1 <= self.concurrency <= 8:
            raise ValueError('episode worker concurrency must be an integer in [1,8]')
        if type(self.memory_headroom_mib) is not int or self.memory_headroom_mib < 0:
            raise ValueError('episode worker memory headroom must be nonnegative MiB')
        self.queue = Path(config['runtime']['episode_worker_queue']).resolve()
        if not self.queue.is_relative_to(self.run):
            raise ValueError('episode queue must belong to this run')
        self.tasks = {}
        for path in [config['dataset'], str(self.run/'holdout.jsonl')]:
            if not Path(path).exists(): continue
            for line in Path(path).read_text().splitlines():
                row=json.loads(line); self.tasks[row['metadata']['id']]={**row['metadata'], 'task':row['task']}
        self.execute = execute or self.native_episode
        self.tokenizer = None

    def validate(self, request):
        cfg=request['config']; directory=Path(request['directory']).resolve()
        if not directory.is_relative_to(self.run) or directory.name != request['session_id']:
            raise ValueError('episode output escapes owned run')
        if (directory/'turns.jsonl.gz').exists():
            raise ValueError('refusing to overwrite original sampled tokens')
        for key in ('student_model','pi_finance','safactory','slime'):
            if cfg[key] != self.config[key]: raise ValueError('worker identity mismatch: '+key)
        for key in ('context_length','tool_protocol','thinking'):
            if cfg['runtime'][key] != self.config['runtime'][key]: raise ValueError('runtime contract mismatch: '+key)
        if cfg['teacher']['model'] != self.config['teacher']['model']:
            raise ValueError('teacher identity mismatch')
        for key in ('references_path','evaluator_config'):
            if cfg['continuation'].get(key) != self.config['continuation'].get(key):
                raise ValueError('private evaluator mismatch')
        if request['task'] != self.tasks.get(request['task'].get('id')):
            raise ValueError('task differs from owned original dataset')
        parsed=urlsplit(request['student_url'])
        if parsed.scheme != 'http' or not parsed.hostname or parsed.username or parsed.password or not parsed.port:
            raise ValueError('student must be an explicit internal model endpoint')
        return directory

    async def native_episode(self, request):
        from .config import add_dependency_paths
        from .episode import EpisodeProxy, run_native
        config=copy.deepcopy(request['config'])
        config['runtime']['pi_python']=sys.executable
        # The CPU host owns tools/egress; never inherit a GPU container's Python.
        config['runtime']['node_bin']=(self.config.get('continuation',{}).get('node_bin')
                                       or self.config['runtime'].get('node_bin')
                                       or str(Path(__file__).resolve().parents[1]/'.runtime/node24'))
        add_dependency_paths(config)
        from rl.examples.pi_finance.qwen_mask_builder import QwenFinanceMaskBuilder
        if self.tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer=AutoTokenizer.from_pretrained(config['student_model'], local_files_only=True)
        builder=QwenFinanceMaskBuilder(self.tokenizer,processor=None,enable_thinking=config['runtime']['thinking']!='off')
        async with EpisodeProxy(builder=builder, task=request['task'],student_url=request['student_url'],
                                config=config,output=Path(request['directory']),session_id=request['session_id'],
                                sampling_params=request['sampling_params']) as proxy:
            status=await run_native(config,request['task'],proxy)
        return status

    async def handle(self, job):
        task=None
        try:
            request=json.loads((job/'request.json').read_text());self.validate(request)
            if (job/'started.json').exists():
                raise RuntimeError('orphaned previous worker attempt; no automatic trajectory replay')
            atomic(job/'started.json',dict(time=time.time(),pid=os.getpid(),host=socket.gethostname()))
            task=asyncio.create_task(self.execute(request))
            while not task.done():
                heartbeat=json.loads((job/'heartbeat.json').read_text())['time']
                if (job/'cancel.json').exists() or time.time()-heartbeat>90:
                    raise RuntimeError('episode client cancelled or heartbeat expired')
                if time.time()>=self.config['job']['deadline_epoch']:
                    raise TimeoutError('campaign deadline reached')
                await asyncio.wait({task},timeout=2)
            status=task.result()
            atomic(job/'result.json',dict(status=status,time=time.time()))
        except Exception as exc:
            atomic(job/'error.json',dict(error=f'{type(exc).__name__}: {exc}',time=time.time()))
        finally:
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task,return_exceptions=True)

    async def serve(self):
        self.queue.mkdir(parents=True,exist_ok=True)
        active={}
        try:
            while time.time()<self.config['job']['deadline_epoch']:
                active={job:task for job,task in active.items() if not task.done()}
                available = available_memory_mib() if self.memory_headroom_mib else None
                memory_paused = bool(self.memory_headroom_mib and
                                     (available is None or available < self.memory_headroom_mib))
                atomic(self.queue/'worker_status.json',dict(time=time.time(),pid=os.getpid(),active=[p.name for p in active],
                    concurrency=self.concurrency,available_memory_mib=available,admission_paused_for_memory=memory_paused))
                for request in sorted(self.queue.glob('*/request.json')):
                    job=request.parent
                    if memory_paused or len(active)>=self.concurrency: break
                    if job in active or (job/'result.json').exists() or (job/'error.json').exists():continue
                    active[job]=asyncio.create_task(self.handle(job))
                await asyncio.sleep(1)
        finally:
            for task in active.values():task.cancel()
            await asyncio.gather(*active.values(),return_exceptions=True)


def available_memory_mib():
    """Backpressure new episodes, without killing or replaying active work."""
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) // 1024
    except (OSError, ValueError):
        pass
    return None


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path);args=parser.parse_args()
    from .config import read_config
    config=read_config(args.run/'config.json')
    with (args.run/'episode_worker.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        asyncio.run(EpisodeWorker(config,args.run).serve())


if __name__=='__main__':main()
