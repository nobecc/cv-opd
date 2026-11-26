import asyncio
import copy
import gzip
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from finance_opd.episode_queue import EpisodeWorker, atomic, queued_episode


class QueueTests(unittest.IsolatedAsyncioTestCase):
    def setup_case(self, root):
        task={'id':'original','benchmark':'finfirst','source':{},'task':'Original question'}
        (root/'tasks.jsonl').write_text(json.dumps({'metadata':{k:v for k,v in task.items() if k!='task'},'task':task['task']})+'\n')
        config=dict(student_model='/model',pi_finance='/pi',safactory='/sa',slime='/slime',dataset=str(root/'tasks.jsonl'),
                    runtime=dict(episode_worker_queue=str(root/'queue'),context_length=262144,tool_protocol='qwen3_xml_v1',thinking='low',max_runtime_seconds=20),
                    teacher=dict(model='/teacher',url='http://10.1.1.2:31051'),continuation=dict(references_path='/gold',evaluator_config='/judge'),
                    job=dict(deadline_epoch=time.time()+60))
        directory=root/'episodes'/'session';directory.mkdir(parents=True)
        request=dict(config=config,task=task,directory=str(directory),session_id='session',student_url='http://10.1.1.3:31061',sampling_params={})
        return config,task,directory,request

    async def test_roundtrip_preserves_authoritative_tokens_and_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config,task,directory,request=self.setup_case(root)
            record=dict(prompt_ids=[9,8],response_ids=[7,6],rollout_log_probs=[-.1,-.2],teacher_log_probs=[-.3,-.4],loss_mask=[0,1],weight_version='actual-v3')
            async def execute(req):
                with gzip.open(directory/'turns.jsonl.gz','wt') as f:f.write(json.dumps(record)+'\n')
                return {'native_finished':True,'generated_turns':1}
            worker=EpisodeWorker(config,root,execute=execute)
            client=asyncio.create_task(queued_episode(config,task,request['student_url'],directory,'session',{}))
            while not (root/'queue/session/request.json').exists():await asyncio.sleep(.01)
            await worker.handle(root/'queue/session')
            status,records=await asyncio.wait_for(client,4)
            self.assertTrue(status['native_finished']);self.assertEqual(records,[record])

    async def test_task_identity_path_and_overwrite_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config,task,directory,request=self.setup_case(root);worker=EpisodeWorker(config,root)
            self.assertEqual(worker.validate(request),directory)
            altered=copy.deepcopy(request);altered['task']['task']='Injected alternate objective'
            with self.assertRaises(ValueError):worker.validate(altered)
            altered=copy.deepcopy(request);altered['directory']='/tmp/elsewhere/session'
            with self.assertRaises(ValueError):worker.validate(altered)
            (directory/'turns.jsonl.gz').write_bytes(b'original')
            with self.assertRaises(ValueError):worker.validate(request)
            self.assertEqual((directory/'turns.jsonl.gz').read_bytes(),b'original')

    async def test_cancel_stops_native_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config,task,directory,request=self.setup_case(root);started=asyncio.Event();stopped=asyncio.Event()
            async def execute(req):
                started.set()
                try:await asyncio.sleep(30)
                finally:stopped.set()
            job=root/'queue/session';job.mkdir(parents=True);atomic(job/'request.json',request);atomic(job/'heartbeat.json',{'time':time.time()})
            worker=EpisodeWorker(config,root,execute=execute);running=asyncio.create_task(worker.handle(job));await started.wait();atomic(job/'cancel.json',{})
            await asyncio.wait_for(running,4);self.assertTrue(stopped.is_set());self.assertTrue((job/'error.json').exists());self.assertFalse((job/'result.json').exists())

    async def test_restart_never_replays_started_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config,task,directory,request=self.setup_case(root)
            async def execute(req):self.fail('orphaned request must not replay')
            job=root/'queue/session';job.mkdir(parents=True);atomic(job/'request.json',request);atomic(job/'started.json',{});atomic(job/'heartbeat.json',{'time':time.time()})
            await EpisodeWorker(config,root,execute=execute).handle(job)
            self.assertIn('orphaned',json.loads((job/'error.json').read_text())['error'])

    async def test_concurrency_and_memory_backpressure_preserve_pending_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config,task,directory,request=self.setup_case(root)
            config['runtime'].update(episode_worker_concurrency=3,episode_worker_memory_headroom_mib=2048)
            started=[];release=asyncio.Event();three=asyncio.Event();four=asyncio.Event()
            async def execute(req):
                started.append(req['session_id'])
                if len(started)==3:three.set()
                if len(started)==4:four.set()
                await release.wait()
                return {'native_finished':True}
            worker=EpisodeWorker(config,root,execute=execute)
            for index in range(4):
                sid=f'session-{index}';d=root/'episodes'/sid;d.mkdir()
                req={**request,'session_id':sid,'directory':str(d)}
                job=root/'queue'/sid;job.mkdir(parents=True)
                atomic(job/'request.json',req);atomic(job/'heartbeat.json',{'time':time.time()})
            with patch('finance_opd.episode_queue.available_memory_mib',return_value=1024) as memory:
                serving=asyncio.create_task(worker.serve())
                try:
                    await asyncio.sleep(.1)
                    self.assertEqual(started,[])
                    self.assertTrue(json.loads((root/'queue/worker_status.json').read_text())['admission_paused_for_memory'])
                    memory.return_value=8192
                    await asyncio.wait_for(three.wait(),4)
                    await asyncio.sleep(.1)
                    self.assertEqual(len(started),3)
                    self.assertEqual(len(list((root/'queue').glob('*/started.json'))),3)
                    release.set()
                    await asyncio.wait_for(four.wait(),5)
                    self.assertEqual(len(set(started)),4)
                finally:
                    serving.cancel();await asyncio.gather(serving,return_exceptions=True)

    async def test_invalid_capacity_is_rejected_before_worker_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config,*_=self.setup_case(root)
            for limit in [0,9,True,2.5]:
                config['runtime']['episode_worker_concurrency']=limit
                with self.subTest(limit=limit),self.assertRaises(ValueError):EpisodeWorker(config,root)
