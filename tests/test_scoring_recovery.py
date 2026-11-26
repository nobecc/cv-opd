import asyncio,gzip,json,tempfile
from pathlib import Path
from unittest.mock import AsyncMock,patch
import aiohttp
import test_episode as base
from finance_opd.episode import EpisodeProxy
from finance_opd.teacher_transport import score_teacher
class ScoringRecoveryTests(base.EpisodeTests):
 async def test_teacher_failure_keeps_exact_sample_without_trainable_record(self):
  self.teacher_status=503
  with tempfile.TemporaryDirectory() as root:
   async with EpisodeProxy(builder=base.FixtureBuilder(),task={},student_url=self.student_url,config=self.config(),output=Path(root),session_id='journal',sampling_params={}) as p:
    with self.assertRaises(aiohttp.ClientResponseError): await p.complete(dict(model='student',messages=[dict(role='user',content='task')]))
    files=list((Path(root)/'sampled.private').glob('*.json.gz'));self.assertEqual(len(files),1)
    with gzip.open(files[0],'rt') as f:x=json.load(f)
    self.assertEqual(x['student_request']['input_ids'],[1,2,3]);self.assertEqual(x['response']['meta_info']['output_token_logprobs'],[[-.2,8],[-.2,9]])
    self.assertEqual(x['teacher_prompt_ids'],[1,2,3]);self.assertEqual(len(self.student_requests),1);self.assertEqual(p.records,[])
    self.assertEqual(p.request_stage,'teacher_token_scoring')
 async def test_transient_teacher_retry_preserves_exact_supplied_ids(self):
  c=self.config();c['runtime']['teacher_score_retries']=2
  fn=AsyncMock(side_effect=[aiohttp.ServerDisconnectedError(),[-.3,-.4]])
  with patch('finance_opd.teacher_transport._score_teacher_once',fn),patch('finance_opd.teacher_transport.asyncio.sleep',new=AsyncMock()):
   self.assertEqual(await score_teacher(None,c,[1,2],[8,9]),[-.3,-.4]);self.assertEqual(fn.await_count,2)
   self.assertEqual(fn.await_args_list[0],fn.await_args_list[1])
 async def test_alignment_failure_is_not_retried(self):
  c=self.config();c['runtime']['teacher_score_retries']=2;fn=AsyncMock(side_effect=ValueError('token mismatch'))
  with patch('finance_opd.teacher_transport._score_teacher_once',fn):
   with self.assertRaisesRegex(ValueError,'token mismatch'):await score_teacher(None,c,[1],[8])
   self.assertEqual(fn.await_count,1)
 async def test_student_disconnect_is_not_resampled(self):
  async def disconnected(request):
   await request.read();self.student_requests.append('disconnected');request.transport.close();return aiohttp.web.Response()
  url=await self.serve(disconnected)
  with tempfile.TemporaryDirectory() as root:
   async with EpisodeProxy(builder=base.FixtureBuilder(),task={},student_url=url,config=self.config(),output=Path(root),session_id='lost',sampling_params={}) as p:
    with self.assertRaises(aiohttp.ClientConnectionError):await p.complete(dict(model='student',messages=[dict(role='user',content='task')]))
    self.assertEqual(len(self.student_requests),1);self.assertEqual(p.records,[]);self.assertEqual(p.request_stage,'student_generation')
if __name__=='__main__':__import__('unittest').main()
