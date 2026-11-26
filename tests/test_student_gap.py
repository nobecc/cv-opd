import json,tempfile,unittest
from pathlib import Path
from unittest.mock import AsyncMock,patch
import test_cv as base
from finance_opd.value import score_student_stop
class StudentGapTests(base.CVEpisodeTests):
 async def test_ceiling_excludes_without_teacher_probe_or_student_resample(self):
  self.config['continuation']['require_student_quality_gap']=True;self.config['runtime']['skip_context_teacher_scoring']=True
  with patch('finance_opd.value.score_student_stop',new=AsyncMock(return_value={'score':1.,'score_upper_bound':1.}))as score:
   async with self.proxy(base.Builder())as proxy:
    await proxy.complete(dict(model='student',messages=self.messages))
    self.assertEqual(score.await_args.kwargs['report'],'FINISH: initial');self.assertEqual(self.probes,[]);self.assertEqual(self.teachers,[]);self.assertEqual(len(self.students),1);self.assertEqual(proxy.records[0]['loss_mask'],[0,0])
  audit=json.loads((self.root/'episode/continuation/probe-000/value_audit.json').read_text());self.assertEqual(audit['status'],'student_metric_ceiling');self.assertFalse(audit['teacher_probe_executed']);self.assertIsNone(audit['opportunity_label'])
 async def test_deficit_still_requires_full_counterfactual_and_fresh_retry(self):
  self.config['continuation']['require_student_quality_gap']=True
  with patch('finance_opd.value.score_student_stop',new=AsyncMock(return_value={'score':.5,'score_upper_bound':1.})):
   async with self.proxy(base.Builder())as proxy:
    await proxy.complete(dict(model='student',messages=self.messages));self.assertEqual(len(self.probes),1);self.assertEqual(len(self.students),2);self.assertEqual([x['input_ids']for x in self.students],[[1,2,3],[1,2,3]]);self.assertEqual(proxy.records[1]['loss_mask'],[1,1])
 async def test_outage_is_pending_not_a_zero_score(self):
  self.config['continuation']['require_student_quality_gap']=True
  with patch('finance_opd.value.score_student_stop',new=AsyncMock(side_effect=RuntimeError('judge503'))):
   async with self.proxy(base.Builder())as proxy:
    await proxy.complete(dict(model='student',messages=self.messages));self.assertEqual(self.probes,[]);self.assertEqual(len(self.students),1)
  audit=json.loads((self.root/'episode/continuation/probe-000/value_audit.json').read_text());self.assertEqual(audit['status'],'student_scoring_pending');self.assertNotIn('student_stop_quality',audit);self.assertIsNone(audit['opportunity_label'])
class NativeStudentScoreTests(unittest.IsolatedAsyncioTestCase):
 async def test_native_bounds_and_no_score_fabrication(self):
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp)
   async def score(config,task,report,output):
    output.mkdir(parents=True);(output/'native_score.json').write_text(json.dumps(dict(status='succeeded',components={'answer_quality':dict(status='scored',score=-.1,score_range=[-1.,1.])})));return -.1
   with patch('finance_opd.value._score',score):
    x=await score_student_stop(config={'continuation':{}},task={'task':'task'},report='FINISH: original',output=root/'valid');self.assertEqual(x['score'],-.1);self.assertEqual(x['score_lower_bound'],-1.)
   async def invalid(config,task,report,output):
    output.mkdir(parents=True);(output/'native_score.json').write_text(json.dumps(dict(status='failed',components={'answer_quality':dict(status='pending',score=0,score_range=[-1.,1.])})));return 0
   with patch('finance_opd.value._score',invalid):
    with self.assertRaisesRegex(ValueError,'verified native range'):await score_student_stop(config={'continuation':{}},task={'task':'task'},report='FINISH: original',output=root/'invalid')
   self.assertFalse((root/'invalid/quality.json').exists())
if __name__=='__main__':unittest.main()
