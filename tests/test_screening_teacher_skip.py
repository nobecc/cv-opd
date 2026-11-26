import json,unittest
from pathlib import Path
from types import SimpleNamespace
import test_cv as base
from finance_opd.rollout import samples_from_records
class ScreeningSkipTests(base.CVEpisodeTests):
 async def test_zero_loss_screening_does_not_call_teacher_or_fabricate_probabilities(self):
  self.config['runtime']['skip_context_teacher_scoring']=True;self.gain=0
  async with self.proxy(base.Builder()) as proxy:
   await proxy.complete(dict(model='student',messages=self.messages))
   self.assertEqual(self.teachers,[]);x=proxy.records[0]
   self.assertIsNone(x['teacher_log_probs']);self.assertEqual(x['teacher_scoring_status'],'not_requested_zero_loss_screening_context')
   self.assertEqual(x['response_ids'],[8,9]);self.assertEqual(x['rollout_log_probs'],[-.7,-.8]);self.assertEqual(x['loss_mask'],[0,0])
   with self.assertRaisesRegex(ValueError,'cannot become a training sample'):samples_from_records(SimpleNamespace(index=0),[x],{},self.root,SimpleNamespace)
 async def test_trainable_retry_still_scores_actual_tokens_under_private_teacher_prefix(self):
  self.config['runtime']['skip_context_teacher_scoring']=True
  async with self.proxy(base.Builder()) as proxy:
   await proxy.complete(dict(model='student',messages=self.messages))
   self.assertEqual(len(self.teachers),1);self.assertEqual(self.teachers[0]['input_ids'],[11,2,3,8,9])
   self.assertEqual([x['input_ids']for x in self.students],[[1,2,3],[1,2,3]])
   self.assertEqual(proxy.records[1]['teacher_log_probs'],[-.4,-.4]);self.assertEqual(proxy.records[1]['loss_mask'],[1,1])
   self.assertEqual(proxy.records[1]['teacher_scoring_status'],'scored')
 async def test_nonboolean_option_rejected(self):
  self.config['runtime']['skip_context_teacher_scoring']='false'
  with self.assertRaisesRegex(ValueError,'must be boolean'):self.proxy(base.Builder())
if __name__=='__main__':unittest.main()
