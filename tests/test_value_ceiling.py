import json,unittest
from unittest.mock import patch
import test_value as base
from finance_opd.value import NATIVE_SCORE_HOOK
class CeilingTests(base.ValueTests):
 async def _ceiling_run(self,before=1.0,bounds=(0.,1.),native_status='succeeded',hook=NATIVE_SCORE_HOOK):
  self.config['continuation'].update(score_hook=hook,score_ceiling_short_circuit=True)
  self.native_calls=[]
  async def score(config,task,report,output):
   self.native_calls.append(output.name);output.mkdir(parents=True,exist_ok=True)
   value=before if output.name=='evaluator-before' else .4
   (output/'native_score.json').write_text(json.dumps(dict(status=native_status,components=dict(answer_quality=dict(status='scored',score=value,score_range=bounds)))))
   return value
  with patch('finance_opd.value.validate_value_config'),patch('finance_opd.value._score',score):return await self.evaluate()
 async def test_ceiling_skips_after_and_verifier_without_fake_gain(self):
  result=await self._ceiling_run()
  self.assertEqual(result['status'],'score_ceiling');self.assertEqual(len(self.requests),1)
  self.assertEqual(self.native_calls,['evaluator-before']);self.assertEqual(self.verifier_requests,[])
  self.assertNotIn('after_score',result);self.assertNotIn('gain',result)
  self.assertFalse((self.root/'value/after-report.md').exists())
 async def test_below_ceiling_keeps_matched_pair_and_reuses_before_score(self):
  result=await self._ceiling_run(before=.2)
  self.assertAlmostEqual(result['gain'],.2);self.assertEqual(len(self.requests),2)
  self.assertEqual(self.native_calls,['evaluator-before','evaluator-after'])
  self.assertEqual(self.requests[0]['sampling_params'],self.requests[1]['sampling_params'])
 async def test_bad_bounds_pending_not_ceiling(self):
  for bounds in [(0.,.5),[1.],(1.,1.),(0.,float('nan'))]:
   with self.subTest(bounds=bounds),self.assertRaises(ValueError):await self._ceiling_run(bounds=bounds)
 async def test_unscored_native_is_not_ceiling(self):
  with self.assertRaises(ValueError):await self._ceiling_run(native_status='failed')
 async def test_custom_hook_cannot_claim_native_bounds(self):
  with self.assertRaises(ValueError):await self._ceiling_run(hook='fixture_value_hooks.score')
if __name__=='__main__':unittest.main()
