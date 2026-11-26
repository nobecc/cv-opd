import json,unittest
import test_cv as cv_base
import test_value as value_base
from finance_opd.value import parse_verification
class NoveltyCVTests(cv_base.CVEpisodeTests):
 async def test_positive_gain_without_novelty_is_not_admitted(self):
  self.config['continuation']['require_novel_evidence']=True
  original=self.backend.value
  async def value(**request):return {**await original(**request),'novelty_valid':False}
  self.backend.value=value
  async with self.proxy(cv_base.Builder()) as proxy:
   await proxy.complete(dict(model='student',messages=self.messages))
   self.assertEqual(len(self.students),1);self.assertIsNone(proxy.continuation.active)
   self.assertFalse(any(proxy.records[0]['loss_mask']))
  audit=json.loads((self.root/'episode/continuation/probe-000/value_audit.json').read_text())
  self.assertEqual(audit['status'],'not_admitted');self.assertFalse(audit['novelty_valid']);self.assertGreater(audit['gain'],0)
 async def test_missing_novelty_is_pending_not_false_or_true(self):
  self.config['continuation']['require_novel_evidence']=True
  async with self.proxy(cv_base.Builder()) as proxy:
   await proxy.complete(dict(model='student',messages=self.messages))
   self.assertEqual(len(self.students),1);self.assertFalse(any(proxy.records[0]['loss_mask']))
  audit=json.loads((self.root/'episode/continuation/probe-000/value_audit.json').read_text())
  self.assertEqual(audit['status'],'evaluation_pending');self.assertIsNone(audit['opportunity_label'])
 async def test_verified_new_evidence_keeps_same_state_training(self):
  self.config['continuation']['require_novel_evidence']=True
  original=self.backend.value
  async def value(**request):return {**await original(**request),'novelty_valid':True}
  self.backend.value=value
  await self.test_same_state_retry_private_reference_and_actual_token_targets()
class NoveltyValueTests(value_base.ValueTests):
 async def test_missing_required_novelty_is_pending(self):
  self.config['continuation']['require_novel_evidence']=True
  with self.assertRaisesRegex(ValueError,'omitted required novelty_valid'):await self.evaluate()
  self.assertEqual(self.score_requests,[])
 async def test_explicit_false_novelty_preserved_for_admission(self):
  self.config['continuation']['require_novel_evidence']=True
  x=json.loads(self.verifier_text);x['novelty_valid']=False;self.verifier_text=json.dumps(x)
  result=await self.evaluate();self.assertFalse(result['novelty_valid']);self.assertGreater(result['gain'],0)
 def test_novelty_strings_rejected(self):
  x=json.loads(self.verifier_text);x['novelty_valid']='false'
  with self.assertRaisesRegex(ValueError,'JSON boolean'):parse_verification(x)
if __name__=='__main__':unittest.main()
