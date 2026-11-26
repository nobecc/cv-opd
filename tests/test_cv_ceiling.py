import json,unittest
import test_cv as base
class CeilingCVTests(base.CVEpisodeTests):
 async def test_ceiling_is_unlabeled_zero_mask_and_no_student_retry(self):
  async def value(**request):return dict(status='score_ceiling',before_score=1.,score_upper_bound=1.,audit=dict(score_hook='finance_opd.value.native_report_score'))
  self.backend.value=value
  builder=base.Builder()
  async with self.proxy(builder) as proxy:
   result=await proxy.complete(dict(model='student',messages=self.messages))
   self.assertEqual(result['choices'][0]['message']['content'],'FINISH: initial')
   self.assertEqual(len(self.students),1);self.assertIsNone(proxy.continuation.active)
   self.assertEqual(proxy.records[0]['loss_mask'],[0,0]);self.assertEqual(proxy.records[0]['opd_weight'],0)
  audit=json.loads((self.root/'episode/continuation/probe-000/value_audit.json').read_text())
  self.assertEqual(audit['status'],'metric_ceiling');self.assertIsNone(audit['opportunity_label'])
  self.assertNotIn('gain',audit);self.assertNotIn('after_score',audit)
 async def test_invalid_ceiling_does_not_create_supervision(self):
  async def value(**request):return dict(status='score_ceiling',before_score=.8,score_upper_bound=1.,audit=dict(score_hook='finance_opd.value.native_report_score'))
  self.backend.value=value
  async with self.proxy(base.Builder()) as proxy:
   with self.assertRaisesRegex(ValueError,'ceiling proof'):await proxy.complete(dict(model='student',messages=self.messages))
   self.assertTrue(all(not any(x['loss_mask']) for x in proxy.records))
if __name__=='__main__':unittest.main()
