import copy,os,unittest
from finance_opd.continuation import prepare_teacher_prompt, probe_instruction
from rl.examples.pi_finance.qwen_mask_builder import QwenFinanceMaskBuilder

class TeacherTailPromptTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  from transformers import AutoTokenizer
  path=os.environ.get('FINANCE_OPD_TEST_TOKENIZER')
  if not path:raise unittest.SkipTest('Explicit target tokenizer required for the real-template test')
  cls.tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
 def setUp(self):
  self.b=QwenFinanceMaskBuilder(self.tok,enable_thinking=True)
  self.msg=[{'role':'system','content':'Research. Private teacher continuation instruction.'},{'role':'user','content':'Investigate.'},{'role':'assistant','content':'<think>HISTORICAL_REASONING_MUST_REMAIN_abc123</think>\n<tool_call>example</tool_call>'},{'role':'tool','name':'execute_code','content':'A reported value: 123'}]
  self.instruction='Inspect any material unresolved question. Stop if none remains.'
 def test_reminder_declares_real_teacher_tool_limits(self):
  text=probe_instruction({'teacher_instruction_position':'tail_reminder','teacher_report_mode':'evidence_delta'})
  self.assertIn('fetch_content(mode=answer) are unavailable',text)
  self.assertIn('fetch_content(mode=readable or raw)',text)
 def test_system_default_preserves_existing_behavior(self):
  original=self.b.prepare_generate_input('a',self.msg)
  p,a=prepare_teacher_prompt(self.b,'b',self.msg,self.instruction,len(self.msg),0,'system')
  self.assertEqual(original.input_ids,p.input_ids);self.assertIsNone(a);self.assertEqual(p.messages,self.msg)
 def test_exact_prefix_keeps_reasoning_that_fresh_user_render_drops(self):
  original=self.b.prepare_generate_input('a',self.msg);before=copy.deepcopy(self.msg)
  naive=self.b.prepare_generate_input('naive',self.msg+[{'role':'user','content':self.instruction}])
  self.assertNotIn('HISTORICAL_REASONING_MUST_REMAIN_abc123',naive.messages_str)
  p,a=prepare_teacher_prompt(self.b,'b',self.msg,self.instruction,len(self.msg),0,'tail_reminder')
  opening=self.b._encode('<|im_start|>assistant\n<think>\n')
  self.assertEqual(original.input_ids[:-len(opening)],p.input_ids[:len(original.input_ids)-len(opening)])
  self.assertIn('HISTORICAL_REASONING_MUST_REMAIN_abc123',p.messages_str);self.assertEqual(before,self.msg)
  self.assertTrue(a['preserved_prior_ids']);self.assertFalse(a['historical_probabilities_reused']);self.assertFalse(any(p.mask));self.assertTrue(all(v is None for v in p.probabilities))
 def test_next_tool_turn_uses_authoritative_generation_cache(self):
  p,_=prepare_teacher_prompt(self.b,'b',self.msg,self.instruction,len(self.msg),0,'tail_reminder')
  text='I will inspect a source. <tool_call>example</tool_call><|im_end|>';ids=self.b._encode(text)
  entry=self.b.record_generation(p,ids,[[-.1,x]for x in ids],text,'stop')
  history=self.msg+[{'role':'assistant','content':text},{'role':'tool','name':'execute_code','content':'A new observation: 456'}]
  nxt,_=prepare_teacher_prompt(self.b,'b',history,self.instruction,len(self.msg),1,'tail_reminder')
  self.assertEqual(entry['ids'],nxt.input_ids[:len(entry['ids'])]);self.assertEqual(nxt.messages[len(self.msg)],{'role':'user','content':self.instruction});self.assertEqual(len(history),len(nxt.messages)-1)
 def test_unknown_position_rejected_before_probe(self):
  from finance_opd.cv import validate_continuation_config
  with self.assertRaises(ValueError):
   validate_continuation_config({'continuation':{'enabled':True,'teacher_instruction_position':'invalid'}})
 def test_unknown_mode_rejected(self):
  with self.assertRaises(ValueError):prepare_teacher_prompt(self.b,'b',self.msg,self.instruction,len(self.msg),0,'invalid')
 def test_non_thinking_template_rejected(self):
  b=QwenFinanceMaskBuilder(self.tok,enable_thinking=False)
  with self.assertRaises(ValueError):prepare_teacher_prompt(b,'b',self.msg,self.instruction,len(self.msg),0,'tail_reminder')
 def test_unrecognized_generation_boundary_rejected(self):
  old=self.b.prepare_generate_input
  def altered(*args):
   p=old(*args);p.input_ids=p.input_ids[:-1];return p
  self.b.prepare_generate_input=altered
  with self.assertRaises(ValueError):prepare_teacher_prompt(self.b,'b',self.msg,self.instruction,len(self.msg),0,'tail_reminder')
if __name__=='__main__':unittest.main()
