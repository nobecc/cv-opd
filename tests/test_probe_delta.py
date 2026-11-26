import copy,unittest
from finance_opd.continuation import probe_instruction,_TEACHER_INSTRUCTION
class ProbeDeltaTests(unittest.TestCase):
 def test_default_unchanged(self):self.assertEqual(probe_instruction({}),_TEACHER_INSTRUCTION)
 def test_delta_is_teacher_only_prompt_option(self):
  x={"teacher_report_mode":"evidence_delta"};old=copy.deepcopy(x);s=probe_instruction(x)
  self.assertIn("separate fixed writer",s);self.assertIn("precise source locators",s);self.assertEqual(x,old)
 def test_invalid_mode_rejected(self):
  with self.assertRaises(ValueError):probe_instruction({"teacher_report_mode":"anything"})
if __name__=="__main__":unittest.main()
