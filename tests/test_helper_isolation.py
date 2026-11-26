import copy,unittest
from types import SimpleNamespace
from unittest.mock import patch
import test_cv as base
from finance_opd.episode import EpisodeProxy
class Builder(base.Builder):
 def __init__(self):super().__init__();self.settings={};self.cleared=[];self.tokenizer=SimpleNamespace(decode=lambda ids,**kwargs:"excerpt")
 def configure_session(self,s,tools,thinking):self.settings[s]=copy.deepcopy(tools)
 def clear_session(self,s):self.cleared.append(s)
class HelperIsolationTests(base.CVEpisodeTests):
 async def test_helper_does_not_overwrite_main_schema_or_trigger_probe(self):
  self.config['runtime']['skip_context_teacher_scoring']=True
  b=Builder();b.settings['student']=[{'main':'original'}]
  async with self.proxy(b)as proxy:
   payload={'model':'student','messages':[{'role':'system','content':'Page helper'},{'role':'user','content':'Revenue?'}],'tools':[], 'chat_template_kwargs':{'enable_thinking':True},'metadata':{'tool_protocol':'qwen3_xml_v1','channel':'document_helper'}}
   with patch('rl.examples.pi_finance.qwen_protocol.parse_response',return_value={'role':'assistant','content':'excerpt'}):await proxy.complete(payload)
   self.assertEqual(self.probes,[]);self.assertEqual(self.teachers,[]);self.assertEqual(b.settings['student'],[{'main':'original'}]);self.assertEqual(len(proxy.helper_sessions),0)
   session=b.contexts[0][0];self.assertEqual(b.contexts[0][0],session);self.assertEqual(b.settings[session],[])
   self.assertEqual(proxy.records[0]['loss_mask'],[0,0]);self.assertEqual(proxy.records[0]['response_ids'],[8,9]);self.assertEqual(proxy.records[0]['rollout_log_probs'],[-.7,-.8])
  self.assertIn(session,b.cleared)
 async def test_unknown_channel_and_helper_tools_rejected_before_sampling(self):
  b=Builder()
  async with self.proxy(b)as proxy:
   p={'model':'student','messages':self.messages,'metadata':{'tool_protocol':'qwen3_xml_v1','channel':'typo'},'chat_template_kwargs':{'enable_thinking':True},'tools':[]}
   with self.assertRaisesRegex(ValueError,'unknown Qwen'):await proxy.complete(p)
   p['metadata']['channel']='document_helper';p['tools']=[{'type':'function','function':{'name':'x'}}]
   with self.assertRaisesRegex(ValueError,'tool-free'):await proxy.complete(p)
   self.assertEqual(self.students,[])
if __name__=='__main__':unittest.main()
