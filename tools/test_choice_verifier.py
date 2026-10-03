import unittest
import ctypes
import tempfile
from pathlib import Path
from copy import deepcopy
from choice_verifier import check_completed, check_refusal, check_repeat, check_tools, check_stream_twin, check_greedy, Invalid
from native_choices import Pressure

def fixture():
    identity='fixture'; output=3
    callbacks=[{'kind':'open','id':identity},{'kind':'prompt','id':identity,'prompt':10,'cached':2}]+[{'kind':'token','id':identity,'output':i} for i in range(1,4)]
    callbacks += [{'kind':'terminal','id':identity,'outcome':'complete','prompt':10,'cached':2,'output':3,'observed_prompt':10,'observed_cached':2,'observed_output':3},{'kind':'drop','id':identity,'terminal':True}]
    return {'id':identity,'n':2,'seed':101,'body':{'choices':[{'index':0,'text':'a','finish_reason':'length'},{'index':1,'text':'bc','finish_reason':'length'}],'usage':{'prompt_tokens':10,'completion_tokens':3,'total_tokens':13,'prompt_tokens_details':{'cached_tokens':2}}},'callbacks':callbacks,'worker_rows':[{'group':identity,'index':0,'seed':101,'prompt':10,'cached':2,'output':1},{'group':identity,'index':1,'seed':102,'prompt':10,'cached':2,'output':2}],'forks':[{'choices':2,'copies':1,'leader':0}],'leader_prime_segments':1,'follower_prime_segments':0,'reserve':{'prompt':10,'output':32},'resolved_output_bound':16,'packets':[{'choices':[{'index':0,'finish_reason':'length'}]},{'choices':[{'index':1,'finish_reason':'length'}]},'[DONE]']}

class Controls(unittest.TestCase):
    def pressure(self,out,fail=False):
        class Driver:
            held=0
            calls=[]
            def cudaSetDevice(self,_):return 0
            def cudaMemGetInfo(self,free,total):
                free._obj.value=4*1024**3-self.held;total._obj.value=24*1024**3;return 0
            def cudaMalloc(self,ptr,amount):
                self.calls.append('allocate')
                if fail:return 2
                self.held=amount;ptr._obj.value=4096;return 0
            def cudaFree(self,_):self.calls.append('free');self.held=0;return 0
            def cudaDeviceReset(self):self.calls.append('reset-own-context');return 0
        p=Pressure.__new__(Pressure);p.out=Path(out);p.ptr=ctypes.c_void_p();p.api=Driver();p.api.calls=[];p.state={};return p
    def test_owned_pressure_lifecycle_keeps_headroom_and_releases_only_its_allocation(self):
        with tempfile.TemporaryDirectory() as out:
            p=self.pressure(out)
            with p:self.assertEqual(p.state['after_free'],1024**3)
            self.assertEqual(p.api.calls,['allocate','free','reset-own-context'])
            self.assertEqual(p.api.held,0)
    def test_failed_owned_pressure_allocation_still_releases_its_context(self):
        with tempfile.TemporaryDirectory() as out:
            p=self.pressure(out,True)
            with self.assertRaisesRegex(AssertionError,'owned CUDA allocation failed'):
                with p:pass
            self.assertEqual(p.api.calls,['allocate','reset-own-context'])
    def reject(self,cell,edge):
        with self.assertRaises(Invalid) as e:check_completed(cell)
        self.assertEqual(e.exception.edge,edge)
    def test_success_cannot_satisfy_an_intended_refusal(self):
        with self.assertRaises(Invalid) as e:check_refusal({'status':200,'body':{'choices':[]},'worker_rows':[]},'best_of')
        self.assertEqual(e.exception.edge,'refusal')
        check_refusal({'status':400,'body':{'error':{'param':'best_of'}},'worker_rows':[]},'best_of')
    def test_positive(self):self.assertTrue(check_completed(fixture())['pass'])
    def test_greedy_terminal_reason_cannot_diverge(self):
        reference={'text':'same','stop_reason':'MaxNew'};group={'choices':[{'index':0,'text':'same','finish_reason':'stop'}]}
        with self.assertRaises(Invalid) as e:check_greedy(reference,group)
        self.assertEqual(e.exception.edge,'greedy_identity')
    def test_committed_complete_native_receipt_and_coherent_controls(self):
        import json,hashlib
        from replay_choice_composite import replay
        root=Path(__file__).resolve().parents[1]/'research/n-choices-528-20261003/receipts/native-final-v5'
        result=replay(root)
        self.assertEqual(result['phases'],6)
        self.assertEqual(result['cached_tokens'],256)
        self.assertEqual(result['coherent_controls'],14)
        self.assertFalse(result['qualification'])
        inventory=json.loads((root/'file-sha256.json').read_text())
        for name,expected in inventory.items():self.assertEqual(hashlib.sha256((root/name).read_bytes()).hexdigest(),expected,name)
    def test_stream_accumulator_compares_semantics_to_full_envelope(self):
        body=fixture()['body'];full={**deepcopy(body),'object':'chat.completion','model':'fixture','id':'ephemeral','created':1}
        check_stream_twin(body,full)
    def test_stream_comparison_keeps_actual_output_bytes(self):
        body=fixture()['body'];full=deepcopy(body);full['choices'][1]['text']='dropped bytes'
        with self.assertRaises(Invalid) as e:check_stream_twin(body,full)
        self.assertEqual(e.exception.edge,'stream_content_identity')
    def test_missing_choice(self):
        x=fixture();x['body']['choices'].pop();self.reject(x,'indexed_termination')
    def test_duplicate_choice(self):
        x=fixture();x['body']['choices'][1]=deepcopy(x['body']['choices'][0]);self.reject(x,'indexed_termination')
    def test_premature_done(self):
        x=fixture();x['packets'].insert(1,'[DONE]');self.reject(x,'premature_done')
    def test_shared_rng(self):
        x=fixture();x['worker_rows'][1]['seed']=101;self.reject(x,'rng_isolation')
    def test_independent_output_sum(self):
        x=fixture();x['worker_rows'][1]['output']=1;self.reject(x,'output_sum')
    def test_missing_fork(self):
        x=fixture();x['forks']=[];self.reject(x,'shared_prefill')
    def test_extra_cold_prime(self):
        x=fixture();x['follower_prime_segments']=1;self.reject(x,'shared_prefill')
    def test_prompt_multiplied(self):
        x=fixture();x['reserve']['prompt']=20;self.reject(x,'reservation')
    def test_output_reservation_not_multiplied(self):
        x=fixture();x['reserve']['output']=16;self.reject(x,'reservation')
    def test_duplicate_prompt_callback_refuses(self):
        x=fixture();x['callbacks'].append(deepcopy(x['callbacks'][1]));self.reject(x,'prompt_once')
    def test_duplicate_terminal(self):
        x=fixture();x['callbacks'].append(deepcopy(x['callbacks'][-2]));self.reject(x,'callback_lifetime')
    def test_repeated_producer_hash_must_match(self):
        x=fixture()
        for r in x['worker_rows']:r['token_sha256']='a'*64
        y=deepcopy(x);check_repeat(x,y);y['worker_rows'][1]['token_sha256']='b'*64
        with self.assertRaises(Invalid) as e:check_repeat(x,y)
        self.assertEqual(e.exception.edge,'token_identity')
    def test_coherent_http_accounting_cannot_change_independent_output(self):
        x=fixture();x['body']['usage'].update(completion_tokens=4,total_tokens=14)
        x['callbacks'][-2].update(output=4,observed_output=4)
        self.reject(x,'accounting')
    def test_each_constrained_row_has_its_own_complete_call(self):
        call={'function':{'name':'weather','arguments':'{"city":"Paris"}'}}
        body={'choices':[{'index':i,'finish_reason':'tool_calls','message':{'tool_calls':[deepcopy(call)]}} for i in range(2)]}
        check_tools(body);body['choices'][1]['message']['tool_calls'].clear()
        with self.assertRaises(Invalid) as e:check_tools(body)
        self.assertEqual(e.exception.edge,'constrained_choices')
    def test_constrained_schema_cannot_be_substituted(self):
        body={'choices':[{'index':0,'finish_reason':'tool_calls','message':{'tool_calls':[{'function':{'name':'weather','arguments':'{"city":"Berlin"}'}}]}}]}
        with self.assertRaises(Invalid) as e:check_tools(body)
        self.assertEqual(e.exception.edge,'constrained_choices')

if __name__=='__main__':unittest.main()
