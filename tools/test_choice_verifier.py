import unittest
from copy import deepcopy
from choice_verifier import check_completed, check_refusal, Invalid

def fixture():
    identity='fixture'; output=3
    callbacks=[{'kind':'open','id':identity}]+[{'kind':'token','id':identity,'output':i} for i in range(1,4)]
    callbacks += [{'kind':'terminal','id':identity,'outcome':'complete','prompt':10,'cached':2,'output':3,'observed_prompt':10,'observed_cached':2,'observed_output':3},{'kind':'drop','id':identity,'terminal':True}]
    return {'id':identity,'n':2,'seed':101,'body':{'choices':[{'index':0,'text':'a','finish_reason':'length'},{'index':1,'text':'bc','finish_reason':'length'}],'usage':{'prompt_tokens':10,'completion_tokens':3,'total_tokens':13,'prompt_tokens_details':{'cached_tokens':2}}},'callbacks':callbacks,'worker_rows':[{'group':identity,'index':0,'seed':101,'prompt':10,'cached':2,'output':1},{'group':identity,'index':1,'seed':102,'prompt':10,'cached':2,'output':2}],'forks':[{'choices':2,'copies':1,'leader':0}],'leader_prime_segments':1,'follower_prime_segments':0,'reserve':{'prompt':10,'output':32},'resolved_output_bound':16,'packets':[{'choices':[{'index':0,'finish_reason':'length'}]},{'choices':[{'index':1,'finish_reason':'length'}]},'[DONE]']}

class Controls(unittest.TestCase):
    def reject(self,cell,edge):
        with self.assertRaises(Invalid) as e:check_completed(cell)
        self.assertEqual(e.exception.edge,edge)
    def test_success_cannot_satisfy_an_intended_refusal(self):
        with self.assertRaises(Invalid) as e:check_refusal({'status':200,'body':{'choices':[]},'worker_rows':[]},'best_of')
        self.assertEqual(e.exception.edge,'refusal')
        check_refusal({'status':400,'body':{'error':{'param':'best_of'}},'worker_rows':[]},'best_of')
    def test_positive(self):self.assertTrue(check_completed(fixture())['pass'])
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
    def test_duplicate_terminal(self):
        x=fixture();x['callbacks'].append(deepcopy(x['callbacks'][-2]));self.reject(x,'callback_lifetime')

if __name__=='__main__':unittest.main()
