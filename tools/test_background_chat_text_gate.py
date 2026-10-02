import copy, importlib.util, unittest, tempfile
from types import SimpleNamespace
from pathlib import Path
spec=importlib.util.spec_from_file_location('bg914_gate',Path(__file__).with_name('background-chat-text-gate.py'))
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
class NativeJudge(unittest.TestCase):
    def setUp(self):
        self.value={'id':'job','text':'hello','tokens':[7],'n_tokens':1,'prompt_tokens':3,'cached_tokens':1}
        self.row={'id':'job','outcome':'complete','prompt_tokens':3,'cached_tokens':1,'completion_tokens':1,'observed_monotonic':100}
    def test_valid_native_callback_and_known_token_truth(self):
        self.assertEqual(g.verify_receipt(self.value,[self.row],False,'complete')['completion_tokens'],1)
    def test_missing_and_double_terminal_callbacks_rejected(self):
        for rows in [[],[self.row,self.row]]:
            with self.subTest(rows=rows),self.assertRaises(AssertionError):g.verify_receipt(self.value,rows,False,'complete')
    def test_usage_and_token_count_corruption_rejected(self):
        for key,value in [('n_tokens',2),('prompt_tokens',4),('cached_tokens',0),('tokens',[])]:
            body=copy.deepcopy(self.value);body[key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):g.verify_receipt(body,[self.row],False,'complete')
    def test_wrong_envelope_and_callback_kind_rejected(self):
        with self.assertRaises(AssertionError):g.verify_receipt(self.value,[self.row],True,'complete')
        with self.assertRaises(AssertionError):g.verify_receipt(self.value,[self.row],False,'cancel_partial')
    def test_native_deadline_boundary_has_a_real_red(self):
        g.verify_deadline(0,[self.row])
        for at in [0,89.9,90]:
            with self.subTest(at=at),self.assertRaises(AssertionError):g.verify_deadline(0,[{**self.row,'observed_monotonic':at}])
    def test_cancel_requires_matching_native_progress(self):
        g.verify_progress('job',[{'id':'job','completion_tokens':32}])
        for p in [[],[{'id':'other','completion_tokens':32}],[{'id':'job','completion_tokens':0}]]:
            with self.subTest(p=p),self.assertRaises(AssertionError):g.verify_progress('job',p)
    def test_vendor_witness_excludes_startup_canary_and_wrong_profile(self):
        profile={'default_temperature':1,'default_top_k':20,'default_top_p':.95,'default_min_p':0}
        trace='[skey] burst sampled=1 temp=1 top_k=20 top_p=.95 min_p=0 pen_on=1\n'
        self.assertEqual(g.vendor_trace('[server] listening on http://localhost\n'+trace,profile)['resolved_arm'],'primary_thinking')
        for log in [trace+'[server] listening on http://localhost\n','[server] listening on http://localhost\n'+trace.replace('top_k=20','top_k=0')]:
            with self.subTest(log=log),self.assertRaises(AssertionError):g.vendor_trace(log,profile)
    def test_cached_path_requires_restore_grid_and_executed_spec_rounds(self):
        cold={**self.value,'prompt_tokens':241,'cached_tokens':0}
        warm={**cold,'cached_tokens':g.capture_len(241,g.gdn_grid({}))}
        row={**self.row,'prompt_tokens':241,'cached_tokens':warm['cached_tokens']}
        trace=f"[prefix-cache] spec restore: {warm['cached_tokens']} of 241 prompt tokens + draft plane\n[R0] pos=241 draft=[7, 8, 9] n_acc=2\n"
        self.assertGreater(g.verify_cache(cold,warm,[row],False,trace)['spec_rounds_observed'],0)
        for wrong in [trace.replace('[prefix-cache] spec restore:','[setup]'),trace.split('[R0]')[0]]:
            with self.subTest(trace=wrong),self.assertRaises(AssertionError):g.verify_cache(cold,warm,[row],False,wrong)
        for wrong in [0,warm['cached_tokens']-1]:
            value={**warm,'cached_tokens':wrong};receipt={**row,'cached_tokens':wrong}
            with self.subTest(cached=wrong),self.assertRaises(AssertionError):g.verify_cache(cold,value,[receipt],False,trace)
    def test_manifest_context_and_actual_server_context_share_the_phase_contract(self):
        for phase,expected in [('short',8192),('cached',8192),('long-chat',32768),('long-text',32768)]:
            with self.subTest(phase=phase),tempfile.TemporaryDirectory() as directory:
                context=g.context_for_phase(phase)
                self.assertEqual(context,expected)
                args=SimpleNamespace(model=Path(directory)/'model.gguf',metadata=Path(directory)/'metadata.toml')
                server=g.Server(args,Path(directory)/'server',18120,context=context)
                try:self.assertEqual(server.env['MEMRA_CTX'],str(expected))
                finally:server.__exit__()
        with self.assertRaises(KeyError):g.context_for_phase('unknown')
if __name__=='__main__':unittest.main()
