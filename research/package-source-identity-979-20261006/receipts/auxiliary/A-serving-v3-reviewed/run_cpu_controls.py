#!/usr/bin/python3 -I -B
"""Root-run narrow CPU wall/HTTP/owned-cleanup controls; no GPU/model/hash work."""
import argparse,hashlib,http.server,importlib.util,json,os,select,signal,subprocess,sys,threading,time,urllib.request
from pathlib import Path
HERE=Path(__file__).absolute().parent
spec=importlib.util.spec_from_file_location('helpers_under_control',HERE/'runtime_helpers.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
def main():
 p=argparse.ArgumentParser();p.add_argument('--records',type=Path,required=True);a=p.parse_args();assert a.records.is_absolute() and not a.records.exists();a.records.mkdir(parents=True)
 results=[]
 b=h.Budget(total=.45,arms=2,term=.05,kill=.05,join=.05)
 begin=time.monotonic()
 try:
  with b.operation():time.sleep(.35)
 except h.CellDeadline:pass
 else:raise AssertionError('slow preflight escaped whole clock')
 assert time.monotonic()-begin<.30;results.append({'case':'clock-before-preflight','elapsed':time.monotonic()-begin,'work_budget':b.work_end-b.started})
 class Trickle(http.server.BaseHTTPRequestHandler):
  def log_message(self,*args):pass
  def do_GET(self):
   self.send_response(200);self.send_header('Content-Length','500');self.end_headers()
   for i in range(500):
    if stop.is_set():return
    try:self.wfile.write(b'x');self.wfile.flush()
    except (BrokenPipeError,ConnectionResetError):return
    time.sleep(.01)
 stop=threading.Event();server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Trickle);server.daemon_threads=True;t=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.01});t.start();b=h.Budget(total=1.2,arms=2,term=.1,kill=.1,join=.1);begin=time.monotonic()
 try:
  try:
   with b.operation(cap=.15):
    with urllib.request.urlopen('http://127.0.0.1:'+str(server.server_port),timeout=.10) as response:response.read(501)
  except h.CellDeadline:pass
  else:raise AssertionError('trickling body escaped absolute wall deadline')
  assert time.monotonic()-begin<.35;results.append({'case':'trickle-body-wall','elapsed':time.monotonic()-begin})
 finally:stop.set();server.shutdown();t.join(timeout=1);server.server_close();assert not t.is_alive()
 # Two owned CPU child sessions ignore TERM and write a readiness line. Each
 # must be killed/reaped, its nonblocking reader joined, and its pipe closed.
 b=h.Budget(total=2.0,arms=2,term=.1,kill=.2,join=.1);proof_calls=[]
 def proof():proof_calls.append(time.monotonic());return {'CPU_fixture_lock_proof':True}
 for i in range(2):
  proc=subprocess.Popen(['/usr/bin/python3','-I','-B','-c',"import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);print('[server] listening on cpu-fixture',flush=True);time.sleep(30)"],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
  reader=h.LogReader(proc,a.records/('reader-'+str(i)+'.log'));reader.start();assert reader.events.get(timeout=1)=='listening'
  outcome=h.cleanup_owned(proc,reader,b,proof,a.records/('cleanup-'+str(i)+'.json'))
  assert outcome['owned_process_retired'] and outcome['reader_joined'] and outcome['reader_pipe_closed'] and not outcome['cleanup_errors'];assert proc.returncode<0
 assert len(proof_calls)==4 and time.monotonic()<b.end;results.append({'case':'two-TERM-KILL-join-cleanups-in-total','elapsed':time.monotonic()-b.started,'lock_proofs':len(proof_calls)})
 # Cleanup failure is recorded and attached; it cannot replace the primary.
 b=h.Budget(total=1.2,arms=2,term=.1,kill=.1,join=.1);proc=subprocess.Popen(['/usr/bin/python3','-I','-B','-c',"print('[server] listening on cpu-fixture',flush=True);import time;time.sleep(30)"],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True);reader=h.LogReader(proc,a.records/'primary-reader.log');reader.start();assert reader.events.get(timeout=1)=='listening';primary=ValueError('synthetic primary body failure')
 def badproof():raise RuntimeError('synthetic cleanup lock-proof failure')
 try:
  try:raise primary
  except BaseException as error:
   h.cleanup_owned(proc,reader,b,badproof,a.records/'primary-cleanup.json',error)
   raise
 except ValueError as caught:assert caught is primary and any('cleanup failed' in note for note in caught.__notes__)
 else:raise AssertionError('primary error was lost')
 assert proc.poll() is not None and proc.stdout.closed and not reader.thread.is_alive();results.append({'case':'primary-error-preserved-with-cleanup-notes','primary':str(primary),'notes':primary.__notes__})
 # Regression: leader exits while its owned descendant holds stdout open.
 # Baseline misses it; cleanup of that CPU fixture uses the child's captured
 # pidfd only, not the stale numeric group already reaped by old helper.
 spec=importlib.util.spec_from_file_location('old_cleanup',HERE/'BASELINE-runtime_helpers.py');baseline=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline)
 script="import os,signal,time; child=os.fork();\nif child==0:\n signal.signal(signal.SIGTERM,signal.SIG_IGN);print('[server] listening on descendant='+str(os.getpid()),flush=True);time.sleep(30)\nelse: os._exit(0)"
 def fixture(label):
  proc=subprocess.Popen(['/usr/bin/python3','-I','-B','-c',script],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
  ownership=h.OwnedSession(proc);reader=h.LogReader(proc,a.records/(label+'.log'));reader.start();assert reader.events.get(timeout=1)=='listening'
  deadline=time.monotonic()+1
  while os.waitid(os.P_PIDFD,ownership.pidfd,os.WEXITED|os.WNOHANG|os.WNOWAIT) is None:
   assert time.monotonic()<deadline;time.sleep(.01)
  text=(a.records/(label+'.log')).read_text();child=int(text.split('descendant=')[1].split()[0]);childfd=os.pidfd_open(child)
  return proc,ownership,reader,childfd,child
 def exited(fd):
  poll=select.poll();poll.register(fd,select.POLLIN);return bool(poll.poll(0))
 proc,anchor,reader,childfd,child=fixture('old-descendant');b=h.Budget(total=1.2,arms=2,term=.1,kill=.1,join=.1)
 try:
  proc.poll();assert proc.returncode==0
  old_result=baseline.cleanup_owned(proc,reader,b,proof,a.records/'old-descendant-cleanup.json')
  assert old_result['owned_process_retired'] and old_result['reader_pipe_closed'] and not exited(childfd),'old gap did not reproduce'
 finally:
  if not exited(childfd):signal.pidfd_send_signal(childfd,signal.SIGKILL)
  poll=select.poll();poll.register(childfd,select.POLLIN);assert poll.poll(1000);os.close(childfd);anchor.close()
 proc,anchor,reader,childfd,child=fixture('new-descendant');b=h.Budget(total=1.8,arms=2,term=.1,kill=.2,join=.1)
 try:
  new_result=h.cleanup_owned(proc,reader,b,proof,a.records/'new-descendant-cleanup.json',ownership=anchor)
  assert exited(childfd) and new_result['owned_process_retired'] and new_result['reader_joined'] and new_result['reader_pipe_closed']
  assert new_result['owned_session_retirement'][0]['live_members_after']==[] and new_result['owned_session_retirement'][0]['leader_reaped_after_session_retired'] is True
  results.append({'case':'exited-leader-live-owned-descendant','old_missing_retirement_reproduced':True,'new_descendant_retired':True,'anchored_signals_before_reap':True})
 finally:
  if not exited(childfd):signal.pidfd_send_signal(childfd,signal.SIGKILL)
  os.close(childfd)
 # No user or canonical GPU lock is acquired by these CPU controls.
 out={'cases':results,'helpers_sha256':hashlib.sha256((HERE/'runtime_helpers.py').read_bytes()).hexdigest(),'runner_sha256':hashlib.sha256((HERE/'run_cell.py').read_bytes()).hexdigest(),'GPU_or_model_or_artifact_hash_or_audit_invoked':False,'qualified':False}
 (a.records/'RESULT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'cases':len(results),'CPU_controls_passed':True,'qualified':False}))
if __name__=='__main__':main()
