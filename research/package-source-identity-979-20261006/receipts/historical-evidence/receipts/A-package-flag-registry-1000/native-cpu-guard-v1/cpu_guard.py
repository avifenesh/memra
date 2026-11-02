#!/usr/bin/python3 -I -B
"""Meter one exact broker job cgroup; pidfd cleanup only, no numeric group kill."""
import argparse,ctypes,json,os,re,selectors,signal,stat,subprocess,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
SAMPLE_SECONDS=.05
class OwnedCgroup:
 def __init__(self,job):
  assert re.fullmatch('[0-9a-f]{12}',job),'missing exact broker job id'
  rows=Path('/proc/self/cgroup').read_text().splitlines();assert len(rows)==1 and rows[0].startswith('0::/')
  self.relative=rows[0][3:];assert self.relative.split('/')[-1]=='memra-rig-job-'+job+'.service','caller is not the exact admitted broker service'
  self.path=Path('/sys/fs/cgroup')/self.relative.lstrip('/');self.fd=os.open(self.path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW);info=os.fstat(self.fd);self.identity=(info.st_dev,info.st_ino);self.self_pid=os.getpid()
  assert self.read('cgroup.type').strip()=='domain';self.max=self.read('cpu.max').split();assert len(self.max)==2 and self.max[0]!='max';self.quota_cores=int(self.max[0])/int(self.max[1]);assert self.quota_cores<=4,'broker CPU quota differs';self.period_seconds=int(self.max[1])/1e6
 def read(self,name):
  fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=self.fd)
  try:return os.read(fd,1024*1024).decode()
  finally:os.close(fd)
 def check(self):
  current=os.stat(self.path,follow_symlinks=False);assert (current.st_dev,current.st_ino)==self.identity and stat.S_ISDIR(current.st_mode),'owned cgroup replaced'
  assert Path('/proc/self/cgroup').read_text().strip()=='0::'+self.relative,'guard moved out of broker cgroup'
 def usage(self):
  self.check();rows=dict(line.split() for line in self.read('cpu.stat').splitlines());return int(rows['usage_usec'])
 def members(self):return [int(value) for value in self.read('cgroup.procs').split() if int(value)!=self.self_pid]
 def descendants(self):
  # The guard is a subreaper. Orphaned/detached descendants reparent to it;
  # broker ancestors/siblings in the same job cgroup are never selected.
  rows={}
  for pid in self.members():
   fd=self.pin(pid)
   if fd is None:continue
   try:
    raw=Path('/proc/'+str(pid)+'/stat').read_text();fields=raw[raw.rfind(')')+2:].split();rows[pid]=(int(fields[1]),fd,fields[0])
   except (FileNotFoundError,ProcessLookupError):os.close(fd)
  owned={self.self_pid}
  while True:
   expanded=owned|{pid for pid,(parent,fd,state) in rows.items() if parent in owned}
   if expanded==owned:break
   owned=expanded
  result=[]
  for pid,(parent,fd,state) in rows.items():
   if pid in owned and state!='Z':result.append((pid,fd))
   else:os.close(fd)
  return result
 def pin(self,pid):
  try:fd=os.pidfd_open(pid)
  except ProcessLookupError:return None
  try:
   # The pidfd fixes the process generation before membership verification.
   if Path('/proc/'+str(pid)+'/cgroup').read_text().strip()!='0::'+self.relative:os.close(fd);return None
   return fd
  except (FileNotFoundError,ProcessLookupError):os.close(fd);return None
 def terminate(self,deadline):
  attempts=[]
  while time.monotonic()<deadline:
   pinned=self.descendants()
   if not pinned:return attempts
   try:
    for pid,fd in pinned:
     try:signal.pidfd_send_signal(fd,signal.SIGKILL);attempts.append({'pid_observed':pid,'signal':'SIGKILL','pidfd_generation_pinned':True})
     except ProcessLookupError:pass
   finally:
    for _,fd in pinned:os.close(fd)
   time.sleep(SAMPLE_SECONDS)
  remaining=self.descendants()
  try:assert remaining==[],'owned descendants did not finish within teardown deadline'
  finally:
   for _,fd in remaining:os.close(fd)
  return attempts
 def close(self):os.close(self.fd)

def run(command,record,CPU_seconds=3600,wall_seconds=1775,teardown_seconds=20):
 assert command and record.is_absolute() and not record.exists();record.mkdir(parents=True)
 group=OwnedCgroup(os.environ.get('MEMRA_RIG_JOB',''));baseline=group.usage();start=time.monotonic();limit_usec=int(CPU_seconds*1_000_000)
 libc=ctypes.CDLL(None,use_errno=True);assert libc.prctl(36,1,0,0,0)==0,'cannot anchor orphan cleanup to guard subreaper'
 assert CPU_seconds>0 and wall_seconds>0 and teardown_seconds>0
 cleanup_CPU_reserve=group.quota_cores*(teardown_seconds+SAMPLE_SECONDS+group.period_seconds)
 work_CPU_stop_usec=limit_usec-int(cleanup_CPU_reserve*1e6);assert work_CPU_stop_usec>0,'CPU budget cannot cover finite cleanup reserve'
 # This is an admission-independent private guard, not compiler options.
 # Reserve cpu.max times the finite teardown interval, sample interval and
 # one quota period. This keeps teardown inside the same3600 budget. Any
 # actual overshoot or failed cleanup is still reported and fails acceptance.
 samples=[];child=None;pidfd=None;selector=selectors.DefaultSelector();reason=None;code=None;cleanup=[];failure=None
 def sample():
  usage=group.usage();elapsed=time.monotonic()-start;delta=usage-baseline;assert delta>=0
  row={'elapsed_wall_seconds':elapsed,'owned_cgroup_CPU_usec':delta};samples.append(row)
  with (record/'cpu-samples.jsonl').open('ab') as stream:stream.write(json.dumps(row,separators=(',',':')).encode()+b'\n')
  return delta,elapsed
 try:
  before_CPU,before_wall=sample()
  assert before_CPU<limit_usec and before_wall<wall_seconds
  child=subprocess.Popen(command,start_new_session=True)
  pidfd=os.pidfd_open(child.pid);assert Path('/proc/'+str(child.pid)+'/cgroup').read_text().strip()=='0::'+group.relative;selector.register(pidfd,selectors.EVENT_READ)
  while True:
   cpu,elapsed=sample()
   if cpu>=work_CPU_stop_usec:reason='CPU-ceiling-cleanup-reserve';break
   if elapsed>=wall_seconds:reason='wall-deadline';break
   if selector.select(min(SAMPLE_SECONDS,wall_seconds-elapsed)):
    code=child.wait(timeout=1);reason='child-exit';break
 except BaseException as error:
  reason=reason or 'guard-error';failure=repr(error)
 finally:
  # The meter began before the native driver's preflight. Its whole child
  # tree and any detached descendants remain within this owned cgroup.
  try:
   cleanup=group.terminate(time.monotonic()+teardown_seconds)
   if child is not None:code=child.wait(timeout=1)
   # Reap adopted children without blocking; live descendants werepidfd-killed.
   while True:
    try:pid,status=os.waitpid(-1,os.WNOHANG)
    except ChildProcessError:break
    if pid==0:break
   final_CPU,final_wall=sample()
  except BaseException as error:
   failure=(failure+'; ' if failure else '')+repr(error);final_CPU=group.usage()-baseline;final_wall=time.monotonic()-start
  if pidfd is not None:os.close(pidfd)
  selector.close();group.close()
  result={'owned_broker_cgroup':group.relative,'command':command,'CPU_ceiling_seconds':CPU_seconds,'wall_deadline_seconds':wall_seconds,'sample_seconds':SAMPLE_SECONDS,'quota_cores':group.quota_cores,'CPU_from_before_preflight_through_teardown_seconds':final_CPU/1e6,'wall_from_before_preflight_through_teardown_seconds':final_wall,'CPU_ceiling_observed':reason=='CPU-ceiling-cleanup-reserve','cleanup_CPU_reserve_seconds':cleanup_CPU_reserve,'work_CPU_stop_seconds':work_CPU_stop_usec/1e6,'CPU_overshoot_seconds':max(0,final_CPU/1e6-CPU_seconds),'reason':reason,'child_exit':code,'owned_cleanup':cleanup,'failure':failure,'native_success':reason=='child-exit' and code==0 and not failure and final_CPU<=limit_usec and final_wall<=wall_seconds+teardown_seconds,'model_or_GPU_qualification':False}
  (record/'CPU-GUARD-RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
 return 0 if result['native_success'] else 124 if reason in ['CPU-ceiling-cleanup-reserve','wall-deadline'] else 1

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--records',type=Path,required=True);parser.add_argument('--fixture',action='store_true');parser.add_argument('--CPU-seconds',type=float,default=3600);parser.add_argument('--wall-seconds',type=float,default=1775);parser.add_argument('--teardown-seconds',type=float,default=20);parser.add_argument('command',nargs=argparse.REMAINDER);args=parser.parse_args();command=args.command[1:] if args.command[:1]==['--'] else args.command
 if not args.fixture:assert args.CPU_seconds==3600 and args.wall_seconds==1775 and args.teardown_seconds==20
 sys.exit(run(command,args.records.absolute(),args.CPU_seconds,args.wall_seconds,args.teardown_seconds))
if __name__=='__main__':main()
