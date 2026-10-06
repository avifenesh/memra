"""Private serving-cell wall deadline, reader and cleanup helpers."""
import contextlib,json,os,queue,selectors,signal,subprocess,threading,time

class CellDeadline(TimeoutError):
    pass

class CellCancelled(RuntimeError):
    pass

class Budget:
    def __init__(self,total=600,arms=2,term=20,kill=10,join=5):
        self.started=time.monotonic()
        self.total=total;self.term=term;self.kill=kill;self.join=join
        self.end=self.started+total
        self.work_end=self.end-arms*(term+kill+join)
        if self.work_end<=self.started:raise ValueError('no work budget after cleanup reserve')
    def remaining(self,end=None,cap=None):
        left=(self.work_end if end is None else end)-time.monotonic()
        if left<=0:raise CellDeadline('whole-cell wall deadline exhausted')
        return min(left,cap) if cap is not None else left
    @contextlib.contextmanager
    def operation(self,cap=None,end=None):
        seconds=self.remaining(end,cap)
        if signal.getitimer(signal.ITIMER_REAL)!=(0.0,0.0):raise RuntimeError('existing alarm would be replaced')
        prior=signal.getsignal(signal.SIGALRM)
        def expired(signum,frame):raise CellDeadline('operation exceeded remaining whole-cell budget')
        signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,seconds)
        try:yield
        finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,prior)
    def report(self):
        return {'total_seconds':self.total,'work_seconds':self.work_end-self.started,
                'cleanup_reserve_seconds':self.end-self.work_end,
                'elapsed_seconds':time.monotonic()-self.started}

@contextlib.contextmanager
def cancellation():
    old={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
    def cancelled(signum,frame):
        # A second cooperative termination cannot interrupt owned cleanup.
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        raise CellCancelled('serving cell cancelled by signal '+str(signum))
    for s in old:signal.signal(s,cancelled)
    try:yield
    finally:
        for s,h in old.items():signal.signal(s,h)

class LogReader:
    def __init__(self,proc,path):
        self.proc=proc;self.path=path;self.events=queue.Queue();self.stop=threading.Event();self.error=None
        self.thread=threading.Thread(target=self._read,daemon=True)
    def start(self):self.thread.start()
    def _read(self):
        selector=selectors.DefaultSelector();pending=b''
        try:
            fd=self.proc.stdout.fileno();os.set_blocking(fd,False);selector.register(fd,selectors.EVENT_READ)
            with self.path.open('wb') as stream:
                while not self.stop.is_set():
                    if not selector.select(0.05):continue
                    try:data=os.read(fd,65536)
                    except BlockingIOError:continue
                    if not data:self.events.put('eof');return
                    stream.write(data);stream.flush();pending+=data
                    while b'\n' in pending:
                        line,pending=pending.split(b'\n',1)
                        if b'[server] listening on' in line:self.events.put('listening')
                    if len(pending)>1024*1024:raise ValueError('server log line exceeds bound')
        except BaseException as error:
            self.error=error;self.events.put('reader-error')
        finally:selector.close()

# Injectable lock proof permits CPU controls without taking the rig GPU lock.
def cleanup_owned(proc,reader,budget,lock_proof,record_path,primary=None):
    began=time.monotonic();cleanup_end=min(budget.end,began+budget.term+budget.kill+budget.join)
    errors=[];proofs=[]
    def attempt(label,operation):
        try:operation()
        except BaseException as error:errors.append({'stage':label,'error':type(error).__name__+': '+str(error)})
    attempt('lock-before-teardown',lambda:proofs.append(lock_proof()))
    if proc.poll() is None:
        attempt('owned-TERM',lambda:os.killpg(proc.pid,signal.SIGTERM))
        try:
            with budget.operation(end=min(cleanup_end,began+budget.term)):
                proc.wait(timeout=budget.remaining(min(cleanup_end,began+budget.term)))
        except (subprocess.TimeoutExpired,CellDeadline):
            if proc.poll() is None:attempt('owned-KILL',lambda:os.killpg(proc.pid,signal.SIGKILL))
            attempt('owned-reap',lambda:proc.wait(timeout=budget.remaining(min(cleanup_end,began+budget.term+budget.kill))))
        except BaseException as error:errors.append({'stage':'owned-wait','error':type(error).__name__+': '+str(error)})
    reader.stop.set()
    attempt('reader-join',lambda:reader.thread.join(timeout=budget.remaining(cleanup_end, budget.join)))
    # Join happens before pipe close. Nonblocking reader/stop makes the join
    # bounded even if an inherited descendant still holds stdout open.
    attempt('reader-pipe-close',lambda:proc.stdout.close())
    if reader.thread.is_alive():errors.append({'stage':'reader-join','error':'reader remains alive'})
    if reader.error is not None:errors.append({'stage':'reader','error':type(reader.error).__name__+': '+str(reader.error)})
    if proc.poll() is None:errors.append({'stage':'owned-reap','error':'owned process remains running'})
    attempt('lock-after-teardown',lambda:proofs.append(lock_proof()))
    report={'owned_pid':proc.pid,'exit':proc.returncode,'owned_process_retired':proc.poll() is not None,
            'reader_joined':not reader.thread.is_alive(),'reader_pipe_closed':proc.stdout.closed,
            'canonical_lock_before_after':proofs,'cleanup_errors':errors,
            'primary_error':None if primary is None else type(primary).__name__+': '+str(primary),
            'elapsed_seconds':time.monotonic()-began,'whole_budget':budget.report()}
    attempt('teardown-record',lambda:record_path.write_text(json.dumps(report,indent=2)+'\n'))
    if errors:
        summary='owned cleanup failed: '+json.dumps(errors)
        if primary is not None:
            primary.add_note(summary)
        else:raise RuntimeError(summary)
    return report
