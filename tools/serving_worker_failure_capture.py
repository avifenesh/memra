"""Bounded worker-failure capture using existing owned-process/HTTP primitives.

collect_worker_failure_cell(required,program,*,server,output) borrows a fresh
ready OwnedServer with the exact one-shot fault env and owns its final stop.
No native model launch, lease acquisition or qualification is performed here.
The completion driver waits for all admitted victim prefixes; health/readiness
is polled concurrently so the source's two-second dead window is not missed by
waiting for all clients first. Every request is sent once. Failed captures retain
all available rows plus the missing/unattempted denominator.
"""
from dataclasses import asdict
import signal
import threading
import time

from serving_capture import EvidenceStore, encoded, file_identity
from serving_cancel_phase import _PinnedLog, _LogFailure
from serving_drain_capture import require_inflight_prefix
from serving_http import capture_request
from serving_listener import ListenerOwnershipError, prove_listener
from serving_process import OwnedServer, process_identity
from serving_release import account_attempts, json_object, require
from serving_policy import _drain_lifecycle
from serving_worker_failure import (DETAIL, MAX_PROBES, PROBE_PREFIX, fault_environment,
                                   probe_facts, validate_program, evaluate_worker_failure_cell)


def _start_owned_client(job):
    """Record launch before propagating an interrupt across Thread.start.

    Python Thread.ident can still be None after native creation. CPython's
    native handle, when available, records creation independently of bootstrap;
    older threading implementations retain the object in their locked launch
    registry until the started Event is set. Neither test relies on ident.

    On the main thread, defer SIGINT only through this handshake. This prevents
    interruption between registry insertion and native creation on older Python.
    Writers wait on the group release Event meanwhile. Restore and invoke the
    original handler after recording the outcome, before releasing any work.
    """
    thread = job['thread']
    registry_lock = getattr(threading, '_active_limbo_lock', None)
    registry = getattr(threading, '_limbo', None)
    require(registry_lock is not None and isinstance(registry, dict)
            and hasattr(thread, '_started'), 'threading launch evidence is unavailable')
    pending = []
    original_handler = None
    defer = threading.current_thread() is threading.main_thread()
    if defer:
        original_handler = signal.getsignal(signal.SIGINT)
        require(callable(original_handler) or original_handler in (signal.SIG_DFL, signal.SIG_IGN),
                'SIGINT handler cannot be restored safely')
        signal.signal(signal.SIGINT, lambda signum, frame: pending.append((signum, frame)))
    try:
        job['unstarted'] = False  # Uncertain launch state must retain ownership.
        thread.start()
    finally:
        # start() has returned/unwound: no native-creation call can run later in
        # this caller. A native handle distinguishes even pre-bootstrap threads.
        try:
            with registry_lock:
                handle = getattr(thread, '_os_thread_handle', getattr(thread, '_handle', None))
                acknowledged = thread._started.is_set()
                if handle is not None:
                    job['unstarted'] = not acknowledged and handle.ident == 0
                else:
                    job['unstarted'] = not acknowledged and thread not in registry
                if job['unstarted']:
                    registry.pop(thread, None)  # Clear only a proven abandoned registration.
        finally:
            if defer:
                signal.signal(signal.SIGINT, original_handler)
                if pending:
                    if callable(original_handler):
                        original_handler(*pending[0])
                    elif original_handler == signal.SIG_DFL:
                        signal.raise_signal(signal.SIGINT)
    require(not job['unstarted'], 'Thread.start returned without a native launch')


def _owned_clients(requests, mode, invoke, persist, timeout, errors):
    """Retain actual client writers through persistence AND native thread exit.

    The work deadline cancels the clients and remains a failure. The existing
    two-second settlement allowance is diagnostic, not permission to return
    with live writers. Python/OS stalls can delay this failed call: finalization
    waits for actual settlement rather than publishing a mutable failure index.
    This local launcher intentionally does not change the shared collector API.
    """
    cancel = threading.Event()
    release = threading.Event()
    jobs = []
    problem = None

    def worker(job):
        try:
            release.wait()
            require(not cancel.is_set(), 'client cancelled before invocation')
            row = invoke(job['request'], cancel)
            persist(row)
        except BaseException as error:
            errors[job['request']['id']] = f'{type(error).__name__}: {error}'
            cancel.set()
        finally:
            job['done'].set()

    def await_work(job, deadline):
        while not job['done'].wait(min(.02, max(0, deadline-time.monotonic()))):
            if time.monotonic() >= deadline:
                errors[job['request']['id']] = 'client exceeded capture deadline'
                cancel.set()
                raise TimeoutError('owned worker-failure client deadline exceeded')
        if time.monotonic() > deadline:
            errors[job['request']['id']] = 'client completion observed after capture deadline'
            cancel.set()
            raise TimeoutError('owned worker-failure client deadline exceeded')

    try:
        for request in requests:
            if mode == 'serial':
                release.clear()  # Each writer waits for its own launch outcome.
            # Until the protected start helper is entered, no launch is possible.
            job = {'request': request, 'done': threading.Event(), 'unstarted': True}
            thread = threading.Thread(target=worker, args=(job,), name='serving-client-' + request['id'])
            job['thread'] = thread
            jobs.append(job)  # Ownership exists before the native launch.
            _start_owned_client(job)
            if mode == 'serial':
                release.set()
                await_work(job, time.monotonic()+timeout)
                require(not errors, 'serial owned client failed')
        release.set()
        deadline = time.monotonic()+timeout
        for job in jobs:
            await_work(job, deadline)
    except BaseException as error:
        problem = error
        cancel.set()
    finally:
        release.set()
        settlement_deadline = time.monotonic()+2
        for job in jobs:
            if job['unstarted']:
                continue
            while not job['done'].is_set():
                try:
                    job['done'].wait(.02)
                except BaseException as error:
                    problem = problem or error
                    cancel.set()
                if time.monotonic() >= settlement_deadline:
                    errors['serving-client-' + job['request']['id']] = 'client exceeded bounded settlement; waited for actual writer exit'
                    cancel.set()
            # A done Event covers persistence, but not the final Thread exit.
            # join() on older CPython may be interrupted before native exit;
            # the worker's own done event remains a separate authority above.
            while job['thread'].is_alive():
                try:
                    job['thread'].join(.02)
                except BaseException as error:
                    problem = problem or error
                    cancel.set()
                if time.monotonic() >= settlement_deadline:
                    errors['serving-client-' + job['request']['id']] = 'client exceeded bounded settlement; waited for actual writer exit'
                    cancel.set()
        require(not any(job['thread'].is_alive() for job in jobs), 'owned client remained alive')
    if problem is not None:
        raise problem
    require(not errors, 'owned client group failed')


def collect_worker_failure_cell(required, program, *, server, output):
    validate_program(required,program)
    required,program=json_object(encoded(required)),json_object(encoded(program))
    roles=validate_program(required,program)
    require(isinstance(server,OwnedServer),"worker failure capture needs actual OwnedServer")
    owner=server.identity;initial=server.receipt()
    identity={"pid":owner.pid,"start_identity":initial.get("boot_id",owner.identity_source)+":"+owner.start_time}
    require(identity==program["server_identity"] and initial.get("state")=="ready" and not initial.get("stop")
            and not initial.get("errors"),"server is not the selected live ready owner")
    launch={"argv":initial["argv"],"cwd":initial["cwd"],"env":dict(server._config["env"]),
            "timeouts":initial["timeouts"],"output_path":str(server.output_path)}
    fault_environment(launch)
    require(program["identities"]["server_binary"]==initial["argv"][0],"server artifact differs from launch")
    store=EvidenceStore(output);lock=threading.RLock();started=time.monotonic_ns()
    deadline=time.monotonic()+program["timing"]["overall_s"]
    state={"schema":"memra-worker-failure-capture-v1","state":"collecting","qualification":False,
        "clock":"monotonic_ns","started_ns":started,"finished_ns":None,"required":store.obj(required),"program":store.obj(program),
        "identities_before":{},"identities_after":{},"process_observations":[],"listeners":[],"observations":[],"probes":[],
        "request_invocations":[],"prefixes":{},"prefix_confirmed_ns":None,"log_baseline":None,"log_descriptor":None,
        "log_chunks":[],"log_observations":[],"log_final":None,"log_observed":None,"lifecycle":None,"stop_call":None,
        "wire_accounting":None,"facts":None,"errors":[],"client_errors":{},"request_denominator":[],
        "unattempted_ids":[],"unobserved_invoked_ids":[]}
    rows={};probes=[];prefixes={};victim_done={r['id']:threading.Event() for r in roles['victim']}
    headers={};invoked=set();cancellations=set();abort=threading.Event()
    group_thread=None;group_done=threading.Event();group_release=threading.Event()
    group_job=None;group_started=None;pinned=None

    def save():
        with lock:store.index(state)

    def remaining():
        left=deadline-time.monotonic();require(left>0,"overall worker-failure capture deadline exceeded");return left

    def live():
        p=process_identity(owner.pid);r=server.receipt()
        require(p is not None and p.key==owner.key and p.ppid==owner.ppid and p.pgid==owner.pgid
                and p.state not in ('Z','X','x') and r.get('server_exit') is None and r.get('state')!='finished',
                "owned server exited/replaced during worker recovery")
        return p

    def process_snapshot(label):
        begin=time.monotonic_ns();p=live()
        state['process_observations'].append(store.obj({'label':label,'started_ns':begin,'finished_ns':time.monotonic_ns(),
                                                       'receipt':server.receipt(),'observed_identity':asdict(p)}))

    def ownership(label):
        until=time.monotonic()+min(3,remaining())
        while True:
            live()
            try:
                proof=prove_listener(owner,**program['endpoint'],timeout=max(.001,until-time.monotonic()))
                state['listeners'].append(store.obj({'label':label,'proof':asdict(proof)}));return
            except ListenerOwnershipError as error:
                state['listeners'].append(store.obj({'label':label,'error':str(error),'observed_ns':time.monotonic_ns()}))
                if time.monotonic()>=until:raise
                abort.wait(min(.02,max(0,until-time.monotonic())))

    def log_read():
        try:meta,offset,delta=pinned.read()
        except _LogFailure as error:
            state['log_observations'].append(store.obj({**error.metadata,'error':str(error),
                **({'observed_raw':store.put(error.observed)} if error.observed is not None else {})}));raise
        if delta:state['log_chunks'].append({'offset':offset,'bytes':len(delta),'raw':store.put(delta)})
        state['log_observations'].append(store.obj(meta))
        return pinned.raw

    def http(name,path,body,method,cancel,observer=None):
        live();wall=min(program['http']['wall_timeout'],remaining())
        with lock:
            invoked.add(name)
            state['request_invocations'].append(store.obj({'id':name,'method':method,'path':path,
                'body':store.put(body),'started_ns':time.monotonic_ns()}))
        result=capture_request(request_id=name,method=method,path=path,body=body,
            headers={'X-Request-Id':name,'Content-Type':'application/json'},cancel_event=cancel,observer=observer,
            **program['endpoint'],**{**program['http'],'wall_timeout':wall})
        result.update(server_identity=dict(identity),method=method,path=path)
        return result

    def probe(path):
        require(len(probes)<MAX_PROBES,'probe budget exhausted')
        name=PROBE_PREFIX+str(len(probes)+1)
        row=http(name,path,b'', 'GET',abort)
        probes.append(row);state['probes'].append(store.observation(row));save()
        return probe_facts(row,required['scope']['model'])

    def observe(value):
        # Bounded immutable snapshots only; no IO or waiting in the HTTP callback.
        name=value['id']
        with lock:
            if value['event']=='headers':headers[name]=value['status']
            elif value['event']=='first_body':
                prefixes[name]={'observed_ns':value['observed_ns'],'end_offset':value['end_offset'],'body':value['data']}

    def persist(row):
        with lock:
            if row['id'] not in rows:
                rows[row['id']]=row;state['observations'].append(store.observation(row));save()

    def invoke(request,cancel):
        with lock:cancellations.add(cancel)
        try:
            require(not abort.is_set(),'worker failure capture cancelled before request')
            if request['role']=='trigger':
                limit=min(deadline,group_started+program['timing']['prefix_timeout_s'])
                while True:
                    require(not abort.is_set() and not cancel.is_set(),'driver cancelled while waiting for victims')
                    with lock:ready=all(r['id'] in prefixes for r in roles['victim'])
                    if ready:break
                    require(not any(done.is_set() for done in victim_done.values()),'victim ended without an admitted prefix')
                    require(time.monotonic()<limit,'victim-prefix deadline exceeded');cancel.wait(.01)
                with lock:
                    for r in roles['victim']:
                        require(not victim_done[r['id']].is_set() and headers.get(r['id'])==200,'victim ended/refused before driver')
                        require_inflight_prefix(r,prefixes[r['id']]['body'])
                    state['prefix_confirmed_ns']=time.monotonic_ns()
            result=http(request['id'],request['path'],encoded(request['payload']),'POST',cancel,
                        observe if request['role']=='victim' else None)
            return result
        except BaseException:
            cancel.set();raise
        finally:
            if request['id'] in victim_done:victim_done[request['id']].set()

    def pressure():
        try:
            group_release.wait()
            require(not abort.is_set(), 'pressure cancelled before launch completed')
            _owned_clients([r for r in program['requests'] if r['role']!='recovery'],'concurrent',invoke,persist,
                           min(program['http']['wall_timeout']+program['timing']['prefix_timeout_s'],remaining()),state['client_errors'])
        except BaseException as error:
            with lock:state['errors'].append(f'pressure clients: {type(error).__name__}: {error}')
        finally:group_done.set()

    save()
    try:
        process_snapshot('before');ownership('before')
        state['identities_before']={k:file_identity(p) for k,p in program['identities'].items()}
        pinned=_PinnedLog(server.output_path);state['log_descriptor']=store.obj(pinned.descriptor)
        raw=log_read();require(not raw or raw.endswith(b'\n'),'unfinished prelaunch log line')
        state['log_baseline']=store.put(raw)
        require(b'[worker] PANIC' not in raw and b'[worker] respawn attempt' not in raw,'worker fault already in baseline')
        h=probe('/health');r=probe('/readyz')
        require(h['status']==r['status']==200 and h['generation']==r['generation']==0,'initial worker is not healthy generation0')
        group_started=time.monotonic();group_thread=threading.Thread(target=pressure,name='wf-pressure')
        group_job={'thread':group_thread,'unstarted':True}
        _start_owned_client(group_job)
        group_release.set()
        dead_seen={};fault_at=None;driver_id=roles['trigger'][0]['id'];recovered=False
        while True:
            require(not state['errors'] and not state['client_errors'],'pressure capture failed')
            remaining()
            with lock:driver_invoked=driver_id in invoked
            if not driver_invoked:require(time.monotonic()-group_started<program['timing']['prefix_timeout_s'],'prefix trigger timeout')
            elif fault_at is None:
                # The bound below is conservatively from group start; the predicate
                # rechecks the exact actual driver start timestamp from its capture.
                require(time.monotonic()-group_started<program['timing']['prefix_timeout_s']+program['timing']['fault_timeout_s'],'worker fault not observed in time')
            else:require(time.monotonic()-fault_at<program['timing']['recovery_timeout_s'],'worker did not recover before deadline')
            for path in ('/health','/readyz'):
                f=probe(path)
                if f['status']==503 and f['generation']==0 and f['phase']=='dead' and f['detail']==DETAIL:
                    dead_seen[path]=f
                    if fault_at is None:fault_at=time.monotonic()
                if f['status']==200 and f['generation']==1 and path=='/readyz':recovered=True
            if recovered and group_done.is_set():break
            abort.wait(min(program['timing']['poll_s'],remaining()))
        require(set(dead_seen)=={'/health','/readyz'} and not state['errors'],'missing complete worker-fault observation')
        # Fresh brackets after ALL pressure clients (including buffered errors) end.
        for path in ('/health','/readyz'):
            f=probe(path);require(f['status']==200 and f['generation']==1,'worker not ready before recovery')
        _owned_clients(roles['recovery'],'serial',invoke,persist,min(remaining(),program['http']['wall_timeout']),state['client_errors'])
        for path in ('/health','/readyz'):
            f=probe(path);require(f['status']==200 and f['generation']==1,'worker failed again after recovery')
        ownership('after');process_snapshot('after')
        state['state']='captured'
    except BaseException as error:
        state['state']='failed';state['errors'].append(f'{type(error).__name__}: {error}')
    finally:
        if state['state']!='captured':
            abort.set()
            with lock:
                for cancel in cancellations:cancel.set()
        group_release.set()
        if group_job is not None and not group_job['unstarted']:
            # pressure signals done only after its owned client writers have
            # persisted and actually exited, including failed deadline overruns.
            while not group_done.wait(.02):pass
            group_thread.join()
        begin=time.monotonic_ns();stop_error=None
        reason='worker_failure_capture_complete' if state['state']=='captured' else 'worker_failure_capture_error'
        try:lifecycle=server.close(reason=reason)
        except BaseException as error:
            lifecycle=server.receipt();stop_error=f'{type(error).__name__}: {error}';state['errors'].append('owned close: '+stop_error)
        state['stop_call']=store.obj({'started_ns':begin,'finished_ns':time.monotonic_ns(),'reason':reason,'error':stop_error})
        state['lifecycle']=store.obj(lifecycle)
        with lock:
            for name,value in prefixes.items():state['prefixes'][name]=store.obj({**value,'body':store.put(value['body'])})
        if pinned is not None:
            try:
                final=log_read();state['log_final']=state['log_observed']=store.put(final)
            except BaseException as error:state['errors'].append('log postcheck: '+str(error));state['log_observed']=store.put(pinned.raw)
            finally:pinned.close()
        try:state['identities_after']={k:file_identity(p) for k,p in program['identities'].items()}
        except BaseException as error:state['errors'].append('artifact postcheck: '+str(error))
        request_ids={r['id'] for r in program['requests']}
        state['request_denominator']=[{'id':r['id'],'role':r['role'],'status':'captured' if r['id'] in rows else
            'invoked_without_observation' if r['id'] in invoked else 'not_invoked'} for r in program['requests']]
        state['unattempted_ids']=[r['id'] for r in program['requests'] if r['id'] not in invoked]
        state['unobserved_invoked_ids']=sorted((invoked & request_ids)-rows.keys())
        try:
            observed=[rows[r['id']] for r in program['requests'] if r['id'] in rows]
            state['wire_accounting']=store.obj(account_attempts([r for r in program['requests'] if r['id'] in rows],observed))
            require(set(rows)==request_ids,'incomplete worker-failure attempt denominator')
            require(state['identities_before']==state['identities_after'],'artifact bytes changed')
            facts=evaluate_worker_failure_cell(required,program,launch=launch,attempts=observed,probes=probes,prefixes=prefixes,
                baseline_log=(store.root/state['log_baseline']['path']).read_bytes(),log=(store.root/state['log_final']['path']).read_bytes())
            state['facts']=store.obj(facts)
            _drain_lifecycle({'server_identity':identity,'stop_reason':'worker_failure_capture_complete',
                'drain_timeout_ns':int(initial['timeouts']['drain']*1e9)},lifecycle)
        except BaseException as error:state['errors'].append(f'worker failure validation: {type(error).__name__}: {error}')
        state['finished_ns']=time.monotonic_ns()
        if time.monotonic()>deadline:state['errors'].append('overall worker-failure deadline exceeded')
        if state['errors'] or state['client_errors']:state['state']='failed'
        save()
    return state
