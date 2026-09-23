"""Replay one worker-failure capture against external immutable expectations.

Authenticity/source/build/model/controller/physical lease binding remains outer.
This adapter checks every declared byte, full request/probe census, actual launch,
owned process/listener/log brackets and clean closure, then reruns the predicate.
It never upgrades a failed capture or returns native/release qualification.
"""
import hashlib

from serving_capture import encoded
from serving_cancel_evidence import _Bundle, _process, _log_capture, _span
from serving_completion import _keys, _same
from serving_drain_evidence import validate_launch
from serving_evidence import _listener, _SHA
from serving_policy import _drain_lifecycle, _seconds
from serving_release import account_attempts, json_object, require
from serving_worker_failure import validate_program, fault_environment, evaluate_worker_failure_cell

ROOT_FIELDS=frozenset('schema state qualification clock started_ns finished_ns program required identities_before identities_after '
    'process_observations listeners observations probes request_invocations prefixes prefix_confirmed_ns log_baseline log_descriptor '
    'log_chunks log_observations log_final log_observed lifecycle stop_call wire_accounting facts errors client_errors '
    'request_denominator unattempted_ids unobserved_invoked_ids payloads'.split())


def read_worker_failure_capture(capture_bytes,evidence_reader,*,expected_capture_sha256,
                                expected_required,expected_program,expected_server,expected_identities,expected_log_identity):
    require(type(capture_bytes) is bytes and len(capture_bytes)<=8*1024*1024
            and type(expected_capture_sha256) is str and _SHA.fullmatch(expected_capture_sha256)
            and hashlib.sha256(capture_bytes).hexdigest()==expected_capture_sha256,'worker failure index differs from external digest')
    roles=validate_program(expected_required,expected_program);fault_environment(expected_server)
    capture=json_object(capture_bytes);_keys(capture,ROOT_FIELDS,'worker failure capture')
    require(capture['schema']=='memra-worker-failure-capture-v1' and capture['state']=='captured'
            and capture['qualification'] is False and capture['clock']=='monotonic_ns'
            and capture['errors']==[] and capture['client_errors']=={},'worker failure capture did not complete')
    bounds=_span(capture)
    require(bounds[1]-bounds[0]<=int(expected_program['timing']['overall_s']*1e9),'overall capture deadline exceeded')
    bundle=_Bundle(capture['payloads'],evidence_reader)
    require(_same(bundle.obj(capture['program']),expected_program) and _same(bundle.obj(capture['required']),expected_required),
            'captured program or required cell differs')
    require(type(expected_identities) is dict and expected_identities.keys()==expected_program['identities'].keys(),
            'external artifact scope differs')
    for item in expected_identities.values():
        _keys(item,{'bytes','sha256'},'artifact identity')
        require(type(item['bytes']) is int and item['bytes']>=0 and type(item['sha256']) is str and _SHA.fullmatch(item['sha256']),
                'invalid external artifact identity')
    require(_same(capture['identities_before'],expected_identities) and _same(capture['identities_after'],expected_identities),
            'source/binary/model artifacts changed')
    lifecycle=bundle.obj(capture['lifecycle']);validate_launch(lifecycle,expected_server)
    require(expected_program['identities']['server_binary']==expected_server['argv'][0],'launched server is not bound binary')
    identity=expected_program['server_identity']
    descriptor,raw,reads=_log_capture(capture,bundle,expected_log_identity,bounds)
    processes=_process(bundle.objects(capture['process_observations'],2),identity,expected_server,descriptor,bounds)
    baseline=bundle.raw(capture['log_baseline'])
    require(reads and baseline==raw[:reads[0]['bytes']] and reads[0]['complete_bytes']==len(baseline),
            'log baseline differs from first complete read')
    cleanup=_drain_lifecycle({'server_identity':identity,'stop_reason':'worker_failure_capture_complete',
        'drain_timeout_ns':int(expected_server['timeouts']['drain']*1e9)},lifecycle)
    stop=bundle.obj(capture['stop_call']);_keys(stop,{'started_ns','finished_ns','reason','error'},'owned stop call')
    stop_span=_span(stop)
    require(stop['reason']=='worker_failure_capture_complete' and stop['error'] is None
            and processes[1]['finished_ns']<=stop_span[0]<=cleanup['stop_requested_ns']
            and cleanup['cleanup_finished_ns']<=stop_span[1]<=reads[-1]['started_ns'], 'stop/log closure chronology differs')
    keys=('pid','ppid','pgid','start_time','identity_source')
    require(_same({k:lifecycle['server'][k] for k in keys},{k:processes[0]['receipt']['server'][k] for k in keys}),
            'server ownership differs at cleanup')
    requests=expected_program['requests'];ids=[r['id'] for r in requests]
    require(capture['unattempted_ids']==[] and capture['unobserved_invoked_ids']==[]
            and _same(capture['request_denominator'],[{'id':r['id'],'role':r['role'],'status':'captured'} for r in requests]),
            'required request denominator is incomplete')
    require(type(capture['observations']) is list and len(capture['observations'])==len(ids),'request capture census differs')
    attempts=[bundle.observation(ref,expected_program['http']['max_body_bytes']) for ref in capture['observations']]
    accounting=account_attempts(requests,attempts)
    index={r['id']:r for r in attempts};attempts=[index[name] for name in ids]  # completion order is not program order
    require(_same(accounting,bundle.obj(capture['wire_accounting'])),'wire accounting differs')
    require(type(capture['probes']) is list and len(capture['probes'])<=512,'invalid probe census')
    probes=[bundle.observation(ref,expected_program['http']['max_body_bytes']) for ref in capture['probes']]
    all_rows=attempts+probes;by_id={r['id']:r for r in all_rows}
    require(len(by_id)==len(all_rows),'probe/request IDs collide')
    invocations=bundle.objects(capture['request_invocations'],544)
    require(len(invocations)==len(all_rows) and {r.get('id') for r in invocations}==set(by_id), 'invocation census differs')
    planned={r['id']:r for r in requests}
    for item in invocations:
        _keys(item,{'id','path','method','body','started_ns'},'HTTP invocation')
        row=by_id[item['id']];req=planned.get(item['id']);start,end=_span(row)
        require(type(item['started_ns']) is int and bounds[0]<=item['started_ns']<=start<=end<=stop_span[0],
                'request/probe occurs outside live capture')
        require(row['server_identity']==identity and item['method']==row['method'] and item['path']==row['path'],
                'HTTP owner/path/method mismatch')
        require(item['method']==('POST' if req else 'GET') and (not req or item['path']==req['path'])
                and bundle.raw(item['body'])==(encoded(req['payload']) if req else b''),'request payload or invocation changed')
        require(row.get('transport_error') is None,'transport/observer interruption cannot supply recovery')
        body=row['body'];chunks=row.get('chunks')
        require(type(chunks) is list and chunks,'missing captured body chunks')
        offset=0;previous=start
        for chunk in chunks:
            _keys(chunk,{'end_offset','observed_ns'},'HTTP chunk')
            require(type(chunk['end_offset']) is int and offset<chunk['end_offset']<=len(body)
                    and type(chunk['observed_ns']) is int and previous<=chunk['observed_ns']<=end,'body chunk order differs')
            offset,previous=chunk['end_offset'],chunk['observed_ns']
        require(offset==len(body) and row.get('first_body_byte_ns')==chunks[0]['observed_ns'],'body byte/timing census differs')
        headers=row.get('headers')
        require(type(headers) is list and all(type(p) is list and len(p)==2 and all(type(v) is str for v in p) for p in headers),'invalid header capture')
        lengths=[v for k,v in headers if k.lower()=='content-length'];transfer=[v for k,v in headers if k.lower()=='transfer-encoding']
        require(len(lengths)<=1 and len(transfer)<=1 and not(lengths and transfer),'ambiguous HTTP framing')
        if lengths:require(lengths[0].isascii() and lengths[0].isdecimal() and len(lengths[0])<=10 and int(lengths[0])==len(body),'Content-Length differs')
        if transfer:require(transfer[0].strip().lower()=='chunked','unknown transfer coding')
    prefixes={}
    require(type(capture['prefixes']) is dict and set(capture['prefixes'])=={r['id'] for r in roles['victim']},'victim prefix census differs')
    for name,ref in capture['prefixes'].items():
        item=dict(bundle.obj(ref));item['body']=bundle.raw(item['body']);prefixes[name]=item
    confirmed=capture['prefix_confirmed_ns'];driver=index[roles['trigger'][0]['id']]
    require(type(confirmed) is int and max(p['observed_ns'] for p in prefixes.values())<=confirmed<=driver['started_ns']
            and driver['started_ns']-min(index[r['id']]['started_ns'] for r in roles['victim'])<=int(expected_program['timing']['prefix_timeout_s']*1e9),
            'driver did not follow observed victims within prefix budget')
    facts=evaluate_worker_failure_cell(expected_required,expected_program,launch=expected_server,attempts=attempts,probes=probes,
                                      prefixes=prefixes,log=raw,baseline_log=baseline)
    require(_same(facts,bundle.obj(capture['facts'])),'stored worker-recovery facts differ from raw evidence')
    proofs={}
    for item in bundle.objects(capture['listeners'],1024):
        label=item.get('label');require(label in ('before','after'),'unknown listener boundary')
        if 'proof' in item:
            _keys(item,{'label','proof'},'listener proof')
            require(label not in proofs,'duplicate listener boundary');proofs[label]=_listener(item['proof'],identity,expected_program['endpoint'])
        else:
            _keys(item,{'label','error','observed_ns'},'listener retry')
            require(type(item['error']) is str and item['error'] and type(item['observed_ns']) is int
                    and bounds[0]<=item['observed_ns']<=bounds[1] and label not in proofs,'invalid listener retry')
    require(set(proofs)=={'before','after'} and processes[0]['finished_ns']<=proofs['before'][0]
            and proofs['before'][1]<=reads[0]['started_ns']<=reads[0]['finished_ns']<probes[0]['started_ns']
            and max(r['finished_ns'] for r in all_rows)<=proofs['after'][0]<=proofs['after'][1]<=processes[1]['started_ns'],
            'process/listener/log snapshots do not bracket complete live HTTP census')
    return {'scope':'worker_failure_capture_consistency_only','qualification':False,'required':expected_required,
            'program':expected_program,'capture_sha256':expected_capture_sha256,'attempts':attempts,'probes':probes,
            'facts':facts,'lifecycle':lifecycle,'verified_payloads':len(bundle.blobs)}
