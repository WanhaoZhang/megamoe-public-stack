"""CPU-only packaging of completed diagnostic runs; original data stay remote.

Select original (unmodified) NPZ files covering each rank/phase/recompute plus
positive replica coverage. Audit ALL records before packaging the selection.
This never changes a recorded execution or the frozen training source tree.
"""
from pathlib import Path
import hashlib,json,sys,tarfile
import numpy as np


def service_probe(meta,data):
    layout=meta['layout']
    if meta['phase']!='backward' or layout.get('version')!=9:return None
    def pairs(kind):
        start,size=layout['extra_ranges'][str(kind)]
        ids=data['stage_indices']
        return data['stage_pairs'][(ids>=start)&(ids<start+size),:2]
    release,service=pairs(6),pairs(20)
    if not len(release):return None
    last=int(release[:,1].max());intervals=[]
    for first,end in sorted(service.tolist()):
        if intervals and first<=intervals[-1][1]:intervals[-1][1]=max(intervals[-1][1],end)
        else:intervals.append([first,end])
    duration=sum(end-first for first,end in intervals)
    overlap=sum(max(0,min(end,last)-first) for first,end in intervals)
    return dict(rank=meta['rank'],invocation=meta['invocation'],service_sample_count=len(service),
        release_sample_count=len(release),sampled_service_union_cycles=duration,
        sampled_service_before_last_local_release_cycles=overlap,
        fraction=overlap/duration if duration else None)


def native_phase_probe(meta,data):
    if meta['phase']!='backward':return None
    # AscendTimerV2.hpp: 10 fixed + 20*64 dynamic pairs, padded to16 words.
    # Slot0 is original KERNEL on each of96 interleaved AIC/AIV lanes.
    raw=data['original_backward'];stride=2592
    if raw.size!=251904:raise RuntimeError('Unexpected original backward Timer layout')
    kernels=raw[:96*stride].reshape(96,stride)[:,:2]
    kernels=kernels[(kernels[:,0]>0)&(kernels[:,1]>=kernels[:,0])]
    if not len(kernels):return None
    start=int(kernels[:,0].min());end=int(kernels[:,1].max());span=end-start
    phases={}
    indices=data['stage_indices'];pairs=data['stage_pairs']
    for kind in (15,16,28,29,41,42,43,19,20):
        first,size=meta['layout']['extra_ranges'][str(kind)]
        intervals=[]
        for begin,finish in sorted(pairs[(indices>=first)&(indices<first+size),:2].tolist()):
            if intervals and begin<=intervals[-1][1]:intervals[-1][1]=max(intervals[-1][1],finish)
            else:intervals.append([begin,finish])
        ticks=sum(finish-begin for begin,finish in intervals)
        phases[str(kind)]=dict(union_cycles=ticks,fraction_of_kernel_span=ticks/span if span else None)
    return dict(rank=meta['rank'],invocation=meta['invocation'],kernel_span_cycles=span,phases=phases)


def package(folder,bundle):
    from_inference=[];selected=set();seen=set();inventory=[];coverage={};service_probes=[];native_phases=[]
    for path in sorted((folder/'timer').glob('*.npz')):
        with np.load(path,allow_pickle=False) as data:
            meta=json.loads(str(data['metadata']));rank=meta['rank'];home=meta['layout']['home']
            copies=data['copies'];quota=data['source_quota'];actual=data['execution_counts']
            ep,slots=copies.shape;routed=quota.astype(np.int64).sum(axis=0)
            predicted=np.zeros((ep,home+slots),dtype=np.int64)
            for dest in range(ep):
                predicted[dest,:home]=routed[dest*home:(dest+1)*home,dest]
                for slot,expert in enumerate(copies[dest]):
                    if expert>=0 and expert//home!=dest:predicted[dest,home+slot]=routed[expert,dest]
            exact=bool(np.array_equal(predicted[rank],actual) and predicted.sum()==quota.sum())
            from_inference.append(exact)
            positive=int(np.count_nonzero((copies[rank]>=0)&(actual[home:]>0)))
            row=coverage.setdefault(str(rank),dict(calls=0,words=0,stage_records=0,active_replicas=0,phases=set(),replay=False))
            row['calls']+=1;row['words']+=int(np.count_nonzero(data['original_'+meta['phase']]))
            row['stage_records']+=len(data['stage_pairs']);row['active_replicas']+=positive
            row['phases'].add(meta['phase']);row['replay']|=bool(meta['recompute'])
            probe=service_probe(meta,data)
            if probe is not None:service_probes.append(probe)
            probe=native_phase_probe(meta,data)
            if probe is not None:native_phases.append(probe)
            keys=[]
            if meta['invocation'][0]==2:keys.append((rank,meta['phase'],bool(meta['recompute'])))
            if positive:keys.append((rank,'positive_replica'))
            if any(key not in seen for key in keys):
                selected.add(path);seen.update(keys)
            inventory.append(dict(file=str(path.relative_to(folder)),bytes=path.stat().st_size,
                                  sha256=hashlib.sha256(path.read_bytes()).hexdigest(),count_inference_exact=exact))
    for row in coverage.values():row['phases']=sorted(row['phases'])
    manifest=dict(local_timer_scope='representative original NPZ files; full raw data retained at remote_source',
        remote_source=str(folder),original_records=len(inventory),selected_records=len(selected),
        all_record_coverage=coverage,plan_count_inference=dict(passed=bool(from_inference) and all(from_inference),
        records=len(from_inference),mismatches=from_inference.count(False)),
        owner_service_probe=dict(records=service_probes,
            scope='all original backward records; rank-local cycles; column0 row-bin samples, not all tiles; no cross-rank or complete-update exposure claim'),
        native_phase_probe=dict(records=native_phases,
            scope='all backward records; original KERNEL slot0 span and coordinate interval unions on the same device; overlapping phases not additive; cycles not calibrated to milliseconds'),
        selected_files=sorted(str(path.relative_to(folder)) for path in selected),inventory=inventory)
    (folder/'collection-scope.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with tarfile.open(bundle,'w:gz') as archive:
        for path in sorted(folder.rglob('*')):
            if path.is_file() and path.suffix!='.pt' and (path.suffix!='.npz' or path in selected):
                archive.add(path,arcname=str(path.relative_to(folder.parent)),recursive=False)
    print(json.dumps({key:manifest[key] for key in ('original_records','selected_records','plan_count_inference')}))


if __name__=='__main__':package(Path(sys.argv[1]),Path(sys.argv[2]))
