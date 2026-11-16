"""Coarse CPU scheduling reference for the measured S9 GM contract.

Inputs are supplied phase costs/release estimates, NOT a calibrated model.
This module neither selects an online plan nor predicts verified step latency.
It captures per-expert SG readiness, early shared-DW1, static parity consumers,
full dW tiles and the existing final barrier. No free overlap is assumed.
"""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class ExpertCost:
    dh: float
    swiglu: float
    dw2: float
    dx: float
    dw1: float
    combine: float=0.
    w2_ready: float=0.
    w1_ready: float=0.

@dataclass(frozen=True)
class RankCost:
    init: float=0.
    xp_retired: float=0.
    even_empty_clear: float=0.
    odd_empty_clear: float=0.
    exit: float=0.


def simulate(destination, costs, rank_costs, *, forward_finish=None,
             owner_service=None, owner_scan=0., early_shared_dw1=True,
             dh_runahead=False, producer_home_first=False, epilogue_home_first=False,
             aic_cores=32,aiv_workers=64,dw1_tiles=896,dw2_tiles=448,
             replica_slots=None,completion_experts=32):
    """One checkpointed MoE layer with fixed owner rank=e//home.

    Only nonempty logical expert costs are required. Owner service callback
    takes(projection, remote_contributors, has_home_contribution), returning
    one complete128x256 output-tile service time; never scale by token share.
    Wave interpolation is an EXPLICIT approximation of tile release, not a
    measured completion trace. AIC is one shared resource for all four GEMMs.
    Separate odd/even owner lanes follow the baseline tail=0,ratio=1 contract.
    """
    e=len(destination);p=len(rank_costs)
    if not p or not e or e%p or aiv_workers!=2*aic_cores or e//p>32:
        raise ValueError('Unsupported fixed-owner topology')
    if (dw1_tiles%aiv_workers or dw2_tiles%aiv_workers or dw1_tiles<=0 or dw2_tiles<=0):
        raise ValueError('This reference requires full static tile groups')
    if any(len(row)!=p or any(not isinstance(n,int) or n<0 for n in row) for row in destination):
        raise ValueError('Nonnegative integer destination counts required')
    home=e//p;shared=[any(row[r]>0 for r in range(p) if r!=x//home) for x,row in enumerate(destination)]
    any_copy=any(shared)
    slots=home if replica_slots is None else replica_slots
    if not isinstance(slots,int) or slots<0 or completion_experts not in (32,64):
        raise ValueError('Invalid physical slot capacity/completion-mask width')
    if any(sum(destination[x][r]>0 for x in range(e) if x//home!=r)>slots for r in range(p)):
        raise ValueError('Replica declaration exceeds physical slot capacity')
    # Match the actual device extent, not Host allocation or occupied slots.
    extent=home+slots if any_copy else home
    early_active=early_shared_dw1 and any_copy and p in (2,4,8) and extent<=completion_experts
    if owner_service is None:owner_service=lambda projection,remote,local:0.
    def checked(value):
        if not math.isfinite(value) or value<0:raise ValueError('Nonnegative finite phase/release cost required')
        return float(value)
    checked(owner_scan)
    for rank in rank_costs:
        for value in vars(rank).values():checked(value)
    for value in costs.values():
        for number in vars(value).values():checked(number)
    releases={};records=[]
    for r,rc in enumerate(rank_costs):
        active=[x for x in range(e) if destination[x][r]>0]
        order=sorted(active,key=lambda x:(not shared[x],x)) if any_copy else active
        home_order=sorted(active,key=lambda x:(x//home!=r,x))
        prod=home_order if producer_home_first else order
        epi=home_order if epilogue_home_first else order
        dh=rc.init if dh_runahead else max(rc.init,rc.xp_retired)
        ready={};nodes={}
        for x in prod:
            c=costs[r,x];start=max(dh,c.w2_ready);dh=start+c.dh;ready[x]=dh
            nodes[x]=dict(dh=(start,dh))
        sg=max(rc.init,rc.xp_retired)
        for x in epi:
            start=max(sg,ready[x]);sg=start+costs[r,x].swiglu;nodes[x]['swiglu']=(start,sg)
        aic=max(dh,rc.xp_retired);combine=sg
        for x in order:
            c=costs[r,x];start=max(aic,nodes[x]['swiglu'][1]);aic=start+c.dw2
            releases[r,x,1]=(start,aic);nodes[x]['dw2']=(start,aic)
            if early_active and shared[x]:
                start=aic;aic+=c.dw1;releases[r,x,0]=(start,aic);nodes[x]['dw1']=(start,aic)
            start=max(aic,c.w1_ready);aic=start+c.dx;nodes[x]['dx']=(start,aic)
            start=max(combine,aic);combine=start+c.combine;nodes[x]['combine']=(start,combine)
        for x in order:
            if (r,x,0) not in releases:
                start=aic;aic+=costs[r,x].dw1;releases[r,x,0]=(start,aic);nodes[x]['dw1']=(start,aic)
        records.append(dict(rank=r,nodes=nodes,aic_end=aic,sg_end=sg,combine_end=combine,
                            local_end=max(aic,combine),owner_end=[]))
    for r,(rc,record) in enumerate(zip(rank_costs,records)):
        if not any_copy:
            # Kernel chooses home tiling and skips the entire owner pipeline.
            record['return_ready']=max(record['aic_end'],record['combine_end']+rc.even_empty_clear,
                                       record['sg_end']+rc.odd_empty_clear)+rc.exit
            continue
        for lane in range(aiv_workers):
            odd=bool(lane&1)
            clock=record['sg_end'] if odd else record['combine_end']+rc.even_empty_clear
            for projection,tiles in ((1,dw2_tiles),(0,dw1_tiles)):
                for x in range(r*home,(r+1)*home):
                    remotes=[d for d in range(p) if d!=r and destination[x][d]>0]
                    contributors=remotes+([r] if destination[x][r]>0 else [])
                    for tile in range(lane,tiles,aiv_workers):
                        clock+=owner_scan
                        if not remotes:continue
                        ready=[]
                        for d in contributors:
                            begin,end=releases[d,x,projection]
                            wave=(tile+aic_cores)//aic_cores
                            ready.append(begin+(end-begin)*wave/(tiles//aic_cores))
                        clock=max(clock,*ready)+checked(owner_service(projection,len(remotes),destination[x][r]>0))
            record['owner_end'].append(clock+(rc.odd_empty_clear if odd else 0.))
        record['return_ready']=max(record['local_end'],*record['owner_end'])+rc.exit
    backward=max(r['return_ready'] for r in records)
    fw=[checked(v) for v in forward_finish] if forward_finish is not None else None
    if fw is not None and len(fw)!=p:raise ValueError('One forward finish per rank required')
    return dict(ranks=records,backward_envelope=backward,device_extent=extent,
                early_dw1_active=bool(early_active),
                checkpoint_moe_envelope=(2*max(fw)+backward if fw is not None else None),
                calibrated=False,step_latency_prediction=False,
                caveats=['Phase inputs and tile-wave interpolation require independent calibration/ranking qualification',
                         'No source transport model is inferred; caller supplies actual quota-dependent release estimates',
                         'No owner/AIC/HBM/network interference feedback is modeled',
                         'Existing final rendezvous prevents treating dX GEMM completion as framework readiness'])
