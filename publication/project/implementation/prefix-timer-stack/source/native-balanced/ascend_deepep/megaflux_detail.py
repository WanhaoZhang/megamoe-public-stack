"""Bounded opt-in trace suffix; production descriptor and parameter ABI unchanged."""
import torch
from .megaflux_early import MegaFluxBF16EarlyTrain
from .megaflux_trace import _TimerReplicaOp

MAGIC=0x4d4644540001
PREFIX={'forward':128*333*8,'backward':256*8*8}
HEADER,LANES,WORDS=16,96,10


def capacities(home,slots,hidden,ffn,capacity,ep,phase):
    if phase=='forward':
        stripes=(ffn+255)//256+(hidden+255)//256
        aic=2*((((capacity+127)//128+home+slots)*stripes+31)//32)+128
        aiv=2*slots*stripes+64
    else:
        tiles=((2*ffn+127)//128)*((hidden+255)//256)+((hidden+127)//128)*((ffn+255)//256)
        aic=2*(((home+slots)*tiles+31)//32)+64
        aiv=((home*tiles+63)//64)*(1+ep*slots+8*(4+3*ep*slots))+64
    return aic,aiv


class MegaFluxBF16DetailTrain(MegaFluxBF16EarlyTrain):
    def __init__(self,*args,detail=True,timer=True,**kwargs):
        super().__init__(*args,trace=True,**kwargs)
        self.detail_enabled=detail
        self.detail_capacity=None
        self.detail_specs={}
        self.trace_timers={}
        if timer:self.op=_TimerReplicaOp(self.op,self.trace_timers)

    def _forward_device(self,x,p,plan,limit):
        self._check_device_plan(plan)
        save_mode=self._record_saved_mode(plan,limit)
        h,f=self.w1.shape[2],self.w1.shape[1]//2
        if self.detail_capacity!=plan.max_recv:
            for phase,attribute in [('forward','pipeline_trace'),('backward','gradient_trace')]:
                ca,cv=capacities(self.home,self.slots,h,f,plan.max_recv,self.shmem.world_size(),phase) if self.detail_enabled else (0,0)
                size=PREFIX[phase]+HEADER+LANES*8+32*(ca+2*cv)*WORDS
                setattr(self,attribute,torch.zeros(size,device=x.device,dtype=torch.int64))
                self.detail_specs[phase]={'aic_capacity':ca,'aiv_capacity':cv,'words':size}
            self.detail_capacity=plan.max_recv
        for phase,tensor in [('forward',self.pipeline_trace),('backward',self.gradient_trace)]:
            tensor.zero_()
            spec=self.detail_specs[phase]
            header=[MAGIC if self.detail_enabled else 0,spec['aic_capacity'],spec['aiv_capacity'],plan.epoch,self.shmem.world_size(),self.home,self.slots,h,f]+[0]*7
            tensor[PREFIX[phase]:PREFIX[phase]+HEADER].copy_(torch.tensor(header,dtype=torch.int64,device=x.device))
        y,counts,state,dispatch,_=self.op.compute_train_replica(
            self.shmem,x,plan.physical_ids,p,self.w1,self.w2,plan.max_recv,float(limit),save_mode,self._descriptor(plan))
        self.last_exec_counts=counts
        rank=self.shmem.rank()
        return y,plan.logical_counts[rank*self.home:(rank+1)*self.home],counts,state,dispatch

    def collect_detail(self):
        """Call only after checked finish; copy used records, not empty capacity."""
        import numpy as np
        result={}
        for phase,tensor in [('forward',self.pipeline_trace),('backward',self.gradient_trace)]:
            offset=PREFIX[phase]
            header=tensor[offset:offset+HEADER+LANES*8].cpu().numpy()
            ca,cv=map(int,header[1:3]);lanes=[];records=[]
            for lane in range(LANES):
                count,overflow=map(int,header[HEADER+lane*8:HEADER+lane*8+2])
                assert overflow==0,('trace overflow',phase,lane,count,overflow)
                group,sub=divmod(lane,3);base=group*(ca+2*cv)+(ca+(sub-1)*cv if sub else 0)
                cap=cv if sub else ca
                assert 0<=count<=cap
                if count:
                    begin=offset+HEADER+LANES*8+base*WORDS
                    records.append(tensor[begin:begin+count*WORDS].cpu().numpy().reshape(count,WORDS))
                    lanes.extend([lane]*count)
            result[phase]={'header':header,'lanes':np.asarray(lanes,dtype=np.int32),
                           'records':np.concatenate(records) if records else np.empty((0,WORDS),dtype=np.int64),
                           'sparse':tensor[:offset].cpu().numpy()}
        return result
