"""ABI3 native X/prob phase scopes, independent from the V2 shared Timer."""
import numpy as np


def native_x_timer_events(array, rank, include_waits=False):
    if array.dtype != np.int64 or array.shape != (4, 128, 2):
        raise ValueError('ABI3 X redispatch Timer requires int64 [4,128,2]')
    if np.any((array[:,:,0] == 0) != (array[:,:,1] == 0)):
        raise ValueError('incomplete native X redispatch Timer pair')
    names=('SOURCE_GATHER_X_DY_PROB','SOURCE_PUBLISH_SYNC','X_PROB_PULL','READ_RETIRE_SYNC')
    output=[]
    for phase,name in enumerate(names):
        expected=list(range(1,64,2)) if phase==2 else list(range(64))
        actual=np.flatnonzero(array[phase,:,0]).tolist()
        if actual!=expected:
            raise ValueError('native X redispatch participant set differs from ABI3: '+name)
        for lane in actual:
            start,end=map(int,array[phase,lane])
            if not 0<start<=end:
                raise ValueError('invalid native X redispatch Timer bounds')
            if phase in (1,3) and not include_waits:
                continue
            output.append(dict(name='X_REDISPATCH/'+name,ph='X',pid=rank,
                tid=4500+128*phase+lane,cat='native_x_redispatch_scope',
                args=dict(start_cycle=start,end_cycle=end,invocation='backward',
                    source='native_x_redispatch_timer_v3',redispatch_phase=phase,
                    recorded_lane=lane,logical_group=lane//2,aiv_subblock=lane%2,
                    scope_only=True,scope_includes_implementation_waits=True,
                    native_original_x_redispatch=True,durations_are_not_additive=True)))
    return output


def native_x_timer_metadata(events,rank):
    return [dict(ph='M',pid=rank,tid=e['tid'],name='thread_name',
                 args=dict(name=e['name']+' / AIV'+str(e['args']['recorded_lane'])))
            for e in events if e.get('cat')=='native_x_redispatch_scope']
