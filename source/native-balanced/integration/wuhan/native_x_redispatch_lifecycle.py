"""Untimed two-live-forward/reverse-backward native input lease gate."""
from native_x_redispatch_audit import NativeRedispatchAudit


def check_two_live_forwards(pool, data, limit, shared_save_mode, experts):
    import torch
    from ascend_deepep.fsdp2_bridge import FSDP2ExpertsBridge
    from ascend_deepep.wuhan_replay_capture import tensor_sha
    output={};prefix_audit=None
    old_sink=pool.x_redispatch_debug_sink
    try:
        for mode in ('saved','redispatch'):
            layers=[]
            prefix_audit=NativeRedispatchAudit() if mode=='redispatch' else None
            pool.x_redispatch_debug_sink=prefix_audit
            for layer in range(2):
                x=(data['x'].detach()+.03125*(layer+1)).detach().requires_grad_(True)
                ids=(data['ids'].detach()+layer*pool.home)%experts
                prob=data['prob'].detach().clone()
                if prob.shape[1]>1:
                    prob[:,0]=.75 if layer==0 else .25
                    prob[:,1:]=(1-prob[:,0:1])/(prob.shape[1]-1)
                prob.requires_grad_(True)
                names=('w1','w2','wg','wu','wd')
                weights={name:(data[name].detach()+.000244140625*layer).detach().requires_grad_(True)
                         for name in names}
                bridge=FSDP2ExpertsBridge(pool,
                    lambda weights=weights:(weights['w1'],weights['w2']),
                    lambda weights=weights:(weights['wg'],weights['wu'],weights['wd']))
                y=bridge.forward_shared(x,ids,prob,weights['wg'],weights['wu'],weights['wd'],
                    limit=limit,shared_save_mode=shared_save_mode,routed_input_mode=mode)
                tape=y.grad_fn.saved_tensors
                if mode=='redispatch' and tape[3].numel()!=0:
                    raise AssertionError('two-live-forward gate retained original packed X')
                plan=[tensor_sha(t) for t in tape[4:8]]
                layers.append(dict(x=x,prob=prob,weights=weights,y=y,plan=plan))
            # The second forward overwrote the shared pool. Poison it again;
            # each backward must publish its own live owner weights and replay
            # its own immutable placement, never the last layer's scratch.
            with torch.no_grad():
                pool.module.w1.fill_(float('nan'));pool.module.w2.fill_(float('nan'))
            torch.npu.synchronize()
            hashes={}
            for index in (1,0):
                layer=layers[index];weights=layer['weights']
                grads=torch.autograd.grad(layer['y'],
                    (layer['x'],layer['prob'],*(weights[name] for name in ('w1','w2','wg','wu','wd'))),data['dy'])
                torch.npu.synchronize()
                hashes[index]=dict(plan=layer['plan'],outputs=[tensor_sha(t) for t in (layer['y'],*grads)])
            output[mode]=hashes
            del layers
        if output['saved']!=output['redispatch']:
            raise AssertionError('two-live-forward native redispatch Y/gradients/placement differs from saved mode')
        return dict(pass_=True,layers=2,backward_order=[1,0],
            distinct_inputs_and_routes=True,nonuniform_probabilities=True,
            pooled_weights_poisoned=True,hashes=output,
            prefix_audit=prefix_audit.receipt())
    finally:
        pool.x_redispatch_debug_sink=old_sink
