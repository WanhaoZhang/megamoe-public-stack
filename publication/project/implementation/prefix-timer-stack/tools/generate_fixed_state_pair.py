"""Generate a reviewable fixed-state correctness driver from frozen train.py.

The original training loop/source is never edited. This diagnostic alternates
two policies before the single Adam update, so gradients share identical weights.
CPU copies are correctness references, not training offload or timing scores.
"""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
LOOP=r'''
    # Diagnostic only: no measured training timing is exported.
    from fsdp_turbo.distributed.expert_parallel.dispatcher_megamoe_balanced import _POOLS,check_step_completions,close_megamoe_balanced
    from ascend_deepep.unified_planner import CommonOnlinePlanner
    from mindspeed_llm.fsdp2.features.ep8_imbalance import plan_invocation
    import math
    from datetime import timedelta
    # CPU reference work takes different amounts of time on different ranks.
    # Align on a host-only group before enqueueing the next HCCL collective.
    host_group=dist.new_group(backend='gloo',timeout=timedelta(minutes=20))
    torch.set_num_threads(1)
    policy_pairs={};current_buffers={};handles=[];records=[]
    def cpu_value(value):return local(value).detach().to('cpu',copy=True).contiguous()
    def output_hook(index):
        def hook(module,ins,output):
            if plan_invocation()[1]:return
            target=current_buffers
            value=output if isinstance(output,torch.Tensor) else output[0]
            target['y'+str(index)]=cpu_value(value)
            for name,tensor in [('dx',ins[0]),('dp',ins[2])]:
                if tensor.requires_grad:
                    def save(grad,key=name+str(index),target=target):target[key]=cpu_value(grad)
                    tensor.register_hook(save)
        return hook
    for index,layer in enumerate(engine.model.model.layers):handles.append(layer.mlp.experts.register_forward_hook(output_hook(index)))
    def compare(a,b):
        a=a.reshape(-1).numpy();b=b.reshape(-1).numpy();err=base=0.
        assert a.shape==b.shape
        for begin in range(0,len(a),4*1024**2):
            aa=np.asarray(a[begin:begin+4*1024**2],dtype=np.float64)
            bb=np.asarray(b[begin:begin+4*1024**2],dtype=np.float64)
            if not np.isfinite(aa).all() or not np.isfinite(bb).all():raise RuntimeError('nonfinite fixed-state reference')
            diff=aa-bb;err+=float(np.sum(diff*diff));base+=float(np.sum(aa*aa))
        return dict(relative_l2=math.sqrt(err/max(base,1e-24)),difference_energy=err,reference_energy=base)
    def switch(name):
        for pool in _POOLS.values():
            key=id(pool)
            if key not in policy_pairs:
                policy_pairs[key]={'step_time':pool.policy,'native':CommonOnlinePlanner(name='native',home=pool.home,slots=pool.slots,
                    tokens=pool.tokens,topk=pool.topk,device=device,group=pool.group)}
            pool.policy=policy_pairs[key][name];pool.planner_name=name
    for step,(inputs,labels) in enumerate(samples):
        set_step(step);cpu_rng=torch.get_rng_state();npu_rng=torch.npu.get_rng_state()
        immutable_before=parameter_samples(engine);reference=None;reference_outputs=None;reference_loss=None;reference_norm=None
        for policy_name in ('step_time','native'):
            dist.barrier(group=host_group)
            switch(policy_name);current_buffers={};optimizer.zero_grad()
            torch.set_rng_state(cpu_rng);torch.npu.set_rng_state(npu_rng)
            alpha=torch.tensor(float(args.tokens),device=device)
            chunks=[dict(shift_labels=chunk,alpha=alpha,reduction='sum') for chunk in labels.split(1024,dim=1)]
            context=partial(chunk_loss,loss_forward=calculate_lm_loss,loss_kwargs_chunks=chunks,chunk_size=1024)
            result=engine(input_ids=inputs,use_cache=False,loss_ctx=context);loss=result.loss;loss.backward();check_step_completions()
            torch.accelerator.synchronize()
            assert set(current_buffers)=={'y0','dx0','dp0','y1','dx1','dp1'},current_buffers.keys()
            gradients={}
            if policy_name=='step_time':
                for name,parameter in engine.named_parameters():
                    if parameter.grad is not None:
                        tensor=local(parameter.grad);assert tensor.dtype==torch.float32 and tensor.device.type=='npu'
                        gradients[name]=cpu_value(parameter.grad)
                reference=gradients;reference_outputs=current_buffers;reference_loss=float(loss.detach())
            else:
                errors={};aggregate_error=aggregate_reference=0.;critical=[]
                for name,parameter in engine.named_parameters():
                    if parameter.grad is None:
                        assert name not in reference;continue
                    assert name in reference
                    metric=compare(reference[name],cpu_value(parameter.grad));errors[name]=metric['relative_l2']
                    aggregate_error+=metric['difference_energy'];aggregate_reference+=metric['reference_energy']
                    if '.experts.' in name or '.shared_experts.' in name:
                        critical.append(name);assert metric['relative_l2']<=.02,(step,name,metric)
                output_errors={key:compare(reference_outputs[key].float(),value.float())['relative_l2'] for key,value in current_buffers.items()}
                assert all(v<=(.004 if key.startswith('y') else .02) for key,v in output_errors.items()),(step,output_errors)
                aggregate=math.sqrt(aggregate_error/max(aggregate_reference,1e-24));assert aggregate<=.02,(step,aggregate)
                assert critical,'no routed/shared gradients were compared'
                loss_error=abs(float(loss.detach())/reference_loss-1);assert loss_error<=.004
            dist.barrier(group=host_group)
            norm=clip_grad_norm(engine,1.0,error_if_nonfinite=True)
            if policy_name=='step_time':reference_norm=float(local(norm))
            norm_error=abs(float(local(norm))/max(reference_norm,1e-12)-1)
            retired=sum(pool.retire_training_step(step) for pool in _POOLS.values());assert retired==2
            after=parameter_samples(engine)
            assert all(torch.equal(immutable_before[name],after[name]) for name in immutable_before),'parameters changed before paired optimizer update'
            if policy_name=='native':
                record=dict(step=step,passed=True,loss_relative=loss_error,fixed_weight_global_norm_relative=norm_error,
                    complete_parameter_gradient_relative_l2=aggregate,critical_max_gradient_relative_l2=max(errors[name] for name in critical),
                    all_parameter_max_gradient_relative_l2=max(errors.values()),output_and_input_errors=output_errors,parameters_compared=len(errors))
                records.append(record)
                with (args.out/f'paired-rank{rank}.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                print('FIXED_STATE_PAIR',rank,json.dumps(record),flush=True)
            del result,loss
        check_persistent(engine,optimizer,gradients=True);optimizer.step();optimizer.zero_grad()
        reference=reference_outputs=None;current_buffers={}
        if step==0:loaded_libraries(args.out,rank);configure_allocator_budget(args,rank,device)
    for handle in handles:handle.remove()
    assert all(provider.calls[s,l]==4 for s in range(25) for l in range(2)),provider.calls
    audit=[dict(policy_calls={name:policy.calls for name,policy in policy_pairs[id(pool)].items()},replay=pool.replay_audit,live=len(pool._plan_bodies)) for pool in _POOLS.values()]
    assert all(x['policy_calls']=={'step_time':50,'native':50} and x['live']==0 and x['replay']==dict(original=100,recompute=100,backward=100) for x in audit),audit
    (args.out/f'pair-audit-rank{rank}.json').write_text(json.dumps(audit,indent=2))
    (args.out/f'rank{rank}.json').write_text(json.dumps(dict(passed=True,steps=25,reference='identical weights/inputs/IDs before one Adam update; same ordinary worker',
        independent_small_shape_reference_is_separate=True,training_performance_score=False,records=records),indent=2))
    dist.destroy_process_group(host_group)
    close_megamoe_balanced();dist.barrier();dist.destroy_process_group()
    print('FIXED_STATE_PAIR_PASS',rank,flush=True)

if __name__=='__main__':main()
'''
def main():
    source=ROOT/'tools/train.py';payload=source.read_bytes();freeze=json.loads((ROOT/'evidence/STEP-TIME-FREEZE.json').read_text())
    assert hashlib.sha256(payload).hexdigest()==freeze['source']['tools/train.py']
    text=payload.decode().replace('\r\n','\n');marker="    memory_mark(-1, 'initialized')"
    assert text.count(marker)==1
    generated=text[:text.index(marker)]+LOOP
    target=ROOT/'tools/fixed_state_pair.py';target.write_text(generated,encoding='utf-8',newline='\n')
    record=dict(original_train_sha256=hashlib.sha256(payload).hexdigest(),generated_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        original_not_modified=True,loop='25 identical-weight policy pairs,one Adam update after each pair',formal_timing=False,
        host_only_alignment_before_collectives=True)
    (ROOT/'evidence/fixed-state-pair-source.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
if __name__=='__main__':main()
