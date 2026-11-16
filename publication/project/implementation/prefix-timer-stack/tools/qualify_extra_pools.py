from controller import run_job
for backend,mode,tag,arguments in [
    ('shared','sharedpool','gate-pool-fused-shared64-r2',['--experts','256','--slots','32','--tokens','128','--ffn','256','--limit','10','--restore','panel','--checkpoint','--parameter-fp32','--require-replicas']),
    ('moonep','moonpool','gate-pool-moon-independent',[])]:
    status=run_job(backend,mode,tag,arguments)
    assert status['qualified'],(tag,status)
print('EXTRA_POOL_GATES_PASS',flush=True)
