from pathlib import Path
import sys
p=Path(__file__).resolve().parent
source=p/'source/integration/wuhan/test_bridge_npu.py'
sys.path.insert(0,str(source.parent))
text=source.read_text()
old='    from fsdp_turbo.distributed.expert_parallel.dispatcher_mega_moe import _local_mem_bytes'
assert text.count(old)==1
new='''    from timer_capture_config import window_bytes
    def _local_mem_bytes(m,k,topk,ep_size,local_e,max_recv_tokens=None):
        return window_bytes(m,k,topk,ep_size,local_e,max_recv_tokens or m*topk)'''
text=text.replace(old,new)
exec(compile(text,str(source),'exec'),dict(__name__='__main__',__file__=str(source)))
