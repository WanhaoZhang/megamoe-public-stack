"""Resolve a routable SHMEM endpoint on each EP group's actual leader."""
import ipaddress
from fsdp_turbo.utils.env import native_env, launcher_env, ensure_shmem_session
import socket
from urllib.parse import urlsplit


def _leader_endpoint(root):
    configured=native_env("ASCEND_DEEPEP_SHMEM_IP_PORT", configured_only=True)
    if configured:
        parsed=urlsplit(configured)
        if parsed.scheme!='tcp' or not parsed.hostname or parsed.port is None:
            raise ValueError('SHMEM endpoint must be tcp://leader-ip:port')
        ip=socket.gethostbyname(parsed.hostname);port=parsed.port+root
    else:
        master=launcher_env("MASTER_ADDR")
        if not master:raise ValueError('large EP needs MASTER_ADDR or an explicit SHMEM endpoint')
        # UDP connect only selects a local route; no packet is sent.
        with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as sock:
            sock.connect((master,int((launcher_env("MASTER_PORT") or "29500"))))
            ip=sock.getsockname()[0]
        port=8899+root
    if ipaddress.ip_address(ip).is_loopback or not 0<port<65536:
        raise ValueError('large EP requires a routable EP-leader address and valid port')
    # An explicit address on another host must fail before SHMEM bootstrap.
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as sock:sock.bind((ip,0))
    return f'tcp://{ip}:{port}'


def resolve_balanced_bootstrap(ep_group,dist):
    root=dist.get_global_rank(ep_group,0) if ep_group is not None else 0
    message=[None]
    if dist.get_rank()==root:
        try:message[0]=dict(endpoint=_leader_endpoint(root))
        except Exception as exc:message[0]=dict(error=str(exc))
    dist.broadcast_object_list(message,src=root,group=ep_group)
    if 'error' in message[0]:raise ValueError('large EP SHMEM bootstrap: '+message[0]['error'])
    return message[0]['endpoint']
