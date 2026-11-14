"""Worker extension to the same AscendTimerV2 coordinate buffer (schema 4)."""
from .megaflux_stage_layout import layout as base_layout,decode as base_decode

def schema5_layout(ep,home,slots,h,f,capacity,phase):
    d=base_layout(ep,home,slots,h,f,capacity,phase)
    d['version']=5;d['base_total']=d['total']
    if phase=='forward':
        d['chunk1']=2*((256*h+32767)//32768)
        d['chunk2']=(256*f+32767)//32768
        count=slots*(d['s1']*d['chunk1']+d['s2']*d['chunk2'])
        d['extra_ranges'][32]=(d['total'],count);d['total']+=count
    else:
        d['restore_panels']=(2*f+255)//256+(h+255)//256
        for kind in (33,34):
            count=slots*d['restore_panels']*32
            d['extra_ranges'][kind]=(d['total'],count);d['total']+=count
    return d

def schema5_decode(d,phase,index):
    if phase=='forward' and index>=d['base_total']:
        i=index-d['base_total'];first=d['slots']*d['s1']*d['chunk1']
        p=int(i>=first)
        if p:i-=first
        stripe,chunk=divmod(i,d['chunk2'] if p else d['chunk1'])
        slot,stripe=divmod(stripe,d['s2'] if p else d['s1'])
        return 32,-1,p,stripe,chunk,-1,slot
    if phase=='backward' and index>=d['base_total']:
        i=index-d['base_total'];count=d['slots']*d['restore_panels']*32
        kind=33+i//count;i%=count;panel,worker=divmod(i,32)
        slot,panel=divmod(panel,d['restore_panels']);w1=(2*d['f']+255)//256
        p=int(panel>=w1)
        if p:panel-=w1
        return kind,-1,p,panel,worker,-1,slot
    return base_decode(d,phase,index)

PARTITIONS=64

def layout(ep,home,slots,h,f,capacity,phase):
    d=schema5_layout(ep,home,slots,h,f,capacity,phase)
    d['schema5_total']=d['total'];d['version']=6
    count=(home+slots)*ep*PARTITIONS
    kinds=(35,36,37) if phase=='forward' else (38,39)
    d['source_mb']=(capacity+255)//256
    for kind in kinds:
        size=count*d['source_mb'] if kind==36 else count
        d['extra_ranges'][kind]=(d['total'],size);d['total']+=size
    return d

def decode(d,phase,index):
    if d.get('summary'):
        from .megaflux_stage_layout import summary_decode
        return summary_decode(d,phase,index)
    if index<d['schema5_total']:return schema5_decode(d,phase,index)
    for kind in ((35,36,37) if phase=='forward' else (38,39)):
        start,count=d['extra_ranges'][kind]
        if start<=index<start+count:
            i=index-start;source,partition=divmod(i,PARTITIONS)
            value,peer=divmod(source,d['ep'])
            e,block=divmod(value,d['source_mb']) if kind==36 else (value,-1)
            return kind,e,-1,block,-1,peer,partition
    raise ValueError('unknown producer Timer coordinate')
