"""Deterministic AscendTimerV2 coordinate slots, mirrored by stage_timer.h."""
MAGIC_V1=0x4d4654560001
MAGIC_V2=0x4d4654560002
MAGIC=0x4d4654560003
PREFIX={'forward':128*333*8,'backward':256*8*8}
HEADER,LANES,WORDS=16,96,8

def layout(ep,home,slots,h,f,capacity,phase,version=3):
    if version not in (1,2,3):raise ValueError('unknown AscendTimerV2 stage schema')
    s1=(f+255)//256;s2=(h+255)//256
    t1=((2*f+127)//128)*((h+255)//256)
    t2=((h+127)//128)*((f+255)//256)
    d=dict(ep=ep,home=home,slots=slots,h=h,f=f,capacity=capacity,s1=s1,s2=s2,t1=t1,t2=t2,mb=(capacity+127)//128)
    if phase=='forward':
        cp=slots*(s1+s2)*32;task=(home+slots)*d['mb']*(s1+s2)
        d.update(copy=cp,task=task,total=2*cp+2*task)
    else:
        prod=(home+slots)*(t1+t2);owner=home*(t1+t2);peers=ep*slots;ops=4+3*peers
        wait=2*prod;op=wait+owner*(1+peers);ph=op+owner*8*ops
        d.update(producer=prod,owner=owner,peers=peers,ops=ops,wait_base=wait,op_base=op,phase_base=ph,total=ph+192)
    d['version']=version;d['legacy_total']=d['total'];d['extra_ranges']={}
    if version>=2:
        sizes=({17:d['copy'],**{kind:d['task'] for kind in range(22,27)}} if phase=='forward'
               else {18:d['producer'],19:d['owner'],20:d['owner'],21:d['owner'],
                     27:d['producer'],28:LANES*8,29:LANES})
        for kind,size in sizes.items():
            d['extra_ranges'][kind]=(d['total'],size);d['total']+=size
    if version>=3 and phase=='backward':
        # Activation-gradient grids differ from weight-gradient grids.
        tasks=(home+slots)*d['mb']*(s1+s2)
        for kind in (30,31):
            d['extra_ranges'][kind]=(d['total'],tasks);d['total']+=tasks
    return d

def decode(d,phase,index):
    """Return kind,e,p,row,col,peer,slot; owner e is local home index."""
    if not 0<=index<d['total']:raise ValueError('stage slot out of bounds')
    if index>=d['legacy_total']:
        for kind,(begin,size) in d['extra_ranges'].items():
            if begin<=index<begin+size:
                offset=index-begin
                if kind==28:return kind,-1,-1,offset//LANES,-1,-1,-1
                if kind==29:return kind,-1,-1,-1,-1,-1,-1
                if kind in (30,31):
                    s=d['s1']+d['s2'];coord,stripe=divmod(offset,s)
                    e,m=divmod(coord,d['mb']);p=int(stripe>=d['s1'])
                    return kind,e,p,m*128,(stripe-(d['s1'] if p else 0))*256,-1,e-d['home']
                # Extra kinds use the exact same task grids as v1 pairs.
                if phase=='forward':
                    legacy=offset if kind==17 else 2*d['copy']+offset
                else:legacy=offset  # producer and owner share tile coordinates
                _,e,p,row,col,peer,slot=decode(d,phase,legacy)
                return kind,e,p,row,col,peer,slot
        raise ValueError('unmapped stage slot')
    if phase=='forward':
        s=d['s1']+d['s2']
        if index<2*d['copy']:
            kind,index=divmod(index,d['copy']);coord,worker=divmod(index,32);slot,stripe=divmod(coord,s)
            p=int(stripe>=d['s1']);return kind+1,-1,p,stripe-(d['s1'] if p else 0),-1,-1,slot
        kind,index=divmod(index-2*d['copy'],d['task']);coord,stripe=divmod(index,s);e,m=divmod(coord,d['mb'])
        p=int(stripe>=d['s1']);return kind+3,e,p,m*128,(stripe-(d['s1'] if p else 0))*256,-1,e-d['home']
    t=d['t1']+d['t2']
    if index<d['wait_base']:
        kind,pt=divmod(index,d['producer']);rest=None;kind+=5
    elif index<d['op_base']:
        pt,peer=divmod(index-d['wait_base'],1+d['peers']);kind=7
        rest=(-1,-1) if peer==0 else divmod(peer-1,d['slots'])
    elif index<d['phase_base']:
        coord,op=divmod(index-d['op_base'],d['ops']);pt,sub=divmod(coord,8)
        if op<2:kind=8+op;rest=(-1,-1)
        elif op>=2+3*d['peers']:kind=13+op-(2+3*d['peers']);rest=(-1,-1)
        else:
            peer,kind=divmod(op-2,3);kind+=10;rest=divmod(peer,d['slots'])
    else:
        i=index-d['phase_base']
        return (15,-1,i%2,-1,-1,-1,-1) if i<128 else (16,-1,-1,-1,-1,-1,-1)
    e,tile=divmod(pt,t);p=int(tile>=d['t1']);tile-=d['t1'] if p else 0
    row,col=divmod(tile,((d['f'] if p else d['h'])+255)//256)
    row=row*128+(sub*16 if 8<=kind<=14 else 0)
    return kind,e,p,row,col*256,*(rest if rest else (-1,-1))


# Schema 8 appends backward owner slots by projection, local expert and lane.
SUMMARY_KINDS=((15,2),(16,1),(28,8),(29,1),(40,2),(41,2),(42,2))
def summary_layout(ep,home,slots,h,f,capacity,phase,version=7):
    if version not in (7,8,9):
        raise ValueError('unknown summary Timer schema')
    ranges={};total=0
    for kind,n in SUMMARY_KINDS:
        ranges[kind]=(total,n*LANES);total+=n*LANES
    if version in (8,9) and phase=='backward':
        count=2*home*LANES
        ranges[43]=(total,count);total+=count
    if version==9 and phase=='backward':
        bins=(max(2*f,h)+127)//128
        for kind,count in ((6,2*(home+slots)*bins),(19,2*home*bins),(20,2*home*bins)):
            ranges[kind]=(total,count);total+=count
    return dict(ep=ep,home=home,slots=slots,h=h,f=f,capacity=capacity,
                version=version,total=total,extra_ranges=ranges,summary=True)

def summary_decode(d,phase,index):
    for kind,(start,count) in d['extra_ranges'].items():
        if start<=index<start+count:
            coordinate,lane=divmod(index-start,LANES)
            if kind==43:
                projection,expert=divmod(coordinate,d['home'])
                return kind,expert,projection,-1,lane,-1,-1
            if d['version']==9 and kind in (6,19,20):
                bins=(max(2*d['f'],d['h'])+127)//128
                coordinate,rowbin=divmod(index-start,bins)
                projection,expert=divmod(coordinate,d['home']+d['slots'] if kind==6 else d['home'])
                return kind,expert,projection,rowbin*128,0,-1,-1
            return kind,-1,coordinate if count==2*LANES else -1,coordinate if kind==28 else -1,lane,-1,-1
    raise ValueError('unknown phase Timer coordinate')
