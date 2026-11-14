#pragma once
// Executor-specific cost search. All times are integer16ns units. The model
// prefix is immutable; the suffix is invocation-private planning workspace.
// Memory exposes costs/scratch/source statistics/quota only, never Parameters.
#include "mega_moe_unified_policy.h"
namespace StepTime {
using I=int64_t;
constexpr I INF=(I(1)<<54);
constexpr int WIDTH=64, CELLS=512, UNIT=16;
constexpr int SQ=0, SC=2048, SI=2304, SOURCE=4352;
constexpr int R10=20736,R11=21248,R20=21760,R21=22272;
constexpr int DATA=22784,W1=23296,W2=23808,ACT=24320,CACHE=25000;
UP_INLINE I maxv(I a,I b){return a>b?a:b;}
UP_INLINE I minv(I a,I b){return a<b?a:b;}
UP_INLINE int ceilv(int n,int d){return (n+d-1)/d;}
struct Move {int e,from,to,rows;};

template<class Memory> struct Engine {
    UnifiedPlanner::State& a;Memory& m;int cap;
    int active[8],rep[256],copies[8];
    I fw[8],p1[8],p2[8],tail[8],owners[8];
    I network_rows[64];int network_copies[64];
    UP_INLINE int c(int i){return m.cost(i);}
    UP_INLINE int slot(int d,int e){
        if(e/a.h==d)return e%a.h;
        for(int j=0;j<a.s;++j)if(a.cp[d*a.s+j]==e)return a.h+j;
        return -1;
    }
    UP_INLINE int logical(int d,int k){return k<a.h?d*a.h+k:a.cp[d*a.s+k-a.h];}
    UP_INLINE I primitive(int phase,int rows,int count,bool activity=true){
        if(rows<=0)return 0;
        int b=208+phase*5,tm=phase<2?192:128;
        if(c(16)==2){
            // Affine large-row derivative of the same calibrated phase fit.
            int cols=phase==0?16:phase==2||phase==6?28:8;
            I rateps=I(c(b+1))*UNIT*1000/tm;
            rateps+=I(c(b+2))*UNIT*1000*(phase==5||phase==7?1:cols)/(tm*(phase==5||phase==7?2:32));
            return c(b)+I(rows)*rateps/(UNIT*1000)+(activity?I(c(b+4))*count/32:0);
        }
        int tile=ceilv(rows,tm),cols=phase==0?16:phase==2||phase==6?28:8;
        int wave=(phase==5||phase==7)?ceilv(tile,2):ceilv(tile*cols,32);
        return c(b)+I(c(b+1))*tile+I(c(b+2))*wave+I(c(b+3))*((rows-1)%tm+1)/tm+
            (activity?I(c(b+4))*count/32:0);
    }
    UP_INLINE void cache_slot(int d,int k){
        int e=logical(d,k),n=e<0?0:a.q[e*a.p+d];
        for(int i=0;i<8;++i)m.put(CACHE+i*CELLS+d*WIDTH+k,int(primitive(i,n,0,false)));
    }
    UP_INLINE I phase(int i,int d,int k){return m.get(CACHE+i*CELLS+d*WIDTH+k)+I(c(208+i*5+4))*active[d]/32;}
    UP_INLINE void refresh(){
        for(int e=0;e<a.e;++e)rep[e]=0;
        for(int d=0;d<a.p;++d){active[d]=copies[d]=0;
            for(int j=0;j<a.s;++j)if(a.cp[d*a.s+j]>=0){++rep[a.cp[d*a.s+j]];++copies[d];}
            for(int e=0;e<a.e;++e)active[d]+=a.q[e*a.p+d]>0;
        }
    }
    UP_INLINE I routing(int kind,int e,int d){
        int n=a.q[e*a.p+d];if(!n)return 0;
        int local=m.quota(d,e,d),b=256+kind*3;
        return c(b)+(I(local)*c(b+1)+I(n-local)*c(b+2))/(UNIT*1000);
    }
    UP_INLINE I link(int s,int d,I bytes){
        if(s==d||!bytes)return 0;
        return c(64+s*8+d)+bytes*c(128+s*8+d)/(UNIT*1000000);
    }
    UP_INLINE I fan(I sum,I peak,int peers,int rank){
        if(!peers)return 0;
        I ppm=1000000+I(c(192+rank)-1000000)*(peers-1)/6;
        return maxv(peak,(sum+peers-1)/peers*ppm/1000000);
    }
    UP_INLINE I network(int token_bytes,int weight_bytes,int gradient_bytes){
        I peer[64];for(int i=0;i<64;++i)peer[i]=0;
        for(int s=0;s<a.p;++s)for(int d=0;d<a.p;++d)if(s!=d)
            peer[s*8+d]=network_rows[s*8+d]*token_bytes+I(network_copies[s*8+d])*weight_bytes+
                I(network_copies[d*8+s])*gradient_bytes;
        I answer=0;
        for(int s=0;s<a.p;++s){I sum=0,peak=0;int count=0;
            for(int d=0;d<a.p;++d){I t=link(s,d,peer[s*8+d]);sum+=t;peak=maxv(peak,t);count+=t>0;}
            answer=maxv(answer,fan(sum,peak,count,s));
        }
        for(int d=0;d<a.p;++d){I sum=0,peak=0;int count=0;
            for(int s=0;s<a.p;++s){I t=link(s,d,peer[s*8+d]);sum+=t;peak=maxv(peak,t);count+=t>0;}
            answer=maxv(answer,fan(sum,peak,count,d));
        }
        return answer;
    }
    UP_INLINE int order(int d,int* out){
        int n=0;
        for(int e=0;e<a.e;++e)if(rep[e]&&a.q[e*a.p+d])out[n++]=slot(d,e);
        for(int j=0;j<a.h;++j){int e=d*a.h+j;if(!rep[e]&&a.q[e*a.p+d])out[n++]=j;}
        return n;
    }
    UP_INLINE I score(){
        refresh();int fullbytes=6*c(7)*c(8);
        for(int i=0;i<64;++i){network_rows[i]=0;network_copies[i]=0;}
        for(int e=0;e<a.e;++e){
            for(int s=0;s<a.p;++s)for(int d=0;d<a.p;++d)if(s!=d)network_rows[s*8+d]+=m.quota(s,e,d);
            for(int d=0;d<a.p;++d)if(d!=e/a.h&&a.q[e*a.p+d])++network_copies[(e/a.h)*8+d];
        }
        int allcopies=0;for(int d=0;d<a.p;++d)allcopies+=copies[d];
        I copy=network(0,fullbytes,0)*c(25)/1000000;
        I in_fw=network(2*c(7)+8,0,0)*c(24)/1000000;
        I net_fw=network(4*c(7)+12,fullbytes,0)*c(24)/1000000;
        I net_bw=network(6*c(7)+16,fullbytes,fullbytes)*c(24)/1000000;
        I global_fw=0,global_tail=0;
        for(int d=0;d<a.p;++d){I dispatch=0;
            for(int k=0;k<a.h+a.s;++k){int e=logical(d,k);if(e<0||!a.q[e*a.p+d])continue;
                dispatch+=routing(0,e,d);m.put(DATA+d*WIDTH+k,int(dispatch));
            }
            I total=maxv(dispatch,in_fw),g1=0,activation=0;int ordinal=0;
            for(int k=0;k<a.h+a.s;++k){int e=logical(d,k);if(e<0||!a.q[e*a.p+d])continue;
                I ready=c(20)+m.get(DATA+d*WIDTH+k);
                if(k>=a.h){
                    I first=copy*(ordinal*1000000+c(40))/(maxv(copies[d],1)*1000000);
                    ready=maxv(ready,total+first*2/3);++ordinal;
                }
                g1=maxv(g1,ready)+phase(0,d,k);
                activation=maxv(activation,g1)+phase(1,d,k);m.put(ACT+d*WIDTH+k,int(activation));
            }
            I g2=g1,combine=0;ordinal=0;
            for(int k=0;k<a.h+a.s;++k){int e=logical(d,k);if(e<0||!a.q[e*a.p+d])continue;
                I ready=m.get(ACT+d*WIDTH+k);
                if(k>=a.h){ready=maxv(ready,copy*(ordinal*1000000+c(41))/(maxv(copies[d],1)*1000000));++ordinal;}
                g2=maxv(g2,ready)+phase(2,d,k);combine=maxv(combine,g2)+routing(1,e,d);
            }
            fw[d]=maxv(maxv(g2,combine),net_fw)+c(21);global_fw=maxv(global_fw,fw[d]);
            int sequence[64],n=order(d,sequence);I data=0,dh=0,sg=0;
            for(int j=0;j<n;++j){int k=sequence[j],e=logical(d,k);
                data+=routing(2,e,d);
                I ready=c(22)+data;
                if(k>=a.h)ready=maxv(ready,copy*(j*1000000+c(41))/(maxv(n,1)*1000000)/3);
                dh=maxv(dh,ready)+phase(3,d,k);sg=maxv(sg,dh)+phase(4,d,k);
            }
            p1[d]=maxv(dh,sg);I clock=p1[d],returned=clock;
            for(int j=0;j<n;++j){int k=sequence[j],e=logical(d,k);I w=phase(5,d,k);
                m.put(R20+d*WIDTH+k,int(clock+w*c(39)/1000000));clock+=w;m.put(R21+d*WIDTH+k,int(clock));
                if(k>=a.h)clock=maxv(clock,copy);
                clock+=phase(6,d,k);returned=maxv(returned,clock)+routing(3,e,d);
            }
            p2[d]=maxv(clock,returned)+I(c(27))*((allcopies?a.h+a.s:a.h)-active[d]);
            for(int j=0;j<n;++j){int k=sequence[j];I w=phase(7,d,k);
                m.put(R10+d*WIDTH+k,int(clock+w*c(38)/1000000));clock+=w;m.put(R11+d*WIDTH+k,int(clock));
            }
            tail[d]=maxv(clock,p2[d]);global_tail=maxv(global_tail,tail[d]);
        }
        I global_owner=0;
        for(int owner=0;owner<a.p;++owner){I finish=tail[owner];
            if(c(16)==1){owners[owner]=finish;continue;}
            for(int parity=0;parity<2;++parity){I clock=parity?p1[owner]:p2[owner];
                for(int turn=0;turn<2;++turn){int projection=1-turn,tiles=c(36+projection),residue=parity?63:62;
                    if(residue>=tiles){residue=tiles-1;if((residue&1)!=parity)--residue;}
                    if(residue<0)continue;int tasks=(tiles-1-residue)/64+1;
                    for(int j=0;j<a.h;++j){int e=owner*a.h+j;if(!rep[e])continue;
                        I first=0,last=0;
                        for(int d=0;d<a.p;++d)if(a.q[e*a.p+d]){int k=slot(d,e),b=(projection?R20:R10)+d*WIDTH+k,z=(projection?R21:R11)+d*WIDTH+k;
                            I begin=m.get(b),end=m.get(z),span=end-begin;
                            first=maxv(first,begin+span*residue/maxv(tiles-1,1));
                            last=maxv(last,begin+span*(residue+64*(tasks-1))/maxv(tiles-1,1));
                        }
                        int b=248+projection*3;I service=(c(b)+I(c(b+1))*rep[e]+(a.q[e*a.p+owner]>0?c(b+2):0))*c(26)/1000000;
                        I stride=tasks>1?(last-first+tasks-2)/(tasks-1):0;
                        clock=maxv(clock+tasks*service,first+service+(tasks-1)*maxv(service,stride));
                    }
                }
                finish=maxv(finish,clock);
            }
            owners[owner]=finish;global_owner=maxv(global_owner,finish);
        }
        I backward=maxv(maxv(global_tail,global_owner),net_bw)+c(23);
        if(c(16)==1)backward=global_tail+network(0,0,fullbytes)*c(24)/1000000+c(23);
        // Representative complete-step DAG. All MoE calls retain their real
        // cross-rank completion boundaries. FSDP prefetch/RS are separate
        // resources, and cannot be made free by summing overlapping scopes.
        I now=0,ag=0,rs=0;int layers=c(55);
        for(int layer=0;layer<layers;++layer){now=maxv(now,ag)+c(44)+c(52);
            ag=now+c(48);now+=c(42)+c(29)+global_fw+c(46);}
        now+=c(53);
        for(int layer=layers-1;layer>=0;--layer){now=maxv(now,ag)+c(44)+c(42)+global_fw+c(46)+c(47)+c(42)+backward+c(45)+c(54);
            rs=maxv(rs,now)+c(49);ag=now+c(48);}
        return maxv(now,rs)+c(50)+c(51)+c(28);
    }
    UP_INLINE void transport(int e){
        int left[8],room[8];
        for(int s=0;s<a.p;++s){left[s]=m.source(s,e);room[s]=a.q[e*a.p+s];}
        for(int s=0;s<a.p;++s)for(int d=0;d<a.p;++d)m.set_quota(s,e,d,0);
        for(int d=0;d<a.p;++d){int n=UnifiedPlanner::mn(left[d],room[d]);m.set_quota(d,e,d,n);left[d]-=n;room[d]-=n;}
        for(int turn=0;turn<a.p*a.p;++turn){int bs=-1,bd=-1;I best=INF;
            for(int s=0;s<a.p;++s)if(left[s])for(int d=0;d<a.p;++d)if(room[d]){
                I value=I(c(128+s*8+d))*c(7)*2+c(64+s*8+d)*UNIT*1000000;
                if(value<best){best=value;bs=s;bd=d;}
            }
            if(bs<0)break;int take=UnifiedPlanner::mn(left[bs],room[bd]);
            m.set_quota(bs,e,bd,m.quota(bs,e,bd)+take);left[bs]-=take;room[bd]-=take;
        }
    }
    UP_INLINE bool apply(Move v){
        if(v.from<0){transport(v.e);return true;}
        int from=v.from,to=v.to,e=v.e,n=v.rows;if(from==to||n<=0||n>a.q[e*a.p+from])return false;
        int load=0;for(int x=0;x<a.e;++x)load+=a.q[x*a.p+to];if(load+n>cap)return false;
        int target=slot(to,e);
        if(target<0){int hole=-1;for(int j=0;j<a.s;++j)if(a.cp[to*a.s+j]<0){hole=j;break;}
            if(hole<0)return false;a.cp[to*a.s+hole]=e;target=a.h+hole;}
        int origin=slot(from,e);a.q[e*a.p+from]-=n;a.q[e*a.p+to]+=n;
        if(!a.q[e*a.p+from]&&from!=e/a.h)a.cp[from*a.s+origin-a.h]=-1;
        cache_slot(from,origin);cache_slot(to,target);transport(e);return true;
    }
    UP_INLINE int proposals(Move* out,int maximum){
        int ranks[8];for(int d=0;d<a.p;++d)ranks[d]=d;
        for(int i=1;i<a.p;++i){int r=ranks[i],j=i-1;while(j>=0&&2*fw[ranks[j]]+tail[ranks[j]]>2*fw[r]+tail[r]){ranks[j+1]=ranks[j];--j;}ranks[j+1]=r;}
        int n=0;
        // Small replicas of the most exposed fixed owner are considered first.
        int owner=0;for(int d=1;d<a.p;++d)if(owners[d]-tail[d]>owners[owner]-tail[owner])owner=d;
        int chosen[8],selected=0;
        int remove_limit=UnifiedPlanner::mn(4,maximum/4);
        for(int iter=0;iter<remove_limit&&n<maximum;++iter){int be=-1,bd=-1,bv=0x7fffffff;
            for(int d=0;d<a.p;++d)for(int j=0;j<a.s;++j){int e=a.cp[d*a.s+j];if(e<0||e/a.h!=owner)continue;
                bool seen=false;for(int k=0;k<selected;++k)if(chosen[k]==d*a.s+j)seen=true;
                if(!seen&&a.q[e*a.p+d]<bv){be=e;bd=d;bv=a.q[e*a.p+d];}}
            if(be<0)break;int k=slot(bd,be)-a.h;chosen[selected++]=bd*a.s+k;
            out[n++]={be,bd,be/a.h,bv};
        }
        int donor=ranks[a.p-1],experts[4],used=0;
        int local_limit=UnifiedPlanner::mn(4,UnifiedPlanner::mx(1,maximum/4)),load_end=maximum-local_limit-1;
        for(int pick=0;pick<4&&n<load_end;++pick){int be=-1,bv=-1;
            for(int e=0;e<a.e;++e){bool seen=false;for(int k=0;k<used;++k)if(experts[k]==e)seen=true;
                if(!seen&&a.q[e*a.p+donor]>bv){be=e;bv=a.q[e*a.p+donor];}}
            if(be<0||bv<=0)break;experts[used++]=be;
            for(int ri=0;ri<3&&ri<a.p-1&&n<load_end;++ri){int d=ranks[ri];
                int take=UnifiedPlanner::mn(bv,UnifiedPlanner::mx(128,(bv/2/128)*128));
                out[n++]={be,donor,d,take};
                if(n<load_end&&take>128)out[n++]={be,donor,d,128};
            }
        }
        // A quota-only locality proposal preserves destination row counts.
        int hot[4],nh=0;
        for(int pick=0;pick<local_limit&&n<maximum-1;++pick){int be=-1,bv=0;
            for(int e=0;e<a.e;++e){bool seen=false;for(int k=0;k<nh;++k)if(hot[k]==e)seen=true;if(seen)continue;
                int remote=0;for(int s=0;s<a.p;++s)for(int d=0;d<a.p;++d)if(s!=d)remote+=m.quota(s,e,d);
                if(remote>bv){be=e;bv=remote;}}
            if(be<0)break;hot[nh++]=be;out[n++]={be,-1,-1,0};
        }
        // Migrate a replica's whole task band to a faster executor, retaining
        // the same fixed parameter/optimizer home.
        for(int j=0;j<a.s&&n<maximum;++j){int e=a.cp[donor*a.s+j];if(e>=0)out[n++]={e,donor,ranks[0],a.q[e*a.p+donor]};}
        return n;
    }
    UP_INLINE bool run(int initial,int target){
        for(int i=0;i<a.e*a.p;++i)a.instances[i]=0;
        if(!UnifiedPlanner::run(a,initial,target))return false;a.source_policy=initial;
        int src[8],dest[8],matrix[64];
        for(int e=0;e<a.e;++e){for(int s=0;s<a.p;++s){src[s]=m.source(s,e);dest[s]=initial==4?a.instances[e*a.p+s]:a.q[e*a.p+s];}
            if(!UnifiedPlanner::source_quota(src,dest,a.p,e/a.h,initial==2?1:initial==3?2:initial==4?3:0,matrix))return false;
            for(int s=0;s<a.p;++s)for(int d=0;d<a.p;++d)m.set_quota(s,e,d,matrix[s*a.p+d]);}
        for(int i=0;i<a.e*a.p;++i){m.put(SQ+i,a.q[i]);m.put(SI+i,a.instances[i]);}
        for(int i=0;i<a.p*a.s;++i)m.put(SC+i,a.cp[i]+1);
        for(int s=0;s<a.p;++s)for(int e=0;e<a.e;++e)for(int d=0;d<a.p;++d)m.put(SOURCE+(s*a.e+e)*a.p+d,m.quota(s,e,d));
        for(int d=0;d<a.p;++d)for(int k=0;k<a.h+a.s;++k)cache_slot(d,k);
        I before=score(),current=before;int rounds=UnifiedPlanner::mn(c(11),4),limit=UnifiedPlanner::mn(c(12),64);
        for(int d=0;d<a.p;++d){bool known=active[d]<=c(19)&&active[d]>=c(60);
            for(int e=0;e<a.e;++e)if(a.q[e*a.p+d]>c(18))known=false;
            if(!known){a.model_flags|=1;rounds=0;}}
        for(int round=0;round<rounds;++round){score();Move moves[64];int nm=proposals(moves,limit);I best=current;int selected=-1;
            for(int i=0;i<nm;++i){Move v=moves[i];int oldcp[64],oldq[8],oldflow[64],oldcache[16],old_from=-1,old_to=-1;
                for(int d=0;d<a.p;++d)oldq[d]=a.q[v.e*a.p+d];
                for(int s=0;s<a.p;++s)for(int d=0;d<a.p;++d)oldflow[s*a.p+d]=m.quota(s,v.e,d);
                int nc=0;if(v.from>=0){for(int j=0;j<a.s;++j)oldcp[nc++]=a.cp[v.from*a.s+j];for(int j=0;j<a.s;++j)oldcp[nc++]=a.cp[v.to*a.s+j];
                    old_from=slot(v.from,v.e);old_to=slot(v.to,v.e);
                    if(old_to<0)for(int j=0;j<a.s;++j)if(a.cp[v.to*a.s+j]<0){old_to=a.h+j;break;}
                    for(int p=0;p<8;++p){oldcache[p]=old_from<0?0:m.get(CACHE+p*CELLS+v.from*WIDTH+old_from);
                        oldcache[8+p]=old_to<0?0:m.get(CACHE+p*CELLS+v.to*WIDTH+old_to);}}
                if(apply(v)){++a.evaluations;I value=score();
                    bool known=true;for(int d=0;d<a.p;++d){if(active[d]>c(19)||active[d]<c(60))known=false;
                        for(int e=0;e<a.e;++e)if(a.q[e*a.p+d]>c(18))known=false;}
                    if(!known)a.model_flags|=1;else if(value<best){best=value;selected=i;}}
                for(int d=0;d<a.p;++d)a.q[v.e*a.p+d]=oldq[d];
                if(v.from>=0){int j=0;for(int k=0;k<a.s;++k)a.cp[v.from*a.s+k]=oldcp[j++];for(int k=0;k<a.s;++k)a.cp[v.to*a.s+k]=oldcp[j++];
                    for(int p=0;p<8;++p){if(old_from>=0)m.put(CACHE+p*CELLS+v.from*WIDTH+old_from,oldcache[p]);
                        if(old_to>=0)m.put(CACHE+p*CELLS+v.to*WIDTH+old_to,oldcache[8+p]);}}
                for(int s=0;s<a.p;++s)for(int d=0;d<a.p;++d)m.set_quota(s,v.e,d,oldflow[s*a.p+d]);
            }
            ++a.rounds;if(selected<0)break;apply(moves[selected]);current=best;a.adopted++;
        }
        I guard=c(56)+I(c(31))*(a.evaluations+2*a.rounds+1)*c(55);
        if(before-current<=guard){
            for(int i=0;i<a.e*a.p;++i){a.q[i]=m.get(SQ+i);a.instances[i]=m.get(SI+i);}
            for(int i=0;i<a.p*a.s;++i)a.cp[i]=m.get(SC+i)-1;
            for(int s=0;s<a.p;++s)for(int e=0;e<a.e;++e)for(int d=0;d<a.p;++d)m.set_quota(s,e,d,m.get(SOURCE+(s*a.e+e)*a.p+d));
            current=before;a.adopted=0;a.source_policy=initial;a.model_flags|=2;
        }else a.source_policy=5;
        a.predicted_before=float(before)*UNIT/1000000.0f;a.predicted_after=float(current)*UNIT/1000000.0f;
        return true;
    }
};
} // namespace StepTime
