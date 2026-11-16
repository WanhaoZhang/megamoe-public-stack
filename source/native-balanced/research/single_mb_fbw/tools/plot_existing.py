"""Render original rank-local intervals; never align independent rank clocks."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parents[1]
COLORS={'dw1':'#7c3aed','dw2':'#a78bfa','dx':'#2563eb','dh':'#0891b2',
    'swiglu_grad':'#10b981','owner_tile_service':'#ea580c','owner_tile_wait':'#fecaca',
    'reduce_phase':'#fb923c','backward_restore_phase':'#78716c',
    'x_dy_probability_gather':'#a3a3a3','source_publish_sync':'#f43f5e',
    'x_dy_redispatch':'#0d9488','source_read_retirement':'#fbbf24'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--baseline',type=Path,required=True);a=ap.parse_args()
    fig,axes=plt.subplots(2,1,figsize=(15,14),layout='constrained')
    with (a.baseline/'calibration/reduced/release-owner.csv').open() as stream:
        release=list(csv.DictReader(stream))
    rows=[]
    for ax,policy in zip(axes,('native','eplb_fixed')):
        path=a.baseline/f'calibration/reduced/calibration-timer-vio7-{policy}-rank0-original-trace.json'
        events=json.loads(path.read_text())['traceEvents']
        selected=[e for e in events if e.get('ph')=='X' and e.get('args',{}).get('step')==1
            and e['args'].get('layer')==0 and e['args'].get('phase')=='backward']
        kernel=next(e for e in selected if e['name']=='kernel')
        origin=kernel['ts']; labels=[]
        # The reducer explicitly defines these CSV times relative to this call's
        # original kernel origin, on the same rank's GetSystemCycle clock.
        existing=set(e['name'] for e in selected)
        for r in release:
            if (r['planner']==policy and r['rank']=='0' and r['step']=='1' and r['layer']=='0'
                and r['phase']=='backward' and r['name'] in COLORS and r['name'] not in existing):
                selected.append(dict(name=r['name'],tid=int(r['lane']),
                    ts=origin+float(r['start_ms'])*1000,dur=float(r['duration_ms'])*1000))
        # One measured lane per event kind, with nested scopes on separate rows.
        # This avoids overpainting nested dH/activation intervals as idle or work.
        ax.broken_barh([(0,kernel['dur']/1000)],(-.4,.8),facecolors='#d1d5db')
        labels.append('whole call envelope')
        for name,color in COLORS.items():
            candidates=[e for e in selected if e['name']==name]
            if not candidates:continue
            tid=min(e['tid'] for e in candidates)
            intervals=[((e['ts']-origin)/1000,e['dur']/1000) for e in candidates if e['tid']==tid]
            ax.broken_barh(intervals,(len(labels)-.4,.8),facecolors=color,alpha=.8)
            labels.append(f'{name} / lane{tid}')
        ax.set_yticks(range(len(labels)),labels,fontsize=8)
        ax.set_xlim(0,max(e['ts']+e['dur']-origin for e in selected)/1000)
        ax.set_title(f'{policy}: rank0 / step1 / layer0 backward (diagnostic only)')
        ax.set_xlabel('ms from this call start; local clock')
        ax.text(.01,.98,'Gaps are unshown/uninstrumented work or waits, not proven idle.',transform=ax.transAxes,va='top',fontsize=8)
        rows.append(dict(policy=policy,rank=0,step=1,layer=0,intervals=len(selected),kernel_ms=kernel['dur']/1000))
    fig.legend(handles=[Patch(color=c,label=n) for n,c in COLORS.items()],loc='outside lower center',ncol=5,fontsize=8)
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    fig.savefig(out/'original-backward.png',dpi=170);fig.savefig(out/'original-backward.svg')
    (ROOT/'analysis/plot-provenance.json').write_text(json.dumps(dict(diagnostic_only=True,rows=rows,
        note='Nested instrumented intervals include waits; do not sum or interpret as hardware utilization.'),indent=2))
    print(rows)


if __name__=='__main__':main()
