"""Scientific figures from audited actual points only (PNG/SVG/PDF)."""
import argparse
import json
import math
from pathlib import Path
import textwrap
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STYLE={
 'moonep':('MoonEP policy','#e98112','o'),
 'native':('Native tile/gradient policy','#266bc4','s'),
 'ultra_ep':('UltraEP policy','#20a077','^'),
 'echo':('ECHO policy','#8b5cf6','D'),
 'eplb_fixed':('EPLB fixed-home policy','#c23d46','v'),
 'step_time':('step_time policy','#182230','*')}
TARGETS=[.2,1,5,10,15,20]

def main():
    p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    data=json.loads((a.analysis/'points.json').read_text());points=data['points'];a.out.mkdir(exist_ok=True,parents=True)
    if not any(row.get('completed')==3 and row.get('unresolved_failures')==0 and 'mean_ms' in row for row in points):
        raise RuntimeError('no fully measured three-repeat point; do not generate a placeholder result figure')
    definitions=[('absolute-step-time','mean_ms','Complete training step (ms)',False),
        ('normalized-step-time','normalized_ms','Step time / one common MoonEP-policy baseline',False),
        ('speedup-vs-moonep-planner','speedup_vs_moonep','Same-MaxVio MoonEP-policy time / policy time',True),
        ('whole-card-hbm','peak_hbm_gib','Peak sampled whole-card HBM (GiB)',False)]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','pdf.fonttype':42})
    manifest={}
    exclusions=sum(row.get('excluded_contention_attempts',0) for row in points)
    init_retries=sum(row.get('recovered_initialization_failures',0) for row in points)
    unresolved=[f'{STYLE[row["planner"]][0]}@{row["maxvio"]:g}' for row in points if row.get('unresolved_failures')]
    for stem,key,ylabel,reference in definitions:
        fig,ax=plt.subplots(figsize=(10.8,6.6))
        for planner,(label,color,marker) in STYLE.items():
            subset={row['maxvio']:row for row in points if row['planner']==planner}
            if not subset:continue
            x=[];y=[];low=[];high=[]
            for target in TARGETS:
                row=subset.get(target,{})
                valid=row.get('state') in ('pass','pass_with_init_retry','pass_with_infrastructure_retry') and row.get('completed')==3 and key in row
                x.append(target);y.append(row[key] if valid else float('nan'))
                if valid and key=='mean_ms':
                    low.append(row[key]-row['repeat_min_ms']);high.append(row['repeat_max_ms']-row[key])
                elif valid and key=='normalized_ms':
                    denominator=data['common_denominator_ms']
                    low.append((row['mean_ms']-row['repeat_min_ms'])/denominator)
                    high.append((row['repeat_max_ms']-row['mean_ms'])/denominator)
                else:low.append(0);high.append(0)
            if not any(math.isfinite(value) for value in y):continue
            ax.errorbar(x,y,yerr=[low,high],label='MegaMoE + '+label,color=color,marker=marker,
                        linewidth=2.3 if planner=='step_time' else 1.7,markersize=7,capsize=3)
        if reference or key=='normalized_ms':ax.axhline(1,color='#6b7280',linewidth=.8,linestyle=':')
        if key=='peak_hbm_gib':
            ax.axhline(90,color='#b65c23',linewidth=1,linestyle='--',label='90 GiB common gate')
            ax.set_ylim(80,91)
        ax.set_xscale('log');ax.set_xticks(TARGETS,[f'{v:g}' for v in TARGETS]);ax.set_xlabel('Target MaxVio')
        ax.set_ylabel(ylabel);ax.grid(alpha=.2)
        ax.set_title('Ascend 950DT x8 | L2 EP8 E256 Top8 H7168 F2048 S16384/rank\nSame ordinary MegaMoE executor; framework shared expert')
        ax.legend(loc='best',fontsize=8.5,framealpha=.95)
        fig.subplots_adjust(bottom=.25 if exclusions or init_retries or unresolved else .22,top=.84,left=.11,right=.98)
        note='25 steps/process; 5 warmup + 20 measured; max wall over 8 ranks each step; 3 independent repeats.\n'
        notes={
            'peak_hbm_gib':'Maximum over all runs/ranks/driver samples, including warmup/cache. Samples may miss transients; y-axis starts at80 GiB.',
            'mean_ms':'All valid measured steps and repeats retained. Error bars: repeat min/max, not confidence intervals.',
            'normalized_ms':'One common MoonEP-policy MaxVio0.2 denominator. Error bars: scaled repeat min/max, not confidence intervals.',
            'speedup_vs_moonep':'Ratio of the two three-run mean times at each MaxVio. Repeat variability is shown in the absolute-time chart.'}
        note+=notes[key]
        if exclusions or init_retries:
            note+=f'\nAll raw attempts retained: {exclusions} foreign-activity exclusions; {init_retries} other initialization retries; linked supplements shown.'
        if unresolved:note+='\n'+textwrap.fill('Unresolved points omitted from curves and retained in CSV: '+', '.join(unresolved),width=145)
        fig.text(.5,.04,note,ha='center',fontsize=8.5,color='#4b5563')
        for extension in ('png','svg','pdf'):fig.savefig(a.out/f'{stem}.{extension}',dpi=160)
        manifest[stem]=dict(metric=key,files=[str(a.out/f'{stem}.{ext}') for ext in ('png','svg','pdf')],
            source_points_sha256=hashlib.sha256((a.analysis/'points.json').read_bytes()).hexdigest(),
            actual_measured_points_only=True,complete_three_repeat_points=sum(r.get('completed')==3 and r.get('unresolved_failures')==0 for r in points))
        plt.close(fig)
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(figures=len(manifest),output=str(a.out))))
if __name__=='__main__':main()
