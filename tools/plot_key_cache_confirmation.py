"""Export the independently audited process-level paired confidence intervals."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def main():
    data=ROOT/'analysis/key-cache-confirmation-summary.json';d=json.loads(data.read_text())
    if not d['passed'] or d['repeats']!=5 or d['new_planner']:raise RuntimeError('Unexpected confirmation scope')
    fig,ax=plt.subplots(figsize=(8.4,3.9));labels={'native':'Native','ultra_ep':'UltraEP','eplb_fixed':'EPLB-fixed'}
    for i,r in enumerate(d['summaries']):
        y=2-i;value=r['geometric_mean_step_reduction']*100;lo,hi=[v*100 for v in r['reduction_log_t95']]
        ax.errorbar(value,y,xerr=[[value-lo],[hi-value]],fmt='o',color='#2176ae',capsize=5,markersize=7)
        ax.text(value+.55,y+.11,f'{value:.3f}%  [{lo:.3f}, {hi:.3f}]',fontsize=10)
    ax.set_yticks([2,1,0],[labels[r['planner']] for r in d['summaries']]);ax.set_xlim(0,25);ax.set_ylim(-.6,2.6)
    ax.set_xlabel('Complete-step reduction (%)');ax.set_title('Exact gradient-key cache: five fresh paired processes')
    ax.grid(axis='x',alpha=.18);ax.spines[['top','right']].set_visible(False)
    fig.text(.5,.025,'MaxVio20; 25 updates/process, 5 warmup; unadjusted paired log-t95, df4. Executor fix, not a new planner.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.07,1,1));out=ROOT/'figures/gradient-key-cache-confirmation.png'
    fig.savefig(out,dpi=180);fig.savefig(out.with_suffix('.svg'))
    p=out.with_suffix('.svg');p.write_text('\n'.join(l.rstrip() for l in p.read_text().splitlines())+'\n',newline='\n')
    (ROOT/'figures/gradient-key-cache-confirmation-provenance.json').write_text(json.dumps(dict(
        input_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),matplotlib=matplotlib.__version__,
        output_sha256=hashlib.sha256(out.read_bytes()).hexdigest()),indent=2))
    print(out)


if __name__=='__main__':main()
