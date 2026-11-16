"""Scientific plot of the explicitly isolated diagnostic, not training timing."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_matrix import STYLE,TARGETS
ROOT=Path(__file__).resolve().parents[1]
def main():
    data=json.loads((ROOT/'analysis/planning-overhead.json').read_text());assert data['passed']
    fig,ax=plt.subplots(figsize=(10.8,6.6))
    for policy,(label,color,marker) in STYLE.items():
        points={x['maxvio']:x for x in data['points'] if x['policy']==policy};assert set(points)==set(TARGETS)
        ax.plot(TARGETS,[points[v]['mean_wall_ms'] for v in TARGETS],color=color,marker=marker,label=label)
    ax.set_xscale('log');ax.set_xticks(TARGETS,[f'{v:g}' for v in TARGETS]);ax.grid(alpha=.2);ax.legend(fontsize=8.5)
    ax.set_xlabel('Target MaxVio');ax.set_ylabel('Isolated online planning call (ms; max8-rank wall)')
    ax.set_title('Current-route statistics + policy + common compiler/validator\nAscend950DT x8 | E256 Top8 H7168 F2048 S16384/rank')
    fig.subplots_adjust(bottom=.23,top=.84)
    fig.text(.5,.04,'One sequential diagnostic process;25 steps/point,first5 warmup;40 measured calls per policy/point.\nPrior rank alignment and subsequent CPU validation/file output excluded. This is not a formal training step score.\nThe complete training measurements already include all online planning and checkpoint replay costs.',ha='center',fontsize=8.5,color='#4b5563')
    for ext in ('png','svg','pdf'):fig.savefig(ROOT/f'figures/isolated-planning-overhead.{ext}',dpi=160)
    plt.close(fig)
if __name__=='__main__':main()
