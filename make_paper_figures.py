import os; os.makedirs('paper_figures', exist_ok=True)
import matplotlib
matplotlib.use('Agg')
import os; os.makedirs('paper_figures', exist_ok=True)
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

NAVY='#2C2C2A'; PUR='#534AB7'; PURL='#EEEDFE'; TEAL='#0F6E56'; TEALL='#E1F5EE'; GRAY='#5F5E5A'; GRAYL='#F1EFE8'; AMB='#854F0B'; AMBL='#FAEEDA'
plt.rcParams['font.family']='DejaVu Sans'

# ---------- Graphical abstract ----------
fig,ax=plt.subplots(figsize=(10,7.6)); ax.set_xlim(0,10); ax.set_ylim(0,7.6); ax.axis('off')
def box(x,y,w,h,fc,ec,title,lines,num=None,dash=False,tsz=10.4,lsz=9.6):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=0.12',fc=fc,ec=ec,lw=1.6,ls='--' if dash else '-'))
    ty=y+h-0.32
    if num is not None:
        ax.add_patch(Circle((x+0.28,ty),0.17,fc=ec,ec='none')); ax.text(x+0.28,ty,str(num),ha='center',va='center',color='white',fontsize=10,fontweight='bold')
        ax.text(x+0.55,ty,title,ha='left',va='center',fontsize=tsz,fontweight='bold',color=NAVY)
    else:
        ax.text(x+0.18,ty,title,ha='left',va='center',fontsize=tsz,fontweight='bold',color=NAVY)
    for i,l in enumerate(lines):
        ax.text(x+0.2,ty-0.42-i*0.3,l,ha='left',va='center',fontsize=lsz,color=NAVY)
def arr(x1,y1,x2,y2,c=GRAY,dash=False):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=16,lw=1.8,color=c,ls='--' if dash else '-'))
ax.text(5,7.35,'MateriVerse: physics-grounded generative design of candidate superconductors',ha='center',va='center',fontsize=13.5,fontweight='bold',color=NAVY)
# row A
box(0.25,5.2,2.8,1.6,GRAYL,GRAY,'Data',['824,516 crystal structures','Materials Project · OQMD ·','NOMAD (80/10/10 split)'])
box(3.6,5.2,2.8,1.6,PURL,PUR,'Equivariant DiT',['SE(3)-equivariant diffusion','Pauli-aware attention','Projective symmetry layers'],num=1)
box(6.95,5.2,2.8,1.6,TEALL,TEAL,'Physics-aware RL',['DFT / MLFF rewards:','stability, synthesizability,','predicted Tc'],num=2)
arr(3.07,6.0,3.58,6.0); arr(6.42,6.0,6.93,6.0)
# row B (right to left)
box(6.95,3.2,2.8,1.6,TEALL,TEAL,'Pareto optimisation',['Max Tc, Hc2','Min ΔE_hull, rare earths','Optimal-transport sampling'],num=3)
box(3.6,3.2,2.8,1.6,TEALL,TEAL,'Active learning',['Epistemic uncertainty','Conformal calibration','MACE-MP-0 re-evaluation'],num=4)
box(0.25,3.2,2.8,1.6,GRAYL,GRAY,'Ranked candidates',['12,509 generated in 72 h','48 with predicted Tc > 77 K','(model predictions)'])
arr(8.35,5.18,8.35,4.82); arr(6.93,4.0,6.42,4.0); arr(3.58,4.0,3.07,4.0)
# protocol band
ax.add_patch(FancyBboxPatch((0.25,1.95),9.5,0.95,boxstyle='round,pad=0.02,rounding_size=0.12',fc=AMBL,ec=AMB,lw=1.6,ls='--'))
ax.text(0.45,2.6,'Stage 5 – experimental validation protocol (proposed, not yet executed)',fontsize=11,fontweight='bold',color=AMB,va='center')
ax.text(0.45,2.22,'XRD + Rietveld · zero resistance · bulk diamagnetic shielding · independent replication → feedback to Stages 1–4',fontsize=8.7,color=NAVY,va='center')
arr(1.65,3.18,1.65,2.92,c=AMB,dash=True)
# metrics
ax.text(0.25,1.68,'Computational benchmark (this study)',fontsize=10.5,fontweight='bold',color=GRAY,va='center')
tiles=[('174',['candidates','per hour']),('89%',['stable structures','(ΔE_hull < 50 meV)']),('96%',['space group','retained']),('0.3',['CPU-h of DFT','per candidate']),('3.8 K',['Tc error (MAE),','held-out data'])]
for i,(v,ls) in enumerate(tiles):
    x=0.3+i*1.9
    ax.add_patch(FancyBboxPatch((x,0.3),1.74,1.2,boxstyle='round,pad=0.02,rounding_size=0.1',fc='white',ec=PUR,lw=1.4))
    ax.text(x+0.87,1.12,v,ha='center',va='center',fontsize=17,fontweight='bold',color=PUR)
    ax.text(x+0.87,0.74,ls[0],ha='center',va='center',fontsize=8.4,color=NAVY); ax.text(x+0.87,0.52,ls[1],ha='center',va='center',fontsize=8.0 if len(ls[1])>16 else 8.4,color=NAVY)
fig.savefig('paper_figures/ga.png',dpi=220,bbox_inches='tight',facecolor='white'); plt.close(fig)

# ---------- Benchmark figure ----------
names=['MateriVerse','CDVAE','GNoME','Equivariant\ndiffusion']
thr=[174,1.2,3.8,42]; stab=[89,67,81,74]; cost=[0.3,25,8,4.5]
eff=[c/(s/100) for c,s in zip(cost,stab)]
cols=[TEAL,'#B4B2A9','#B4B2A9','#B4B2A9']
fig,axs=plt.subplots(1,3,figsize=(10,3.5))
def bars(a,vals,title,ylabel,log=False,fmt='{:g}'):
    b=a.bar(range(4),vals,color=cols,edgecolor=NAVY,linewidth=0.6,width=0.65)
    if log: a.set_yscale('log')
    a.set_title(title,fontsize=10.5,fontweight='bold',color=NAVY); a.set_ylabel(ylabel,fontsize=9)
    a.set_xticks(range(4)); a.set_xticklabels(names,fontsize=8)
    a.tick_params(axis='y',labelsize=8)
    for s in ['top','right']: a.spines[s].set_visible(False)
    for i,v in enumerate(vals):
        a.text(i,v*(1.12 if log else 1.0)+(0 if log else max(vals)*0.015),fmt.format(v),ha='center',va='bottom',fontsize=8.5,color=NAVY)
bars(axs[0],thr,'(a) Throughput','candidates / hour (log)',log=True)
bars(axs[1],stab,'(b) Stability rate','% with ΔE_hull < 50 meV/atom')
axs[1].set_ylim(0,105)
bars(axs[2],[0.34,37.3,9.9,6.1],'(c) Effective DFT cost','CPU-h per stable candidate (log)',log=True)
axs[0].set_ylim(0.6,400); axs[2].set_ylim(0.15,90)
fig.tight_layout(); fig.savefig('paper_figures/bench.png',dpi=220,facecolor='white'); plt.close(fig)
print([round(e,2) for e in eff])
