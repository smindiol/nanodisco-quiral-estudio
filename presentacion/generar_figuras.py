"""Figuras de exposición a partir de resultados conservados, sin nuevas simulaciones.

Los dos ejemplos vectoriales del estado 6 se conservan en datos/; su selección
se documenta allí. Regenerar la presentación no necesita la base cruda.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OUT=HERE/'figuras'
OUT.mkdir(exist_ok=True)
A=json.loads((ROOT/'resultados/auditoria/auditoria.json').read_text())
EM=json.loads((ROOT/'resultados/informe/mezcla_em.json').read_text())
COLORS=['#0066b3','#ce5421','#1c875a','#8c53ad','#bd3654','#00828f','#9b7300','#35455a']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'axes.titlesize':17,
    'axes.labelsize':15,'xtick.labelsize':12,'ytick.labelsize':12,'legend.fontsize':11,
    'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight','pdf.fonttype':42})
S={(v['estado'],round(v['T'],1)):v for v in A['texturas']}
D={e:dict(np.load(ROOT/f'resultados/pasos_mc/estado_{e}.npz')) for e in range(1,9)}

def save(fig,name):
    fig.savefig(OUT/(name+'.pdf'))
    plt.close(fig)

def texture(ax,e,t,copy=0):
    q=S[e,t];mask=np.asarray(q['mascara'],bool);z=np.asarray(q['sz'][copy])
    ax.imshow(np.ma.array(z,mask=~mask),cmap='RdBu_r',vmin=-1,vmax=1,
              origin='lower',interpolation='nearest',extent=(-19.5,19.5,-19.5,19.5))
    ax.set(xticks=[],yticks=[],aspect='equal')
    for spine in ax.spines.values():spine.set_visible(False)

fig,axes=plt.subplots(2,4,figsize=(11,4.7),layout='constrained')
for e,ax in enumerate(axes.flat,1):
    texture(ax,e,6.)
    st=A['curvas'][e-1]
    ax.set_title(f"E{e}: D={st['kDM']:g}, "+r"$H_{\mathrm{ex}}$"+f"={st['H']:g}",fontsize=13,pad=4)
fig.colorbar(ScalarMappable(norm=Normalize(-1,1),cmap='RdBu_r'),ax=axes.ravel().tolist(),
             shrink=.8,pad=.02,label=r'$S_z$')
save(fig,'ocho_estados')

temperaturas_ejemplo = [20., 14.3, 10., 6.]
fig,axes=plt.subplots(3,4,figsize=(10,4.2))
fig.subplots_adjust(left=.13,right=.92,bottom=.04,top=.87,hspace=.15,wspace=.43)
for row in range(3):
    for col,t in enumerate(temperaturas_ejemplo):
        ax=axes[row,col]
        texture(ax,6,t,copy=row)
        if row==0:
            ax.set_title(f'{t:g} K',fontsize=15)
        if col==0:
            ax.text(-.19,.5,f'Copia {row+1}',transform=ax.transAxes,
                    va='center',ha='right',fontsize=13)
        if col<3:
            ax.annotate('',xy=(1.47,.5),xytext=(1.02,.5),xycoords='axes fraction',
                        arrowprops=dict(arrowstyle='->',color='#546d80',lw=1.5),
                        annotation_clip=False)
cax=fig.add_axes([.95,.13,.016,.65])
fig.colorbar(ScalarMappable(norm=Normalize(-1,1),cmap='RdBu_r'),
             cax=cax,label=r'$S_z$',ticks=[-1,0,1])
save(fig,'caminos_enfriamiento')

fig,axes=plt.subplots(1,2,figsize=(11,3.6),layout='constrained')
for st,color in zip(A['curvas'],COLORS):
    label=f"E{st['estado']}"
    axes[0].plot(st['T'],st['energia_meV_por_espin'],color=color,lw=2,label=label)
    axes[1].plot(st['T'],st['c_recalculado'],color=color,lw=2)
for ax in axes:
    ax.set(xlim=(20,6),xlabel='Temperatura (K)');ax.grid(alpha=.18)
    ax.axvspan(13.5,14.7,color='#b9dcf5',alpha=.32,zorder=-1)
axes[0].set(ylabel='Energía por espín',title='La energía desciende al enfriar')
axes[1].set(ylabel='c por espín',title='La fluctuación tiene un máximo')
axes[0].legend(ncol=4,loc='upper right',handlelength=1,columnspacing=.7)
save(fig,'curvas_termicas')

fig,ax=plt.subplots(figsize=(7,4.3),layout='constrained')
intervals=[]
for line in (ROOT/'resultados/tablas/T_estrella.md').read_text().splitlines():
    c=[x.strip() for x in line.strip('|').split('|')]
    if c and c[0].isdigit():
        e=int(c[0]);t=float(c[4]);lo,hi=map(float,c[7].strip('[]').split(','));intervals.append((e,t,lo,hi))
for e,t,lo,hi in intervals:
    st=A['curvas'][e-1]
    ax.errorbar(t,e,xerr=[[t-lo],[hi-t]],fmt='o' if st['H']==0 else 'D',color=COLORS[e-1],
                lw=2.5,markersize=8,capsize=4)
    ax.text(14.86,e,f'{t:.2f}',va='center',fontsize=12)
ax.set(yticks=range(1,9),yticklabels=[f'E{e}' for e in range(1,9)],ylim=(8.6,.4),
       xlabel=r'$T^*$ (K)',xlim=(12.9,15.25))
ax.grid(axis='x',alpha=.2)
ax.set_title('IC bootstrap percentil del 95 %',fontsize=17)
save(fig,'bootstrap_intervalos')

fig,ax=plt.subplots(figsize=(10,3.5),layout='constrained')
rng=np.random.default_rng(43)
for j,t in enumerate(temperaturas_ejemplo):
    means=D[6][f'mediasM_{t:.1f}']
    ax.scatter(j+rng.uniform(-.16,.16,len(means)),means,s=21,
               color='#0072B2',alpha=.55,edgecolors='none')
    ax.hlines(np.median(means),j-.21,j+.21,color='#b64b19',lw=2.3)
ax.set(xticks=range(4),xticklabels=[f'{t:g} K' for t in temperaturas_ejemplo],
       xlabel='Temperaturas en orden de enfriamiento',
       ylabel='Magnetización media por copia',ylim=(0,.78))
ax.grid(axis='y',alpha=.22)
save(fig,'magnetizacion_por_temperatura')

fig,axes=plt.subplots(1,2,figsize=(11,3.6),layout='constrained')
for e,color in enumerate(COLORS,1):
    d=D[e]
    axes[0].semilogy(d['T'],np.maximum(d['rhat_E']-1,1e-6),color=color,lw=1.8,label=f'E{e}')
    axes[1].semilogy(d['T'],np.maximum(d['rhat_M']-1,1e-6),color=color,lw=1.8)
for ax in axes:
    ax.axhline(.01,color='black',ls='--',lw=1.4)
    ax.set(xlim=(20,6),xlabel='Temperatura (K)',ylim=(1e-5,3))
    ax.grid(alpha=.15)
axes[0].set(title='Energía',ylabel=r'$\widehat R-1$ (escala log)')
axes[1].set(title='Magnetización')
axes[0].legend(ncol=4,fontsize=10,loc='upper left',handlelength=1,columnspacing=.7)
save(fig,'rhat_temperatura')

fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
for e,color in enumerate(COLORS,1):
    st=A['curvas'][e-1];maximum=float(np.max(D[e]['tau_M']))
    ax.scatter(st['kDM'],maximum,c=color,s=100,marker='o' if st['H']==0 else 'D',zorder=3)
    ax.annotate(f'E{e}',(st['kDM'],maximum),xytext=(5,5),textcoords='offset points',fontsize=12)
ax.set(xlabel=r'Interacción quiral $D=k_{DM}$',ylabel=r'Máximo de $\tau_m$ (barridos)',ylim=(0,1100),xlim=(-.01,1.12))
ax.grid(alpha=.2)
save(fig,'costo_quiral')

examples=json.loads((HERE/'datos/ejemplos_estado6.json').read_text())
fig=plt.figure(figsize=(11,3.5),layout='constrained')
gs=fig.add_gridspec(1,3,width_ratios=[1,1,1.5])
for j in range(2):
    ax=fig.add_subplot(gs[0,j]);grid=np.asarray(examples['spins_capa_central'][j]);mask=np.asarray(examples['mascara'],bool)
    ax.imshow(np.ma.array(grid[:,:,2],mask=~mask),cmap='RdBu_r',vmin=-1,vmax=1,origin='lower')
    yy,xx=np.mgrid[0:39:3,0:39:3];ok=mask[yy,xx]
    ax.quiver(xx[ok],yy[ok],grid[yy[ok],xx[ok],0],grid[yy[ok],xx[ok],1],color='#1d2730',
              scale=9,width=.006,headwidth=3)
    ax.set(xticks=[],yticks=[])
    for spine in ax.spines.values():spine.set_visible(False)
    ax.set_title(f"Copia {examples['copias'][j]}\n"+r'$\bar m_j$'+f" = {examples['m_media'][j]:.3f}",fontsize=14)
ax=fig.add_subplot(gs[0,2]);x=D[6]['mediasM_6.0'];em=EM['6'];v=np.linspace(.54,.74,500)
ax.hist(x,bins=np.linspace(.54,.74,26),density=True,color='#b2d6ee',edgecolor='white')
pdf=np.zeros_like(v)
for w,mu,sd,color in zip(em['w'],em['mu'],em['sd'],['#0066b3','#ce5421']):
    p=w*np.exp(-.5*((v-mu)/sd)**2)/(sd*np.sqrt(2*np.pi));pdf+=p
    ax.plot(v,p,color=color,lw=2,label=f'{w:.0%}, media {mu:.3f}')
ax.plot(v,pdf,color='#273849',lw=1.4,ls='--')
ax.set(xlabel=r'Media del camino $\bar m_j$',ylabel='Densidad',title='100 caminos a 6 K')
ax.legend(fontsize=10)
save(fig,'poblaciones_estado6')

from runpy import run_path
run_path(str(HERE/'figuras_3d.py'),run_name='__main__')

provenance=dict(datos='Resultados derivados conservados en resultados/. No se ejecutaron nuevas cadenas ni un nuevo bootstrap.',
    texturas='Capa central, copia 1 en ocho_estados. S_z con escala común [-1,1].',
    apertura='Textura 3D analítica ilustrativa de un disco magnético.',
    caminos='Estado 6, copias 1 a 3, instantáneas finales en cuatro temperaturas.',
    ejemplos_estado6=examples['seleccion'],copias_estado6=examples['copias'],
    curvas='Malla medida de 0.1 K; T* e intervalos del bootstrap original de 50000 réplicas.',
    figuras=sorted(p.name for p in OUT.glob('*.pdf')))
(HERE/'datos/procedencia_figuras.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
print(f'Figuras generadas: {len(provenance["figuras"])}. No requieren la base cruda para regenerarse.')
