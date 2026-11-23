"""Geometría introductoria y vista 3D de una configuración real conservada.

La vista completa dibuja los 5425 sitios. Las flechas se submuestrean para evitar
superposición y el eje z se amplía cuatro veces en la visualización. No altera
la configuración, su geometría numérica ni los resultados del análisis.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE=Path(__file__).resolve().parent
OUT=HERE/'figuras'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'pdf.fonttype':42,
                     'axes.labelsize':14,'savefig.bbox':'tight'})

# Esquema geométrico general; no representa una configuración simulada.
fig=plt.figure(figsize=(6,4.3))
ax=fig.add_subplot(projection='3d')
theta=np.linspace(0,2*np.pi,100)
for z in [0,.22]:
    poly=np.column_stack((np.cos(theta),np.sin(theta),np.full_like(theta,z)))
    ax.add_collection3d(Poly3DCollection([poly],facecolor='#a9cce7' if z else '#dce7ef',
                                        edgecolor='#185581',alpha=.85,linewidth=1))
tt,zz=np.meshgrid(theta,[0,.22])
ax.plot_surface(np.cos(tt),np.sin(tt),zz,color='#70a9d0',alpha=.45,shade=False)
ax.plot([0,1],[0,0],[.24,.24],color='#103e61',lw=2)
ax.text(.3,-.25,.43,'radio R',fontsize=15,color='#103e61')
ax.plot([1.08,1.08],[0,0],[0,.22],color='#103e61',lw=2)
ax.text(1.15,0,.1,'L capas',fontsize=14,color='#103e61')
ax.set(xlim=(-1.2,1.5),ylim=(-1.2,1.2),zlim=(-.05,.65))
ax.set_box_aspect((2.7,2.4,.9));ax.view_init(elev=24,azim=-58);ax.set_axis_off()
fig.savefig(OUT/'geometria_nanodisco.pdf');plt.close(fig)

with np.load(HERE/'datos/estado6_copia63_3d.npz') as d:
    mask=d['mascara'].astype(bool);spins=d['spins'];N=int(d['N'])
assert spins.shape==(N,3) and N==5425 and mask.sum()==N
zyx=np.argwhere(mask)
xyz=np.column_stack((zyx[:,2]-19,zyx[:,1]-19,zyx[:,0]))
norm=Normalize(-1,1);cmap=plt.get_cmap('RdBu_r')

fig=plt.figure(figsize=(11,4.5),layout='constrained')
gs=fig.add_gridspec(1,2,width_ratios=[1.6,1])
ax=fig.add_subplot(gs[0,0],projection='3d')
ax.scatter(*xyz.T,c=spins[:,2],cmap=cmap,norm=norm,s=3,alpha=.35,depthshade=False,rasterized=False)
choose=(zyx[:,1]%4==0)&(zyx[:,2]%4==0)
p=xyz[choose];v=spins[choose]
ax.quiver(*p.T,*v.T,length=.8,normalize=False,colors=cmap(norm(v[:,2])),
          linewidths=.65,arrow_length_ratio=.3)
for z in range(5):
    ax.plot(18.5*np.cos(theta),18.5*np.sin(theta),np.full_like(theta,z),
            color='#4a6477',alpha=.55,lw=.8)
ax.set(xlim=(-21,21),ylim=(-21,21),zlim=(-.8,5.2),xlabel='x (sitios)',ylabel='y (sitios)',zlabel='capa z',
       zticks=range(5),xticks=[-18,0,18],yticks=[-18,0,18])
ax.set_box_aspect((42,42,24));ax.view_init(elev=23,azim=-53)
ax.set_title('Disco completo: cinco capas',fontsize=17,pad=0)
ax.tick_params(labelsize=10,pad=0)
for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:axis.pane.fill=False
ax.grid(alpha=.15)

ax2=fig.add_subplot(gs[0,1],projection='3d')
choose=(abs(xyz[:,0])<=5)&(abs(xyz[:,1])<=5)&(xyz[:,2]==2)&(zyx[:,1]%2==1)&(zyx[:,2]%2==1)
p=xyz[choose];v=spins[choose]
ax2.scatter(*p.T,c=spins[choose,2],cmap=cmap,norm=norm,s=15,depthshade=False)
ax2.quiver(*p.T,*v.T,length=1.25,normalize=False,colors=cmap(norm(v[:,2])),
           linewidths=1.3,arrow_length_ratio=.3)
ax2.set(xlim=(-6,6),ylim=(-6,6),zlim=(.4,3.6),xlabel='x',ylabel='y',zlabel='z',
        xticks=[-4,0,4],yticks=[-4,0,4],zticks=[1,2,3])
ax2.set_box_aspect((12,12,12.8));ax2.view_init(elev=32,azim=-55)
ax2.set_title('Detalle de los espines',fontsize=17,pad=0)
ax2.tick_params(labelsize=10,pad=0)
for axis in [ax2.xaxis,ax2.yaxis,ax2.zaxis]:axis.pane.fill=False
fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=[ax,ax2],shrink=.7,pad=.01,label=r'$S_z$')
fig.savefig(OUT/'nanodisco_completo_3d.pdf');plt.close(fig)
print('Vista 3D: todos los sitios; flechas submuestreadas. Geometría general guardada aparte.')
