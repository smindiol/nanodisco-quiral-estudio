"""Esquema del nanodisco con una textura analítica ilustrativa."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / 'figuras'
OUT.mkdir(exist_ok=True)

# Proyección oblicua de una textura analítica. El recorte conserva solo el disco.
from matplotlib.patches import Polygon

fig, ax = plt.subplots(figsize=(6.2, 3.9))
fig.subplots_adjust(0, 0, 1, 1)
angle = np.linspace(0, 2*np.pi, 240)
radius = 1.0
squash = .57
top = .17
bottom = -.17
rim = np.column_stack((np.cos(angle), squash*np.sin(angle)))
front = rim[angle >= np.pi]
side = np.vstack((front + [0, top], (front + [0, bottom])[::-1]))
ax.add_patch(Polygon(side, facecolor='#a4bdd0', edgecolor='#2a526d', linewidth=1.4))
for height in np.linspace(bottom, top, 5)[:-1]:
    ax.plot(front[:, 0], front[:, 1]+height, color='#53738b', lw=1)
x, y = np.meshgrid(np.linspace(-1, 1, 500), np.linspace(-1, 1, 500))
r = np.hypot(x, y)
sz = np.tanh((r-.37)/.115)
palette = plt.get_cmap('RdBu_r')
pixels = palette((sz+1)/2)
pixels[..., 3] = (r <= radius)
ax.imshow(pixels, extent=(-1, 1, top-squash, top+squash), origin='lower', zorder=2)
ax.plot(rim[:, 0], rim[:, 1]+top, color='#2a526d', lw=1.5, zorder=3)
x, y = np.meshgrid(np.arange(-.84, .85, .14), np.arange(-.84, .85, .14))
r = np.hypot(x, y)
inside = r < .92
x, y, r = x[inside], y[inside], r[inside]
sz = np.tanh((r-.37)/.115)
in_plane = np.sqrt(1-sz**2)
theta = np.arctan2(y,x)
u = in_plane*np.cos(theta)
v = squash*in_plane*np.sin(theta)+.82*sz
for linewidth, color, zorder in [(1.55, '#fff9ef', 4), (.78, '#172c42', 5)]:
    ax.quiver(x, squash*y+top, u, v, angles='xy', scale_units='xy', scale=8.8,
              width=.008 if zorder==4 else .005, color=color, zorder=zorder,
              headwidth=3.4, headlength=4)
ax.set(xlim=(-1.035, 1.035), ylim=(-.76,.88), aspect='equal')
ax.axis('off')
fig.savefig(OUT/'nanodisco_apertura.pdf', transparent=True, bbox_inches='tight', pad_inches=.015)
plt.close(fig)
