import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10})

# desaturated, academic palette
regimes = [
 dict(name="Dominant vulnerable subgroup", x=0.5, col="#A6343C", tint="#FAF1F1",
      drho=r"$\Delta\bar\rho<0$:  $\bar\rho$ falls ($-0.08$)", neff=r"$n_{\mathrm{eff}}$ rises  ($3.4\!\to\!4.6$)",
      cons="consensus fails", cf="CorrFilter hurts", cfk=-1,
      mit="bias-cluster filter", ex="position-aligned poisoning"),
 dict(name="Weak dependence", x=3.0, col="#3F7A5A", tint="#F0F5F2",
      drho=r"$\Delta\bar\rho\approx 0$:  $\bar\rho\approx$ clean", neff=r"$n_{\mathrm{eff}}\approx$ clean  ($3.3$)",
      cons="consensus reliable", cf="CorrFilter neutral", cfk=0,
      mit="supermajority", ex="synthetic-poisoned UltraFeedback"),
 dict(name="Globally correlated co-failure", x=5.5, col="#B5852A", tint="#F8F3E9",
      drho=r"$\Delta\bar\rho>0$:  $\bar\rho$ rises ($+0.11$)", neff=r"$n_{\mathrm{eff}}$ falls  ($3.4\!\to\!2.6$)",
      cons="confidently wrong", cf="CorrFilter helps", cfk=1,
      mit="CorrFilter (adaptive $R$)", ex="correlated-failure injection"),
]

fig, ax = plt.subplots(figsize=(10.0, 4.35))
ax.set_xlim(-0.45, 6.55); ax.set_ylim(-0.5, 4.55); ax.axis("off")

cardw, cardh, ybot = 2.0, 2.95, 0.30

# --- diagnostic axis (thin, restrained) ---
axis_y = 4.18
ax.plot([-0.2, 6.3],[axis_y,axis_y], color="#888", lw=1.0, solid_capstyle="round")
for xx,dx in [(-0.2,-1),(6.3,1)]:
    ax.plot([xx],[axis_y], marker=(3,0,90 if dx>0 else -90), ms=7, color="#888")
ax.plot([3.0,3.0],[axis_y-0.07,axis_y+0.07], color="#888", lw=1.0)
ax.text(3.0, axis_y+0.13, "clean bank", color="#777", fontsize=8.5, ha="center", va="bottom")
ax.text(-0.2, axis_y+0.13, r"$\bar\rho$ falls", color="#A6343C", fontsize=9.5, ha="left", va="bottom")
ax.text(6.3, axis_y+0.13, r"$\bar\rho$ rises", color="#B5852A", fontsize=9.5, ha="right", va="bottom")
ax.text(3.0, axis_y-0.27, "diagnostic axis: correlation drift relative to the clean bank",
        color="#666", fontsize=8.8, ha="center", va="top", style="italic")

def row(ax,x,y,txt,c="#1f1f1f",fs=9.2,it=False,bold=False):
    ax.text(x,y,txt,ha="center",va="center",fontsize=fs,color=c,
            style="italic" if it else "normal", fontweight="bold" if bold else "normal")

for r in regimes:
    x=r["x"]; left=x-cardw/2
    # card body: white with thin border + faint tint
    ax.add_patch(FancyBboxPatch((left,ybot),cardw,cardh,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        fc="white",ec="#d9d9d9",lw=1.0,zorder=2))
    ax.add_patch(FancyBboxPatch((left,ybot),cardw,cardh,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        fc=r["tint"],ec="none",lw=0,zorder=1))
    # colored top rule
    ax.add_patch(Rectangle((left+0.02,ybot+cardh-0.06),cardw-0.04,0.07,
        fc=r["col"],ec="none",zorder=4))
    yy=ybot+cardh-0.36
    row(ax,x,yy,r["name"],c=r["col"],fs=10.0,bold=True); yy-=0.46
    row(ax,x,yy,r["drho"],fs=9.0); yy-=0.40
    row(ax,x,yy,r["neff"],fs=9.0); yy-=0.44
    row(ax,x,yy,r["cons"],c="#444",fs=9.0,it=True); yy-=0.44
    cfcol = {1:"#2E7D32",-1:"#A6343C",0:"#777"}[r["cfk"]]
    row(ax,x,yy,r["cf"],c=cfcol,fs=9.3,bold=True); yy-=0.46
    # mitigation pill: white fill, accent border + text
    ax.add_patch(FancyBboxPatch((x-0.92,yy-0.16),1.84,0.34,
        boxstyle="round,pad=0.02,rounding_size=0.10",fc="white",ec=r["col"],lw=1.2,zorder=5))
    ax.text(x,yy+0.01,r["mit"],ha="center",va="center",color=r["col"],fontsize=8.7,fontweight="bold",zorder=6)
    ax.text(x,ybot-0.14,r["ex"],ha="center",va="top",fontsize=8.0,color="#888",style="italic")

fig.text(0.5,0.018,
  "The two failure regimes move in opposite diagnostic directions: a taxonomy of distinct mechanisms, not a single severity scale.",
  ha="center",fontsize=9.8,color="#1a1a1a")
plt.subplots_adjust(left=0.008,right=0.992,top=0.995,bottom=0.075)
fig.savefig("outputs/taxonomy_validation/taxonomy_hero.png",dpi=200)
fig.savefig("outputs/taxonomy_validation/taxonomy_hero.pdf")
print("saved")
