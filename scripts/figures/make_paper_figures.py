import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.linewidth":0.8})
NAVY="#2E5A88"; CRIM="#A41E34"; GREEN="#1F7A3D"; AMBER="#B06A00"; GREY="#6B7280"; SLATE="#3C475A"
def despine(ax):
    for s in ("top","right"): ax.spines[s].set_visible(False)
    ax.grid(axis="y",alpha=0.22,lw=0.8); ax.set_axisbelow(True)

# ---------- MAIN RESULTS: regime x filter precision (double dissociation) ----------
regimes=["Weak\n(synthetic UF)","Globally correlated\n(CFI)","Vulnerable subgroup\n(position)"]
super_=[0.956,0.722,0.824]; corr=[0.949,0.738,0.820]; bias=[0.945,0.721,0.872]
best=[0,1,2]  # index of best filter per regime: super, corrfilter, bias-cluster
x=np.arange(3); w=0.26
fig,ax=plt.subplots(figsize=(9.2,4.2))
b1=ax.bar(x-w,super_,w,label="supermajority",color=GREY)
b2=ax.bar(x,corr,w,label="CorrFilter (dependence-aware)",color=NAVY)
b3=ax.bar(x+w,bias,w,label="bias-cluster (vulnerability-aware)",color=GREEN)
groups=[b1,b2,b3]
for bars in groups:
    for r in bars:
        ax.text(r.get_x()+r.get_width()/2,r.get_height()+0.004,f"{r.get_height():.3f}",
                ha="center",va="bottom",fontsize=8.0,color="#333")
despine(ax); ax.set_xticks(x); ax.set_xticklabels(regimes,fontsize=10)
ax.set_ylabel("filtering precision (held-out, matched retention)")
ax.set_ylim(0.65,1.0)
ax.set_title("No single filter wins everywhere: each regime has a different best filter",
             fontsize=12.5,fontweight="bold",pad=12)
ax.legend(loc="lower center",ncol=3,frameon=False,fontsize=9.0,bbox_to_anchor=(0.5,-0.30))
fig.subplots_adjust(left=0.085,right=0.985,top=0.88,bottom=0.235)
fig.savefig("outputs/paper_figures/main_results.pdf"); fig.savefig("outputs/paper_figures/main_results.png",dpi=200)
plt.close(fig)

# ---------- CFI: false retention vs biased fraction ----------
frac=[0,25,50,75,100]
naive=[0.2665,0.2567,0.2738,0.3301,0.3227]; superm=[0.1801,0.1687,0.201,0.2653,0.2739]
fig,ax=plt.subplots(figsize=(6.6,4.2))
ax.plot(frac,naive,"-o",color=CRIM,lw=2.2,ms=6,label="naive majority")
ax.plot(frac,superm,"-s",color=GREY,lw=2.0,ms=5,label="supermajority-0.75")
ax.plot([100],[0.183],marker="*",ms=15,color=GREEN,ls="none",label="CorrFilter (adaptive $R$)")
ax.annotate("adaptive $R$\n0.18",xy=(100,0.183),xytext=(74,0.135),fontsize=8.8,color=GREEN,
            ha="center",arrowprops=dict(arrowstyle="-|>",color=GREEN,lw=1.1))
despine(ax); ax.set_xlabel("biased fraction of the bank (%)"); ax.set_ylabel("false-retention rate")
ax.set_title("Globally correlated co-failure (CFI)",fontsize=12,fontweight="bold")
ax.set_ylim(0.13,0.36); ax.legend(frameon=False,fontsize=9,loc="upper left")
fig.tight_layout(); fig.savefig("outputs/paper_figures/cfi_frr.pdf"); fig.savefig("outputs/paper_figures/cfi_frr.png",dpi=200); plt.close(fig)

# ---------- POSITION: drift (rho down, neff up) + gain over naive ----------
rate=[5,10,20]
rho=[0.1745,0.1348,0.0748]; neff=[3.89,4.519,5.976]; rho_clean=0.2168; neff_clean=3.389
cf_gain=[0.0,-0.6,-3.3]; bc_gain=[3.1,4.9,2.3]
fig,(axL,axR)=plt.subplots(1,2,figsize=(10.4,4.1))
# left: dual axis drift
axL.plot([0]+rate,[rho_clean]+rho,"-o",color=CRIM,lw=2.2,ms=6,label=r"$\bar\rho$ (mean corr.)")
axL.set_xlabel("position-poisoned fraction (%)"); axL.set_ylabel(r"mean correlation $\bar\rho$",color=CRIM)
axL.tick_params(axis="y",labelcolor=CRIM); axL.set_ylim(0,0.26)
for s in ("top",): axL.spines[s].set_visible(False)
axL.grid(axis="y",alpha=0.2); axL.set_axisbelow(True)
ax2=axL.twinx(); ax2.plot([0]+rate,[neff_clean]+neff,"-s",color=NAVY,lw=2.2,ms=6,label=r"$n_{\mathrm{eff}}$")
ax2.set_ylabel(r"effective size $n_{\mathrm{eff}}$",color=NAVY); ax2.tick_params(axis="y",labelcolor=NAVY)
ax2.set_ylim(3,6.3); ax2.spines["top"].set_visible(False)
axL.set_title(r"Attack lowers $\bar\rho$, raises $n_{\mathrm{eff}}$"+"\n(bank looks MORE independent)",fontsize=10.5,fontweight="bold")
axL.text(0,0.245,"clean",fontsize=8,color=GREY,ha="left")
# right: gain over naive
xr=np.arange(3); w=0.36
axR.axhline(0,color="#333",lw=0.9)
axR.axhline(5,color=GREEN,ls="--",lw=1.2,label="+5-pt target")
b1=axR.bar(xr-w/2,cf_gain,w,color=NAVY,label="CorrFilter")
b2=axR.bar(xr+w/2,bc_gain,w,color=GREEN,label="bias-cluster")
for bars in (b1,b2):
    for r in bars:
        h=r.get_height(); axR.text(r.get_x()+r.get_width()/2,h+(0.15 if h>=0 else -0.15),
            f"{h:+.1f}",ha="center",va="bottom" if h>=0 else "top",fontsize=8.2,color="#333")
for s in ("top","right"): axR.spines[s].set_visible(False)
axR.grid(axis="y",alpha=0.2); axR.set_axisbelow(True)
axR.set_xticks(xr); axR.set_xticklabels(["5%","10%","20%"]); axR.set_xlabel("position-poisoned fraction")
axR.set_ylabel("precision gain over naive (pts)"); axR.set_ylim(-4.5,6)
axR.set_title("CorrFilter hurts; bias-cluster helps",fontsize=10.5,fontweight="bold")
axR.legend(frameon=False,fontsize=8.6,loc="lower left")
fig.suptitle("Vulnerable-subgroup regime: position-aligned poisoning",fontsize=12.5,fontweight="bold",y=0.995)
fig.tight_layout(rect=[0,0,1,0.93]); fig.savefig("outputs/paper_figures/position_drift.pdf"); fig.savefig("outputs/paper_figures/position_drift.png",dpi=200); plt.close(fig)
print("done")

# ---------- LEARNABLE CLUSTER: precision vs calibration size (3 rates) ----------
import json
LC={
 0.05:dict(sizes=[25,50,100,200],
    learned=[0.9082,0.9111,0.9118,0.9175],learned_sd=[0.0067,0.007,0.0051,0.0066],
    labelfree=[0.9268,0.9275,0.9262,0.927],labelfree_sd=[0.0006,0.0017,0.001,0.0037],
    h1=0.9269,naive=0.8963),
 0.10:dict(sizes=[25,50,100,200],
    learned=[0.8613,0.8698,0.8724,0.8695],learned_sd=[0.0093,0.0094,0.0057,0.005],
    labelfree=[0.8717,0.8718,0.8726,0.8725],labelfree_sd=[0.001,0.0026,0.0023,0.0038],
    h1=0.8721,naive=0.8237),
 0.20:dict(sizes=[25,50,100,200],
    learned=[0.8066,0.7969,0.8151,0.8115],learned_sd=[0.009,0.017,0.0088,0.0099],
    labelfree=[0.8149,0.814,0.8149,0.8135],labelfree_sd=[0.0016,0.0013,0.0025,0.0035],
    h1=0.8143,naive=0.7911),
}
from matplotlib.lines import Line2D
fig,axes=plt.subplots(1,3,figsize=(11.2,3.7))
for ax,rate in zip(axes,[0.05,0.10,0.20]):
    d=LC[rate]; s=np.array(d["sizes"])
    lf=np.array(d["labelfree"]); lf_sd=np.array(d["labelfree_sd"])
    ln=np.array(d["learned"]);  ln_sd=np.array(d["learned_sd"])
    # shaded headroom: gap that the cluster filter recovers over naive consensus
    ax.axhspan(d["naive"],d["h1"],color=GREEN,alpha=0.07,lw=0,zorder=0)
    # reference levels (value labels placed in the clear right margin, below)
    ax.axhline(d["h1"],color=GREEN,ls="--",lw=1.4,zorder=1)
    ax.axhline(d["naive"],color=GREY,ls="--",lw=1.4,zorder=1)
    # measured series
    ax.errorbar(s,lf,yerr=lf_sd,fmt="-^",color=AMBER,lw=2.1,ms=7,capsize=3,zorder=3,
                label="label-free (position)")
    ax.errorbar(s,ln,yerr=ln_sd,fmt="-o",color=NAVY,lw=2.1,ms=6,capsize=3,zorder=3,
                label="learned ($\\sim$labels)")
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    ax.grid(axis="y",alpha=0.18); ax.set_axisbelow(True)
    # per-panel range that fully contains the error bars, with margin so nothing touches the frame
    hi=max((lf+lf_sd).max(),(ln+ln_sd).max(),d["h1"])+0.006
    lo=min((lf-lf_sd).min(),(ln-ln_sd).min(),d["naive"])-0.006
    ax.set_ylim(lo,hi)
    # reference value labels in the empty right margin (dashed lines run full width)
    xr=s.max()+16
    bbox=dict(facecolor="white",edgecolor="none",pad=0.6)
    ax.text(xr,d["h1"],f"H1 {d['h1']:.3f}",fontsize=7.8,color=GREEN,va="center",ha="left",bbox=bbox,zorder=4)
    ax.text(xr,d["naive"],f"naive {d['naive']:.3f}",fontsize=7.8,color=GREY,va="center",ha="left",bbox=bbox,zorder=4)
    ax.set_title(f"{int(rate*100)}% contamination",fontsize=11.5,fontweight="bold")
    ax.set_xlabel("calibration labels"); ax.set_xticks(s); ax.set_xlim(s.min()-14,s.max()+78)
axes[0].set_ylabel("precision (held-out)")
# single shared legend below all panels
handles=[Line2D([0],[0],color=NAVY,marker="o",lw=2.1,ms=6),
         Line2D([0],[0],color=AMBER,marker="^",lw=2.1,ms=7),
         Line2D([0],[0],color=GREEN,ls="--",lw=1.4),
         Line2D([0],[0],color=GREY,ls="--",lw=1.4)]
labels=["learned cluster ($\\sim$labels)","label-free cluster (position)",
        "H1-imported cluster","naive consensus"]
fig.legend(handles,labels,loc="lower center",ncol=4,frameon=False,fontsize=9.2,bbox_to_anchor=(0.5,-0.02))
fig.suptitle("The vulnerable cluster is learnable in-domain: label-free recovers the H1-imported cluster",
             fontsize=12.5,fontweight="bold",y=1.0)
fig.tight_layout(rect=[0,0.06,1,0.93])
fig.savefig("outputs/paper_figures/learnable_cluster.pdf"); fig.savefig("outputs/paper_figures/learnable_cluster.png",dpi=200); plt.close(fig)
print("lc done")
