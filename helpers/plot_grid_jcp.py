#!/usr/bin/env python3
import os, glob, argparse, numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def vec_mag(uv): return np.sqrt(uv[0]**2 + uv[1]**2)
def nmse(a,b):
    num=np.sum((a-b)**2, dtype=np.float64); den=max(np.sum(b**2, dtype=np.float64),1e-12)
    return float(num/den)

def find_sample(root, model, idx, seed=None):
    fn=f"sample_{idx:03d}.npz"
    if seed is not None:
        p=os.path.join(root, f"{model}_seed{seed}", fn)
        if os.path.exists(p): return p
    hits=sorted(glob.glob(os.path.join(root, f"{model}_seed*", fn)))
    if hits: return hits[0]
    p=os.path.join(root, model, fn)
    return p if os.path.exists(p) else None

def load_panel(root, model, idx, field, seed=None):
    p=find_sample(root, model, idx, seed)
    if p is None:
        raise FileNotFoundError(f"Missing {model} sample_{idx:03d}.npz in {root}")
    D=np.load(p)
    gt=D["gt"].astype(np.float32); pr=D["pred"].astype(np.float32)
    if field=="mag":
        GT,PR=vec_mag(gt),vec_mag(pr); label=r"$\|\mathrm{velocity}\|$"
    elif field=="u": GT,PR=gt[0],pr[0]; label="u"
    elif field=="v": GT,PR=gt[1],pr[1]; label="v"
    else: raise ValueError("field must be one of: mag|u|v")
    return GT,PR,nmse(PR,GT),label

def main():
    ap=argparse.ArgumentParser("3x4 CFD Benchmark vs Pred")
    # roots for each case
    ap.add_argument("--cyl_root", required=True)
    ap.add_argument("--tube_root", required=True)
    ap.add_argument("--cav_root",  required=True)
    # models & sample indices
    ap.add_argument("--models", nargs=2, default=["PINN","PIBERT"])
    ap.add_argument("--idx", type=int, default=None)
    ap.add_argument("--idx_cyl", type=int, default=None)
    ap.add_argument("--idx_tube", type=int, default=None)
    ap.add_argument("--idx_cav", type=int, default=None)
    ap.add_argument("--seed", default=None)
    # appearance
    ap.add_argument("--field", choices=["mag","u","v"], default="mag")
    ap.add_argument("--cmap", default="coolwarm")
    ap.add_argument("--show_nmse", action="store_true")
    ap.add_argument("--outer_ticks_only", action="store_true", default=True)
    # figure margins / spacing
    ap.add_argument("--left",   type=float, default=0.12)
    ap.add_argument("--right",  type=float, default=0.98)
    ap.add_argument("--top",    type=float, default=0.92)
    ap.add_argument("--bottom", type=float, default=0.12)
    ap.add_argument("--wspace", type=float, default=0.30)
    ap.add_argument("--hspace", type=float, default=0.30)
    # header paddings (in figure fraction, relative to top-row axes box)
    ap.add_argument("--subhdr_y_pad", type=float, default=0.012)  # space above axes for "CFD Benchmark/Pred"
    ap.add_argument("--model_y_pad",  type=float, default=0.030)  # extra space above sub-headers for "PINN/PIBERT"
    ap.add_argument("--row_x_pad",    type=float, default=0.018)  # gap to the left of first col for row labels
    # fonts
    ap.add_argument("--model_fs", type=int, default=26)
    ap.add_argument("--subhdr_fs", type=int, default=16)
    ap.add_argument("--row_fs", type=int, default=18)
    ap.add_argument("--badge_fs", type=int, default=12)
    # colorbar
    ap.add_argument("--cbar_frac",   type=float, default=0.38)
    ap.add_argument("--cbar_height", type=float, default=0.020)
    ap.add_argument("--cbar_bottom", type=float, default=0.055)
    # output
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--out", default="fig_benchmark_3x4.png")
    args=ap.parse_args()

    # indices
    idx_cyl  = args.idx_cyl  if args.idx_cyl  is not None else (args.idx if args.idx is not None else 0)
    idx_tube = args.idx_tube if args.idx_tube is not None else (args.idx if args.idx is not None else 0)
    idx_cav  = args.idx_cav  if args.idx_cav  is not None else (args.idx if args.idx is not None else 0)

    cases=[("Cylinder", args.cyl_root, idx_cyl),
           ("Tube",     args.tube_root, idx_tube),
           ("Cavity",   args.cav_root,  idx_cav)]
    left_m,right_m=args.models

    # load data
    panels=[]; all_vals=[]; field_label="field"
    for name,root,idx in cases:
        GTL,PRL,NMSL,field_label=load_panel(root,left_m,idx,args.field,args.seed)
        GTR,PRR,NMSR,_           =load_panel(root,right_m,idx,args.field,args.seed)
        panels.append((name,(GTL,PRL,NMSL),(GTR,PRR,NMSR)))
        all_vals += [GTL.ravel(), PRL.ravel(), GTR.ravel(), PRR.ravel()]
    vals=np.concatenate(all_vals)
    vmin,vmax=np.percentile(vals,[0.5,99.5]).astype(float)
    if args.field in ("u","v"): vmax=max(abs(vmin),abs(vmax)); vmin=-vmax

    # figure + grid
    fig=plt.figure(figsize=(12.5,9.6), dpi=args.dpi)
    gs=GridSpec(3,4, figure=fig, left=args.left, right=args.right, top=args.top,
                bottom=args.bottom, wspace=args.wspace, hspace=args.hspace)
    axs=np.array([[fig.add_subplot(gs[i,j]) for j in range(4)] for i in range(3)])

    ims=[]
    for i,(row_name,(GTL,PRL,NMSL),(GTR,PRR,NMSR)) in enumerate(panels):
        imgs=[GTL,PRL,GTR,PRR]
        for j,img in enumerate(imgs):
            ax=axs[i,j]
            im=ax.imshow(img, origin="lower", cmap=args.cmap, vmin=vmin, vmax=vmax, interpolation="bilinear")
            ims.append(im)
            # outer ticks only
            if args.outer_ticks_only:
                if i<2: ax.set_xticklabels([])
                if j>0: ax.set_yticklabels([])
            if i==2: ax.set_xlabel("x", fontsize=18)
            if j==0: ax.set_ylabel("y", fontsize=18)
            # NMSE badge on pred columns
            if args.show_nmse and j in (1,3):
                nm = NMSL if j==1 else NMSR
                ax.text(0.03, 0.97, f"NMSE = {nm:.3e}", transform=ax.transAxes,
                        ha="left", va="top", fontsize=args.badge_fs,
                        bbox=dict(fc="white", ec="0.6", alpha=0.9, boxstyle="round,pad=0.25"))

        # row label: right-aligned just left of first column box
        left_x = axs[i,0].get_position().x0 - args.row_x_pad
        y_mid  = (axs[i,0].get_position().y0 + axs[i,0].get_position().y1)/2
        fig.text(left_x, y_mid, row_name, ha="right", va="center",
                 fontsize=args.row_fs, weight="bold")

    # compute anchors from actual axes boxes
    top_y  = axs[0,0].get_position().y1
    sub_y  = top_y + args.subhdr_y_pad
    model_y= sub_y + args.model_y_pad

    def span_center(j0,j1):
        x0=axs[0,j0].get_position().x0; x1=axs[0,j1].get_position().x1
        return 0.5*(x0+x1)

    # sub-headers above each column
    fig.text(span_center(0,0), sub_y, "CFD Benchmark", ha="center", va="bottom",
             fontsize=args.subhdr_fs, weight="bold")
    fig.text(span_center(1,1), sub_y, "Pred",          ha="center", va="bottom",
             fontsize=args.subhdr_fs, weight="bold")
    fig.text(span_center(2,2), sub_y, "CFD Benchmark", ha="center", va="bottom",
             fontsize=args.subhdr_fs, weight="bold")
    fig.text(span_center(3,3), sub_y, "Pred",          ha="center", va="bottom",
             fontsize=args.subhdr_fs, weight="bold")

    # model headers centered over their 2-column spans
    fig.text(span_center(0,1), model_y, left_m,  ha="center", va="bottom",
             fontsize=args.model_fs, weight="bold")
    fig.text(span_center(2,3), model_y, right_m, ha="center", va="bottom",
             fontsize=args.model_fs, weight="bold")

    # centered bottom colorbar
    frac=args.cbar_frac; cbar_w=frac; cbar_x=0.57-cbar_w/2
    cax=fig.add_axes([cbar_x, args.cbar_bottom, cbar_w, args.cbar_height])
    cb=fig.colorbar(ims[0], cax=cax, orientation="horizontal")
    cb.set_label(field_label, fontsize=12)
    cb.ax.tick_params(labelsize=11, length=3, width=0.8)

    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print("Saved", args.out)

if __name__=="__main__":
    main()
