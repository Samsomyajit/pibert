# demo_make_dummy.py
import os, argparse, numpy as np

def make(out, n=64, h=64, w=64, cin=3, cout=3):
    os.makedirs(out, exist_ok=True)
    def synth(N):
        X = np.zeros((N, cin, h, w), np.float32)
        Y = np.zeros((N, cout, h, w), np.float32)
        yy, xx = np.meshgrid(np.linspace(0,1,h), np.linspace(0,1,w), indexing="ij")
        for i in range(N):
            for c in range(cin):
                X[i,c] = np.sin((c+1)*np.pi*xx) * np.cos((c+1)*np.pi*yy) * (1 + 0.1*np.random.randn())
            U = np.sin(2*np.pi*xx) * np.cos(3*np.pi*yy)
            V = np.cos(2*np.pi*xx) * np.sin(3*np.pi*yy)
            P = U*V + 0.05*np.random.randn(h,w)
            Y[i,0] = U; Y[i,1] = V; Y[i,2] = P
        return X,Y
    for split, N in [("train", n), ("val", max(8, n//4)), ("test", max(8, n//4))]:
        X,Y = synth(N)
        np.savez(os.path.join(out, f"{split}.npz"), input=X, target=Y)
    print(f"Dummy NPZ data written to {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--h", type=int, default=64)
    ap.add_argument("--w", type=int, default=64)
    ap.add_argument("--cin", type=int, default=3)
    ap.add_argument("--cout", type=int, default=3)
    args = ap.parse_args()
    make(args.out, args.n, args.h, args.w, args.cin, args.cout)
