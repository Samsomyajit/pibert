# data.py
import os, numpy as np, torch
from torch.utils.data import Dataset, DataLoader

def _load_npz_arrays(path):
    with np.load(path, allow_pickle=True) as data:
        X = np.array(data["input"], dtype=np.float32)
        Y = np.array(data["target"], dtype=np.float32)
    return X, Y  # (N,C,H,W)

def _compute_stats(X, Y):
    x_mean = X.mean(axis=(0,2,3), keepdims=True)
    x_std  = X.std(axis=(0,2,3), keepdims=True) + 1e-8
    y_mean = Y.mean(axis=(0,2,3), keepdims=True)
    y_std  = Y.std(axis=(0,2,3), keepdims=True) + 1e-8
    return {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}

class CFDBenchFolder(Dataset):
    def __init__(self, npz_path, stats=None, normalize=True):
        self.X, self.Y = _load_npz_arrays(npz_path)  # (N,C,H,W)
        self.N, self.Cin, self.H, self.W = self.X.shape
        _, self.Cout, _, _ = self.Y.shape

        if normalize and stats is not None:
            self.x_mean, self.x_std = stats["x_mean"], stats["x_std"]
            self.y_mean, self.y_std = stats["y_mean"], stats["y_std"]
        else:
            # fall back to self-computed (only used if stats=None, e.g., quick sanity runs)
            s = _compute_stats(self.X, self.Y)
            self.x_mean, self.x_std = s["x_mean"], s["x_std"]
            self.y_mean, self.y_std = s["y_mean"], s["y_std"]

        self.Xn = (self.X - self.x_mean) / self.x_std if normalize else self.X
        self.Yn = (self.Y - self.y_mean) / self.y_std if normalize else self.Y

        # precompute grid (x,y,t)
        yy, xx = np.meshgrid(np.linspace(0,1,self.H), np.linspace(0,1,self.W), indexing="ij")
        self.xy = np.stack([xx, yy], axis=-1).astype(np.float32)  # (H,W,2)
        self.t = 0.0

    def __len__(self): return self.N

    def __getitem__(self, i):
        x = torch.from_numpy(self.Xn[i])                 # (Cin,H,W)
        y = torch.from_numpy(self.Yn[i])                 # (Cout,H,W)
        t = np.full((self.H, self.W, 1), self.t, np.float32)
        xyt = np.concatenate([self.xy, t], axis=-1)      # (H,W,3)
        xyt = torch.from_numpy(xyt).float()
        return x, y, xyt

def make_loaders(root, fmt="npz", batch_size=4, workers=None, normalize=True):
    assert fmt == "npz", "only npz is supported here"
    tr_path = os.path.join(root, "train.npz")
    va_path = os.path.join(root, "val.npz")
    te_path = os.path.join(root, "test.npz")

    # load train once to compute stats
    Xtr, Ytr = _load_npz_arrays(tr_path)
    stats = _compute_stats(Xtr, Ytr)

    tr = CFDBenchFolder(tr_path, stats=stats, normalize=normalize)
    va = CFDBenchFolder(va_path, stats=stats, normalize=normalize)
    te = CFDBenchFolder(te_path, stats=stats, normalize=normalize)

    use_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if workers is None:
        workers = 0 if use_mps else 2
    pin_memory = False if use_mps else True

    train_loader = DataLoader(tr, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=pin_memory)
    val_loader   = DataLoader(va, batch_size=batch_size, shuffle=False,
                              num_workers=workers, pin_memory=pin_memory)
    test_loader  = DataLoader(te, batch_size=batch_size, shuffle=False,
                              num_workers=workers, pin_memory=pin_memory)

    # return train stats so eval can unnormalize with the same
    shapes = {"H": tr.H, "W": tr.W, "Cin": tr.Cin, "Cout": tr.Cout}
    return train_loader, val_loader, test_loader, stats, shapes
