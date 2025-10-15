# scripts/train_model_v2.py
import glob, os, json, math, pickle, lzma, argparse, random
from dataclasses import dataclass
from typing import List, Tuple, Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# -------- utils --------
def to_controls_dict(ctrl_obj) -> Dict[str, bool]:
    if isinstance(ctrl_obj, (tuple, list)) and len(ctrl_obj) == 4:
        f, b, l, r = ctrl_obj
        def as_bool(v):
            try: return bool(int(v))
            except Exception: return bool(v)
        return {"forward": as_bool(f), "back": as_bool(b), "left": as_bool(l), "right": as_bool(r)}
    if isinstance(ctrl_obj, dict):
        return {k: bool(ctrl_obj.get(k, False)) for k in ["forward","back","left","right"]}
    return {k: bool(getattr(ctrl_obj, k, False)) for k in ["forward","back","left","right"]}

def compute_action_from_controls(ctrl: Dict[str,bool]) -> np.ndarray:
    throttle = (1.0 if ctrl["forward"] else 0.0) + (-1.0 if ctrl["back"] else 0.0)
    steer    = (-1.0 if ctrl["left"]   else 0.0) + ( 1.0 if ctrl["right"] else 0.0)
    return np.array([steer, throttle], dtype=np.float32)   # (can be 0,0 = "do nothing")

def angle_to_sin_cos(angle_raw: float) -> Tuple[float, float]:
    if angle_raw is None:
        return 0.0, 1.0
    ang = float(angle_raw)
    if abs(ang) > 2*math.pi:  # prob. degrees
        ang = math.radians(ang)
    return math.sin(ang), math.cos(ang)

def extract_rays(msg) -> Optional[np.ndarray]:
    if hasattr(msg, "raycast_distances"):
        arr = getattr(msg, "raycast_distances")
        if arr is not None:
            a = np.asarray(arr, dtype=np.float32).flatten()
            if a.size > 0:
                return a
    return None

def load_messages(path: str) -> List[Any]:
    with lzma.open(path, "rb") as f:
        data = pickle.load(f)
    return data

# -------- dataset --------
@dataclass
class Sample:
    x: np.ndarray
    y: np.ndarray

class RaysDataset(Dataset):
    def __init__(self, files: List[str], delta_s: float = 0.15, hz: float = 10.0, use_speed: bool = True):
        self.samples: List[Sample] = []
        rays_len_ref = None
        self.hz = hz
        k = max(1, int(round(delta_s * hz)))  # index shift for t+Δ

        for path in files:
            msgs = load_messages(path)
            if not msgs:
                continue

            # Pas de sous-échantillonnage ici: tes records sont déjà à ~10 Hz
            # Si besoin: msgs = msgs[::step]

            # Déterminer len des rays à partir du 1er msg valide
            if rays_len_ref is None:
                for m in msgs:
                    r = extract_rays(m)
                    if r is not None and r.size > 0:
                        rays_len_ref = r.size
                        break
            if rays_len_ref is None:
                print(f"[!] {path}: aucun raycast_distances valide — ignoré")
                continue

            # Paires (obs_t -> action_{t+k})
            for i in range(len(msgs) - k - 1):
                m_obs = msgs[i]
                m_act = msgs[i + k]

                r = extract_rays(m_obs)
                if r is None or r.size != rays_len_ref:
                    continue

                s, c = angle_to_sin_cos(getattr(m_obs, "car_angle", 0.0))
                if use_speed:
                    spd = float(getattr(m_obs, "car_speed", 0.0))
                    x = np.concatenate([r, np.array([s, c, spd], dtype=np.float32)], dtype=np.float32)
                else:
                    x = np.concatenate([r, np.array([s, c], dtype=np.float32)], dtype=np.float32)

                ctrl = to_controls_dict(getattr(m_act, "current_controls", (0,0,0,0)))
                y = compute_action_from_controls(ctrl)
                self.samples.append(Sample(x=x, y=y))

        if not self.samples:
            raise RuntimeError("Dataset vide: pas de rays/controls trouvés.")

        X = np.stack([s.x for s in self.samples])
        self.mean = X.mean(axis=0).astype(np.float32)
        self.std  = X.std(axis=0).astype(np.float32)
        self.std[self.std < 1e-6] = 1.0

        self.input_dim = X.shape[1]
        self.rays_len  = rays_len_ref
        self.use_speed = use_speed
        self.delta_s   = delta_s

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        x = (s.x - self.mean) / self.std
        return torch.from_numpy(x), torch.from_numpy(s.y)

# -------- modèle --------
class RaysMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 128),       nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64),        nn.ReLU(),
            nn.Linear(64, 2),
            nn.Tanh(),
        )
    def forward(self, x): return self.net(x)

# -------- train / eval --------
@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()
    n, loss_sum, mae_steer, mae_thr = 0, 0.0, 0.0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss_sum += loss_fn(pred, yb).item() * xb.size(0)
        mae = (pred - yb).abs().mean(dim=0)  # [mae_steer, mae_thr]
        mae_steer += mae[0].item() * xb.size(0)
        mae_thr   += mae[1].item() * xb.size(0)
        n += xb.size(0)
    return {
        "loss": loss_sum / max(1, n),
        "mae_steer": mae_steer / max(1, n),
        "mae_thr":   mae_thr   / max(1, n),
        "n": n
    }

def train(args):
    # fichiers & split par fichiers
    all_files = sorted(glob.glob(args.records))
    if not all_files:
        raise FileNotFoundError(f"Aucun fichier pour {args.records}")

    random.Random(args.seed).shuffle(all_files)
    n_total = len(all_files)
    n_test = max(1, int(round(n_total * args.test_split)))
    test_files = all_files[:n_test]
    trainval_files = all_files[n_test:]

    if not trainval_files:
        raise RuntimeError("Pas assez de fichiers pour train/val après extraction du test.")

    # dataset train+val et test (features identiques)
    ds_trainval = RaysDataset(trainval_files, delta_s=args.delta, hz=10.0, use_speed=not args.no_speed)
    ds_test     = RaysDataset(test_files,     delta_s=args.delta, hz=10.0, use_speed=not args.no_speed)

    # split interne train/val (par échantillons, simple et efficace)
    n = len(ds_trainval)
    n_val = max(100, int(round(n * args.val_split)))
    n_train = n - n_val
    train_ds, val_ds = random_split(ds_trainval, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    # loaders
    bs = min(args.bs, max(1, len(train_ds)))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=min(bs, len(val_ds)), shuffle=False, drop_last=False)
    test_loader  = DataLoader(ds_test,  batch_size=min(bs, len(ds_test)), shuffle=False, drop_last=False)

    print(f"[FILES] total={n_total}  test_files={len(test_files)}  trainval_files={len(trainval_files)}")
    print(f"[SAMPLES] train={len(train_ds)}  val={len(val_ds)}  test={len(ds_test)}  bs={bs}")
    print(f"[INPUT] dim={ds_trainval.input_dim}  rays_len={ds_trainval.rays_len}  use_speed={ds_trainval.use_speed}")

    # modèle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RaysMLP(ds_trainval.input_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()

    best_val = float("inf")
    os.makedirs(args.out, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            # petit lissage sur steer (optionnel)
            if pred.shape[0] > 1:
                loss = loss + 0.001 * (pred[:,0].diff().abs().mean())
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * xb.size(0)
        train_loss = tot / len(train_ds)

        val_metrics = evaluate(model, val_loader, device, loss_fn)
        print(f"[Epoch {epoch:02d}] train={train_loss:.4f}  val={val_metrics['loss']:.4f} "
              f"(mae_s={val_metrics['mae_steer']:.3f}, mae_t={val_metrics['mae_thr']:.3f})")

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            # save best
            torch.save(model.state_dict(), os.path.join(args.out, "rays_mlp.pt"))
            meta = {
                "input_dim": ds_trainval.input_dim,
                "rays_len": ds_trainval.rays_len,
                "mean": ds_trainval.mean.tolist(),
                "std":  ds_trainval.std.tolist(),
                "delta_s": args.delta,
                "hz": 10.0,
                "use_speed": ds_trainval.use_speed,
                "version": 2,
            }
            with open(os.path.join(args.out, "rays_mlp_meta.json"), "w") as f:
                json.dump(meta, f)
            print(f"  ↳ saved best to {args.out}/rays_mlp.pt (val={best_val:.4f})")

    # évaluation finale sur train/val/test
    train_metrics = evaluate(model, DataLoader(train_ds, batch_size=bs, shuffle=False), device, loss_fn)
    val_metrics   = evaluate(model, val_loader, device, loss_fn)
    test_metrics  = evaluate(model, test_loader, device, loss_fn)
    print("\n=== Final metrics ===")
    print(f"Train: loss={train_metrics['loss']:.4f}  mae_s={train_metrics['mae_steer']:.3f}  mae_t={train_metrics['mae_thr']:.3f}  n={train_metrics['n']}")
    print(f"Val  : loss={val_metrics['loss']:.4f}    mae_s={val_metrics['mae_steer']:.3f}    mae_t={val_metrics['mae_thr']:.3f}    n={val_metrics['n']}")
    print(f"Test : loss={test_metrics['loss']:.4f}   mae_s={test_metrics['mae_steer']:.3f}   mae_t={test_metrics['mae_thr']:.3f}   n={test_metrics['n']}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--records", default="record_*.npz", help="pattern des fichiers")
    p.add_argument("--out", default="models", help="dossier de sortie")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--delta", type=float, default=0.15, help="décalage (s) actions vs obs")
    p.add_argument("--val_split", type=float, default=0.10, help="part de val sur trainval (0..1)")
    p.add_argument("--test_split", type=float, default=0.10, help="part des fichiers en test (0..1)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_speed", action="store_true", help="désactiver car_speed dans les features")
    args = p.parse_args()
    train(args)
