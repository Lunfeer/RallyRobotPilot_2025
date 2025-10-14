import glob, os, json, math, pickle, lzma, argparse, random
from dataclasses import dataclass
from typing import List, Tuple, Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# ----------------- Utils d'extraction -----------------
def get_attr(obj, *names, default=None):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default

def to_controls_dict(ctrl_obj) -> Dict[str, bool]:
    # Cas tuple / liste: (forward, back, left, right)
    if isinstance(ctrl_obj, (tuple, list)) and len(ctrl_obj) == 4:
        f, b, l, r = ctrl_obj
        def as_bool(v):
            try:
                return bool(int(v))
            except Exception:
                return bool(v)
        return {
            "forward": as_bool(f),
            "back":    as_bool(b),
            "left":    as_bool(l),
            "right":   as_bool(r),
        }

    # Cas dict déjà formé
    if isinstance(ctrl_obj, dict):
        return {
            "forward": bool(ctrl_obj.get("forward", False)),
            "back":    bool(ctrl_obj.get("back", False)),
            "left":    bool(ctrl_obj.get("left", False)),
            "right":   bool(ctrl_obj.get("right", False)),
        }

    # Cas objet avec attributs (fallback)
    names = ["forward","back","left","right"]
    return {k: bool(getattr(ctrl_obj, k, False)) for k in names}


def compute_action_from_controls(ctrl: Dict[str,bool]) -> np.ndarray:
    throttle = (1.0 if ctrl["forward"] else 0.0) + (-1.0 if ctrl["back"] else 0.0)
    steer = (-1.0 if ctrl["left"] else 0.0) + (1.0 if ctrl["right"] else 0.0)
    return np.array([steer, throttle], dtype=np.float32)

def angle_to_sin_cos(angle_raw: float) -> Tuple[float, float]:
    if angle_raw is None:
        return 0.0, 1.0
    ang = float(angle_raw)
    # heuristique: si on voit des degrés, convertir
    if abs(ang) > 2*math.pi:
        ang = math.radians(ang)
    return math.sin(ang), math.cos(ang)

def extract_rays(msg) -> np.ndarray:
    # Ton format: raycast_distances
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

# ----------------- Dataset -----------------
@dataclass
class Sample:
    x: np.ndarray
    y: np.ndarray

class RaysDataset(Dataset):
    def __init__(self, file_patterns: str, delta_s: float = 0.15, target_hz: float = 10.0):
        files = sorted(glob.glob(file_patterns))
        if not files:
            raise FileNotFoundError(f"Aucun fichier trouvé avec le pattern: {file_patterns}")

        self.samples: List[Sample] = []
        rays_len_ref = None

        for path in files:
            msgs = load_messages(path)
            if not msgs: 
                continue

            # Sous-échantillonnage simple (si ~40Hz → 10Hz)
            step = 1
            approx_hz = 40.0  # valeur par défaut raisonnable
            step = max(1, int(round(approx_hz / target_hz)))
            msgs = msgs[::step]

            # Calcul du décalage delta en indices
            k = max(1, int(round(delta_s * target_hz)))

            # Déterminer taille rays à partir du premier msg valide
            if rays_len_ref is None:
                for m in msgs:
                    r = extract_rays(m)
                    if r is not None and r.size > 0:
                        rays_len_ref = r.size
                        break

            if rays_len_ref is None:
                print(f"[!] {path}: aucun 'rays' détecté — ignoré")
                continue

            # Construire les paires (obs_t -> action_{t+k})
            for i in range(len(msgs) - k - 1):
                m_obs = msgs[i]
                m_act = msgs[i + k]

                r = extract_rays(m_obs)
                if r is None or r.size != rays_len_ref:
                    continue

                s, c = angle_to_sin_cos(get_attr(m_obs, "car_angle", default=0.0))
                x = np.concatenate([r, np.array([s, c], dtype=np.float32)], dtype=np.float32)

                ctrl = to_controls_dict(get_attr(m_act, "current_controls", default={}))
                y = compute_action_from_controls(ctrl)

                self.samples.append(Sample(x=x, y=y))

        if not self.samples:
            raise RuntimeError("Dataset vide: vérifie que les messages contiennent 'rays' et 'current_controls'.")

        # Empiler pour calculer normalisation
        X = np.stack([s.x for s in self.samples])
        self.mean = X.mean(axis=0).astype(np.float32)
        self.std = X.std(axis=0).astype(np.float32)
        self.std[self.std < 1e-6] = 1.0

        self.input_dim = X.shape[1]
        self.rays_len = rays_len_ref

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        x = (s.x - self.mean) / self.std
        return torch.from_numpy(x), torch.from_numpy(s.y)

# ----------------- Modèle -----------------
class RaysMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Tanh(),  # steer, throttle in [-1,1]
        )

    def forward(self, x):
        return self.net(x)

# ----------------- Entraînement -----------------
def train(args):
    ds = RaysDataset(args.records, delta_s=args.delta, target_hz=args.hz)
    n = len(ds)
    n_val = max(100, int(0.1 * n))
    n_train = n - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    bs = min(args.bs, max(1, len(train_ds)))   
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=min(bs, len(val_ds)), shuffle=False, drop_last=False)
    print(f"[INFO] train={len(train_ds)}  val={len(val_ds)}  bs={bs}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RaysMLP(ds.input_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()  # Huber

    best_val = float("inf")
    os.makedirs(args.out, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            # régularisation de lissage (facultative)
            loss = loss + 0.001 * (pred[:,0].diff().abs().mean() if pred.shape[0] > 1 else 0.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * xb.size(0)
        train_loss = tot / len(train_ds)

        # Validation
        model.eval()
        vtot = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                vtot += loss_fn(pred, yb).item() * xb.size(0)
        val_loss = vtot / len(val_ds)

        print(f"[Epoch {epoch:02d}] train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(args.out, "rays_mlp.pt"))
            meta = {
                "input_dim": ds.input_dim,
                "rays_len": ds.rays_len,
                "mean": ds.mean.tolist(),
                "std": ds.std.tolist(),
                "delta_s": args.delta,
                "target_hz": args.hz,
                "version": 1,
            }
            with open(os.path.join(args.out, "rays_mlp_meta.json"), "w") as f:
                json.dump(meta, f)
            print(f"  ↳ saved best to {args.out}/rays_mlp.pt (val={best_val:.4f})")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--records", default="record_*.npz", help="pattern des fichiers")
    p.add_argument("--out", default="models", help="dossier de sortie")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--delta", type=float, default=0.15, help="décalage (s) actions vs obs")
    p.add_argument("--hz", type=float, default=10.0, help="Hz utilisés après sous-échantillonnage")
    args = p.parse_args()
    train(args)
