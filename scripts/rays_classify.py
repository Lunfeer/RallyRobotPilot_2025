import os, glob, json, pickle, lzma, argparse, random, time
from dataclasses import dataclass
from typing import List, Any, Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

CLASS_NAMES = [
    "NOP",          # (0,0,0,0): 0,
    "FWD",          # (1,0,0,0): 1,
    "BACK",         # (0,1,0,0): 2,
    "LEFT",         # (0,0,1,0): 3,
    "RIGHT",        # (0,0,0,1): 4,
    "FWD_LEFT",     # (1,0,1,0): 5,
    "FWD_RIGHT",    # (1,0,0,1): 6,
    "BACK_LEFT",    # (0,1,1,0): 7,
    "BACK_RIGHT",   # (0,1,0,1): 8,
]
BITS_TO_CLASS = {
    (0,0,0,0): 0,
    (1,0,0,0): 1,
    (0,1,0,0): 2,
    (0,0,1,0): 3,
    (0,0,0,1): 4,
    (1,0,1,0): 5,
    (1,0,0,1): 6,
    (0,1,1,0): 7,
    (0,1,0,1): 8,
}
CLASS_TO_BITS = {v:k for k,v in BITS_TO_CLASS.items()}

def to_bool(v) -> bool:
    try:
        return bool(int(v))
    except Exception:
        return bool(v)

def controls_to_bits(ctrl_obj):
    f=b=l=r=False
    if isinstance(ctrl_obj, (tuple, list)) and len(ctrl_obj) == 4:
        f,b,l,r = map(to_bool, ctrl_obj)
    elif isinstance(ctrl_obj, dict):
        f = to_bool(ctrl_obj.get("forward", False))
        b = to_bool(ctrl_obj.get("back",    False))
        l = to_bool(ctrl_obj.get("left",    False))
        r = to_bool(ctrl_obj.get("right",   False))
    else:
        f = to_bool(getattr(ctrl_obj, "forward", False))
        b = to_bool(getattr(ctrl_obj, "back",    False))
        l = to_bool(getattr(ctrl_obj, "left",    False))
        r = to_bool(getattr(ctrl_obj, "right",   False))
    if f and b:
        return None
    return (int(f), int(b), int(l), int(r))

def bits_to_class(bits) -> Optional[int]:
    return BITS_TO_CLASS.get(bits, None)

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

@dataclass
class ClsSample:
    x: np.ndarray
    y: int

class RaysClsDataset(Dataset):
    def __init__(self, files: List[str], delta_s: float = 0.2):
        self.samples: List[ClsSample] = []
        rays_len_ref = None
        self.hz = 10.0
        k = max(1, int(round(delta_s * self.hz)))

        for path in files:
            msgs = load_messages(path)

            if rays_len_ref is None:
                for m in msgs:
                    r = extract_rays(m)
                    if r is not None and r.size > 0:
                        rays_len_ref = r.size
                        break

            for i in range(len(msgs) - k - 1):
                m_obs = msgs[i]
                m_act = msgs[i + k]

                r = extract_rays(m_obs)
                spd = float(getattr(m_obs, "car_speed", 0.0))
                x = np.concatenate([r, np.array([spd], dtype=np.float32)], dtype=np.float32)

                bits = controls_to_bits(getattr(m_act, "current_controls", (0,0,0,0)))
                cls = bits_to_class(bits)

                self.samples.append(ClsSample(x=x, y=cls))

        X = np.stack([s.x for s in self.samples])
        self.mean = X.mean(axis=0).astype(np.float32)
        self.std  = X.std(axis=0).astype(np.float32)
        self.std[self.std < 1e-6] = 1.0

        self.input_dim = X.shape[1]
        self.rays_len  = rays_len_ref
        self.delta_s   = delta_s

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        x = (s.x - self.mean) / self.std
        return torch.from_numpy(x), torch.tensor(s.y, dtype=torch.long)

class ActionMLP(nn.Module):
    def __init__(self, input_dim: int, n_classes: int = len(CLASS_NAMES)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),       nn.ReLU(),
            nn.Linear(128, n_classes)
        )
    def forward(self, x): 
        return self.net(x)

@torch.no_grad()
def eval_acc_loss(model, loader, device):
    model.eval()
    n, loss_sum, correct = 0, 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb, reduction="sum")
        loss_sum += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        n += xb.size(0)
    return (loss_sum / max(1, n)), (correct / max(1, n))

def train_mode(args):
    records_pattern = "record_*.npz"
    out_dir = "models"
    os.makedirs(out_dir, exist_ok=True)

    all_files = sorted(glob.glob(records_pattern))

    ds = RaysClsDataset(all_files, delta_s=args.delta)

    n = len(ds)
    n_val = max(100, int(round(0.10 * n)))
    n_train = n - n_val
    g = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=g)

    # dataloaders
    bs = args.bs
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=min(bs, len(val_ds)), shuffle=False, drop_last=False)
    print(f"[DATA] total={len(ds)}  train={len(train_ds)}  val={len(val_ds)}  bs={bs}")
    print(f"[INPUT] dim={ds.input_dim}  rays_len={ds.rays_len}  delta={ds.delta_s}s  (10 Hz -> k≈{int(round(ds.delta_s*10))})")

    # modèle + optim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActionMLP(ds.input_dim, n_classes=len(CLASS_NAMES)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4) 

    # training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, n_seen = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item() * xb.size(0)
            n_seen += xb.size(0)
        train_loss = loss_sum / max(1, n_seen)

        val_loss, val_acc = eval_acc_loss(model, val_loader, device)
        print(f"[Epoch {epoch:02d}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")

    torch.save(model.state_dict(), os.path.join(out_dir, "rays_cls.pt"))
    meta = {
        "input_dim": ds.input_dim,
        "rays_len": ds.rays_len,
        "mean": ds.mean.tolist(),
        "std":  ds.std.tolist(),
        "delta_s": ds.delta_s,
        "class_names": CLASS_NAMES,
        "version": 0,
    }
    with open(os.path.join(out_dir, "rays_cls_meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"[SAVE] modèle + méta dans '{out_dir}/'")

def extract_rays_runtime(msg):
    if hasattr(msg, "raycast_distances") and msg.raycast_distances is not None:
        a = np.asarray(msg.raycast_distances, dtype=np.float32).flatten()
        return a if a.size > 0 else None
    return None

class ClassifierAutopilot:
    def __init__(self, model_dir="models", debug=True):
        meta_path = os.path.join(model_dir, "rays_cls_meta.json")
        weight_path = os.path.join(model_dir, "rays_cls.pt")
        if not (os.path.exists(meta_path) and os.path.exists(weight_path)):
            raise FileNotFoundError(f"Modèle introuvable: {meta_path} / {weight_path}")
        with open(meta_path, "r") as f:
            meta = json.load(f)

        self.mean = np.asarray(meta["mean"], dtype=np.float32)
        self.std  = np.asarray(meta["std"], dtype=np.float32)
        self.input_dim = int(meta["input_dim"])
        self.rays_len  = int(meta["rays_len"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ActionMLP(self.input_dim, n_classes=len(CLASS_NAMES)).to(self.device)
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.model.eval()

        self.state = {"forward": False, "back": False, "left": False, "right": False}
        self.DEBUG = debug
        self._frame = 0

        if self.DEBUG:
            print(f"[AUTO] loaded: input_dim={self.input_dim}, rays_len={self.rays_len}")

    def _class_to_buttons(self, cls_id: int):
        f,b,l,r = CLASS_TO_BITS.get(int(cls_id), (0,0,0,0))
        return {"forward": bool(f), "back": bool(b), "left": bool(l), "right": bool(r)}

    def nn_infer(self, message):
        rays = extract_rays_runtime(message)
        if rays is None or rays.size != self.rays_len:
            return {"forward": False, "back": False, "left": False, "right": False}

        spd  = float(getattr(message, "car_speed", 0.0))
        x = np.concatenate([rays, np.array([spd], dtype=np.float32)], dtype=np.float32)
        x = (x - self.mean) / self.std

        xt = torch.from_numpy(x).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(xt)
            cls = int(torch.argmax(logits, dim=1).item())

        desired = self._class_to_buttons(cls)
        if self.DEBUG and self._frame % 10 == 0:
            print(f"[AUTO] pred={CLASS_NAMES[cls]} -> {desired}")
        self._frame += 1
        return desired

    def process_message(self, message, data_collector):
        desired = self.nn_infer(message)
        for key in ["forward","back","left","right"]:
            cur, want = self.state[key], desired[key]
            if cur != want:
                data_collector.onCarControlled(key, want)
                self.state[key] = want


def autopilot_mode():
    from PyQt6 import QtWidgets
    from data_collector import DataCollectionUI

    app = QtWidgets.QApplication([])

    brain = ClassifierAutopilot(model_dir="models", debug=True)
    data_window = DataCollectionUI(brain.process_message)
    data_window.show()

    ni = data_window.network_interface
    def send(cmd: str):
        cmd = cmd.strip()
        if not cmd.endswith(";"):
            cmd += ";"
        ni.send_cmd(cmd)

    send("set ray visible")
    send("release all")
    send("reset")

    app.exec()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "autopilot"], default="train")
    p.add_argument("--delta", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--bs", type=int, default=256)
    args = p.parse_args()

    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    if args.mode == "train":
        train_mode(args)
    else:
        autopilot_mode()


if __name__ == "__main__":
    main()
