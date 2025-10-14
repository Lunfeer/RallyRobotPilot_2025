# scripts/auto_pilot_rays.py
import os, json, math, time
import numpy as np
import torch
import torch.nn as nn

from PyQt6 import QtWidgets
from data_collector import DataCollectionUI

# --------- Modèle identique à l'entraînement ----------
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

# -------------- Utilitaires features ------------------
def angle_to_sin_cos(angle_raw: float):
    if angle_raw is None:
        return 0.0, 1.0
    ang = float(angle_raw)
    if abs(ang) > 2*math.pi:
        ang = math.radians(ang)
    return math.sin(ang), math.cos(ang)

def extract_rays_runtime(msg):
    # Ton format runtime: 'raycast_distances'
    if hasattr(msg, "raycast_distances") and msg.raycast_distances is not None:
        a = np.asarray(msg.raycast_distances, dtype=np.float32).flatten()
        return a if a.size > 0 else None
    return None

# -------------- Brain temps-réel -----------------------
class RaysNNMsgProcessor:
    def __init__(self, model_dir="models"):
        # ---- Charger meta + modèle
        meta_path = os.path.join(model_dir, "rays_mlp_meta.json")
        weight_path = os.path.join(model_dir, "rays_mlp.pt")
        if not (os.path.exists(meta_path) and os.path.exists(weight_path)):
            raise FileNotFoundError(
                f"Modèle introuvable. Entraîne d'abord: {meta_path} et {weight_path}"
            )
        with open(meta_path, "r") as f:
            meta = json.load(f)

        self.mean = np.asarray(meta["mean"], dtype=np.float32)
        self.std  = np.asarray(meta["std"], dtype=np.float32)
        self.input_dim = int(meta["input_dim"])
        self.rays_len  = int(meta["rays_len"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = RaysMLP(self.input_dim).to(self.device)
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.model.eval()

        # ---- États & seuils
        self.state = {"forward": False, "back": False, "left": False, "right": False}
        self.last_change_t = time.time()
        self.DEBUG = True
        self._frame = 0
        self.STEER_ON, self.STEER_OFF = 0.12, 0.06
        self.THR_ON,   self.THR_OFF   = 0.20, 0.10
        self.MIN_HOLD_S = 0.06

        # Sanity log
        if self.DEBUG:
            print(f"[AUTO] loaded model: input_dim={self.input_dim}, rays_len(train)={self.rays_len}")

    def desire_to_buttons(self, steer, throttle):
        desired = {"forward": False, "back": False, "left": False, "right": False}
        # throttle
        if throttle >= self.THR_ON:
            desired["forward"] = True
        elif throttle <= -self.THR_ON:
            desired["back"] = True
        else:
            if self.state["forward"] and throttle >= self.THR_OFF:
                desired["forward"] = True
            if self.state["back"] and throttle <= -self.THR_OFF:
                desired["back"] = True
        # steer
        if steer >= self.STEER_ON:
            desired["right"] = True
        elif steer <= -self.STEER_ON:
            desired["left"] = True
        else:
            if self.state["right"] and steer >= self.STEER_OFF:
                desired["right"] = True
            if self.state["left"] and steer <= -self.STEER_OFF:
                desired["left"] = True
        # contradictions
        if desired["forward"] and desired["back"]:
            desired["forward"] = throttle > 0
            desired["back"]    = throttle < 0
        if desired["left"] and desired["right"]:
            desired["left"]  = steer < 0
            desired["right"] = steer > 0
        return desired

    def nn_infer(self, message):
        rays = extract_rays_runtime(message)
        if rays is None:
            if self.DEBUG and self._frame % 10 == 0:
                print("[AUTO] no rays in message")
            return {"forward": False, "back": False, "left": False, "right": False}

        if rays.size != self.rays_len:
            if self.DEBUG and self._frame % 10 == 0:
                print(f"[AUTO] rays size mismatch: got {rays.size}, expected {self.rays_len} (retrain ou capteurs différents)")
            return {"forward": False, "back": False, "left": False, "right": False}

        s, c = angle_to_sin_cos(getattr(message, "car_angle", 0.0))
        x = np.concatenate([rays, np.array([s, c], dtype=np.float32)], dtype=np.float32)
        # normalisation identique au training
        x = (x - self.mean) / self.std

        xt = torch.from_numpy(x).unsqueeze(0).to(self.device)
        with torch.no_grad():
            steer, throttle = self.model(xt)[0].cpu().numpy().tolist()

        # petit coup de pouce au démarrage
        if self._frame < 20 and abs(throttle) < 0.12:
            throttle = 0.25

        desired = self.desire_to_buttons(steer, throttle)

        if self.DEBUG and self._frame % 5 == 0:
            print(f"[AUTO] rays={rays.size} steer={steer:.2f} thr={throttle:.2f} -> {desired}")

        self._frame += 1
        return desired

    def process_message(self, message, data_collector):
        desired = self.nn_infer(message)
        now = time.time()
        if now - self.last_change_t < self.MIN_HOLD_S:
            return
        changed = False
        for key in ["forward","back","left","right"]:
            cur, want = self.state[key], desired[key]
            if cur != want:
                data_collector.onCarControlled(key, want)
                self.state[key] = want
                changed = True
        if changed:
            self.last_change_t = now
            if self.DEBUG:
                print(f"[AUTO] sent: {self.state}")

# ---------------------- Entrée Qt ----------------------
if __name__ == "__main__":
    import sys
    def except_hook(cls, exception, traceback):
        sys.__excepthook__(cls, exception, traceback)
    sys.excepthook = except_hook

    app = QtWidgets.QApplication(sys.argv)

    brain = RaysNNMsgProcessor(model_dir="models")
    data_window = DataCollectionUI(brain.process_message)
    data_window.show()

    # commandes d'init (une par send)
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
