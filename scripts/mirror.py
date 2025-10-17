# mirror_any.py
import sys, os, copy, pickle, lzma
import numpy as np

# Hypothèse labels: [gaz, frein, gauche, droite]
LEFT_IDX, RIGHT_IDX = 2, 3

def try_load_lzma_pickle(path):
    """Retourne (data, 'lzma_pickle') si succès, sinon (None, None)."""
    try:
        with lzma.open(path, "rb") as f:
            data = pickle.load(f)
        return data, "lzma_pickle"
    except Exception:
        return None, None

def try_load_npz(path):
    """Retourne (npz, 'npz_xy'|'npz_separe') si succès, sinon (None, None)."""
    try:
        npz = np.load(path, allow_pickle=True)
        keys = set(npz.files)
        if {"X","y"}.issubset(keys):
            return npz, "npz_xy"
        if {"speed","raycast_distances","current_controls"}.issubset(keys):
            return npz, "npz_separe"
        # format NPZ inconnu
        return None, None
    except Exception:
        return None, None

def mirror_sample_obj(e):
    """Copie un objet et inverse ses distances + échange gauche/droite."""
    m = copy.deepcopy(e)
    # distances
    d = list(getattr(e, "raycast_distances"))
    if len(d) != 15:
        raise ValueError(f"raycast_distances doit avoir 15 valeurs, trouvé {len(d)}")
    m.raycast_distances = tuple(d[::-1])
    # contrôles
    c = list(getattr(e, "current_controls"))
    if len(c) != 4:
        raise ValueError(f"current_controls doit avoir 4 valeurs, trouvé {len(c)}")
    c[LEFT_IDX], c[RIGHT_IDX] = c[RIGHT_IDX], c[LEFT_IDX]
    m.current_controls = tuple(c)
    # vitesse inchangée
    return m

def mirror_npz_xy(X, y):
    """X:[N,16] (0=vitesse, 1..15=dists), y:[N,4] -> retourne miroirs."""
    if X.ndim != 2 or X.shape[1] < 16:
        raise ValueError(f"X doit avoir au moins 16 colonnes, trouvé {X.shape}")
    if y.ndim != 2 or y.shape[1] != 4:
        raise ValueError(f"y doit avoir shape [N,4], trouvé {y.shape}")
    X_m = X.copy()
    dist_slice = slice(1,16)
    X_m[:, dist_slice] = X[:, dist_slice][:, ::-1]
    y_m = y.copy()
    y_m[:, [LEFT_IDX, RIGHT_IDX]] = y[:, [RIGHT_IDX, LEFT_IDX]]
    return X_m, y_m

def mirror_npz_separe(speed, dists, ctrls):
    """speed:[N], dists:[N,15], ctrls:[N,4] -> retourne miroirs."""
    if dists.ndim != 2 or dists.shape[1] != 15:
        raise ValueError(f"dists doit être [N,15], trouvé {dists.shape}")
    if ctrls.ndim != 2 or ctrls.shape[1] != 4:
        raise ValueError(f"ctrls doit être [N,4], trouvé {ctrls.shape}")
    speed_m = speed.copy()
    dists_m = dists[:, ::-1].copy()
    ctrls_m = ctrls.copy()
    ctrls_m[:, [LEFT_IDX, RIGHT_IDX]] = ctrls[:, [RIGHT_IDX, LEFT_IDX]]
    return speed_m, dists_m, ctrls_m

def main():
    if len(sys.argv) != 3:
        print("Usage: python mirror_any.py <input.npz> <output.npz>")
        sys.exit(1)
    inp, outp = sys.argv[1], sys.argv[2]
    if not os.path.isfile(inp):
        print(f"Fichier introuvable: {inp}")
        sys.exit(1)

    # 1) Essayer LZMA+pickle d'abord (cas de tes record_*.npz)
    data, fmt = try_load_lzma_pickle(inp)
    if fmt == "lzma_pickle":
        if not isinstance(data, (list, tuple)):
            raise SystemExit("Pickle LZMA inattendu: on attend une liste d'objets échantillons.")
        mirrored = [mirror_sample_obj(e) for e in data]
        with lzma.open(outp, "wb") as f:
            pickle.dump(mirrored, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"✅ Écrit (LZMA+pickle) → {outp} | N={len(mirrored)}")
        return

    # 2) Sinon, vrai NPZ
    npz, fmt = try_load_npz(inp)
    if fmt == "npz_xy":
        X_m, y_m = mirror_npz_xy(npz["X"], npz["y"])
        np.savez_compressed(outp, X=X_m, y=y_m)
        print(f"✅ Écrit (NPZ X/y) → {outp} | X={X_m.shape} y={y_m.shape}")
        return
    if fmt == "npz_separe":
        s_m, d_m, c_m = mirror_npz_separe(npz["speed"], npz["raycast_distances"], npz["current_controls"])
        np.savez_compressed(outp, speed=s_m, raycast_distances=d_m, current_controls=c_m)
        print(f"✅ Écrit (NPZ séparé) → {outp} | N={s_m.shape[0]}")
        return

    # 3) Rien n'a marché
    print("❌ Format d'entrée non reconnu.")
    print(" - Si c'est un pickle LZMA, le fichier devrait se charger via lzma+pickle.")
    print(" - Si c'est un vrai NPZ, il doit contenir soit X/y, soit speed/raycast_distances/current_controls.")
    sys.exit(1)

if __name__ == "__main__":
    main()
