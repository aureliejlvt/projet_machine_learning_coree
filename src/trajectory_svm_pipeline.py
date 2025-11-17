import argparse
import re
from pathlib import Path
from typing import Iterator, List, Tuple, Dict

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D  # utile selon versions de matplotlib


# ---- AUGMENTATION SOBRE (pas de rotation, pas de time-warp) ----
from typing import Optional
from sklearn.model_selection import GroupKFold

_rng = np.random.RandomState(42)

def jitter_xyz(xyz: np.ndarray, sigma_mm: float = 1.5,
               rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """Bruit gaussien i.i.d. (zéro-mean) sur chaque point, n'altère pas la longueur."""
    rng = rng or _rng
    return xyz + rng.normal(0.0, sigma_mm, size=xyz.shape)

def scale_xyz(xyz: np.ndarray, scale_std: float = 0.03,
              per_axis: bool = False,
              rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """
    Mise à l'échelle douce autour de 1.0. Par défaut isotrope.
    per_axis=True permet une petite anisotropie (X/Y/Z).
    """
    rng = rng or _rng
    if per_axis:
        s = rng.normal(1.0, scale_std, size=3)
        s = np.clip(s, 0.9, 1.1)
        return xyz * s
    else:
        s = float(np.clip(rng.normal(1.0, scale_std), 0.9, 1.1))
        return xyz * s

def augment_one_simple(xyz_resampled: np.ndarray,
                       jitter_sigma_mm: float = 1.5,
                       scale_std: float = 0.03,
                       per_axis_scale: bool = False,
                       rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """
    Chaîne d'augmentation minimaliste : (optionnel) scale -> (optionnel) jitter.
    - Pas de rotation, pas de time-warp, pas de miroir par défaut.
    - Conserve la longueur et l'ordre temporel.
    """
    rng = rng or _rng
    P = xyz_resampled.astype(float, copy=True)
    if scale_std and scale_std > 0:
        P = scale_xyz(P, scale_std=scale_std, per_axis=per_axis_scale, rng=rng)
    if jitter_sigma_mm and jitter_sigma_mm > 0:
        P = jitter_xyz(P, sigma_mm=jitter_sigma_mm, rng=rng)
    return P

def make_augmented_batch_simple(xyz_resampled: np.ndarray, n_aug: int,
                                jitter_sigma_mm: float = 1.5,
                                scale_std: float = 0.03,
                                per_axis_scale: bool = False,
                                rng_seed: int = 42) -> List[np.ndarray]:
    rng = np.random.RandomState(rng_seed)
    return [
        augment_one_simple(
            xyz_resampled,
            jitter_sigma_mm=jitter_sigma_mm,
            scale_std=scale_std,
            per_axis_scale=per_axis_scale,
            rng=rng
        )
        for _ in range(n_aug)
    ]

def load_dataset(data_dir: Path, resample_len: int = 100,
                 augment: bool = False, aug_per_sample: int = 0,
                 jitter_sigma_mm: float = 1.5,
                 scale_std: float = 0.03,
                 per_axis_scale: bool = False,
                 rng_seed: int = 42) -> Tuple[np.ndarray, np.ndarray, List[Path], List[str]]:
    rng = np.random.RandomState(rng_seed)

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    files: List[Path] = []
    classes: List[str] = []

    for class_dir in sorted([p for p in data_dir.iterdir() if p.is_dir()]):
        classes.append(class_dir.name)
    label_to_id: Dict[str, int] = {c: i for i, c in enumerate(classes)}

    for xyz, label, path in iter_trajectories(data_dir):
        xyz_res = resample_traj(xyz, n=resample_len)

        # original
        feats = extract_features(xyz_res)
        X_list.append(feats); y_list.append(label_to_id[label]); files.append(path)

        # augmentations simples (sans rot/warp)
        if augment and aug_per_sample > 0:
            for _ in range(aug_per_sample):
                P_aug = augment_one_simple(
                    xyz_res,
                    jitter_sigma_mm=jitter_sigma_mm,
                    scale_std=scale_std,
                    per_axis_scale=per_axis_scale,
                    rng=rng
                )
                feats_aug = extract_features(P_aug)
                X_list.append(feats_aug); y_list.append(label_to_id[label]); files.append(path)

    if not X_list:
        raise RuntimeError("No trajectories found. Check your data_dir structure.")

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=int)
    return X, y, files, classes


def parse_xyz_from_col6(line: str) -> np.ndarray:
    """
    Parse a single line with comma-separated columns, reading only column index 6 (the 7th column),
    which is expected to contain 'X/Y/Z' like '392/-440/-84'. Returns np.array([X,Y,Z], dtype=float).
    Ignores malformed lines.
    """
    parts = line.strip().split(',')
    if len(parts) <= 6:
        raise ValueError("Line has fewer than 7 columns; cannot read column [6].")
    col6 = parts[6].strip()
    # Accept separators "/", possibly with spaces
    xyz_parts = re.split(r'\s*/\s*', col6)
    if len(xyz_parts) != 3:
        raise ValueError(f"Column[6] is not 'X/Y/Z' format: got '{col6}'")
    try:
        x, y, z = map(float, xyz_parts)
    except ValueError as e:
        raise ValueError(f"Failed to parse floats from '{col6}': {e}")
    return np.array([x, y, z], dtype=float)


def read_trajectory_txt(txt_path: Path) -> np.ndarray:
    """
    Read a .txt trajectory file. Each line has comma-separated columns; we extract ONLY column [6]
    and build a sequence of 3D points. Returns array of shape (T, 3).
    """
    seq = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                xyz = parse_xyz_from_col6(line)
                seq.append(xyz)
            except Exception:
                # Silently skip malformed lines; alternatively, raise to catch file issues
                continue
    if len(seq) == 0:
        raise ValueError(f"No valid XYZ rows parsed from {txt_path}")
    return np.vstack(seq)  # (T, 3)


def resample_traj(xyz: np.ndarray, n: int = 100) -> np.ndarray:
    """
    Linear interpolation to resample a 3D trajectory (T,3) into exactly n samples.
    """
    T = xyz.shape[0]
    if T == 1:
        return np.repeat(xyz, n, axis=0)
    t_old = np.linspace(0.0, 1.0, T)
    t_new = np.linspace(0.0, 1.0, n)
    out = np.column_stack([np.interp(t_new, t_old, xyz[:, i]) for i in range(3)])
    return out  # (n, 3)


def extract_features(xyz: np.ndarray) -> np.ndarray:
    """
    Compute interpretable geometric features from a resampled trajectory (n,3).
    Returns a 1D vector of features.
    """
    P = xyz
    dP = np.diff(P, axis=0)  # (n-1, 3)
    speeds = np.linalg.norm(dP, axis=1)  # (n-1,)
    length = speeds.sum()  # total path length
    disp_vec = P[-1] - P[0]
    disp = float(np.linalg.norm(disp_vec))  # net displacement

    # Avoid division by zero
    rectilinearity = disp / (length + 1e-9)

    amp = P.max(axis=0) - P.min(axis=0)      # amplitude per axis (X,Y,Z)
    var = P.var(axis=0)                      # variance per axis
    mean_pos = P.mean(axis=0)

    # Direction (start->end) via azimuth (xy-plane) and elevation
    azim = float(np.arctan2(disp_vec[1], disp_vec[0]))
    elev = float(np.arctan2(disp_vec[2], np.linalg.norm(disp_vec[:2]) + 1e-9))

    # Curvature proxy: variance of direction changes
    eps = 1e-9
    dirs = dP / (speeds[:, None] + eps)  # (n-1, 3)
    turn = np.diff(dirs, axis=0)         # (n-2, 3)
    turn_mag = np.linalg.norm(turn, axis=1) if len(turn) > 0 else np.array([0.0])
    turn_var = float(np.var(turn_mag))

    # Speed stats
    speed_mean = float(np.mean(speeds)) if len(speeds) > 0 else 0.0
    speed_std = float(np.std(speeds)) if len(speeds) > 0 else 0.0

    feats = np.hstack([
        length, disp, rectilinearity,
        amp, var, mean_pos,
        azim, elev, turn_var,
        disp / (length + 1e-9),  # closure_ratio
        speed_mean, speed_std
    ]).astype(float)

    return feats  # shape (F,)


def iter_trajectories(data_dir: Path) -> Iterator[Tuple[np.ndarray, str, Path]]:
    """
    Yield (trajectory_xyz, class_label, file_path) for all .txt files in subfolders.
    Class label = subfolder name.
    """
    for class_dir in sorted([p for p in data_dir.iterdir() if p.is_dir()]):
        label = class_dir.name
        for txt in sorted(class_dir.rglob("*.txt")):
            xyz = read_trajectory_txt(txt)  # (T,3) strictly from column [6]
            yield xyz, label, txt


def plot_and_save_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: Path):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title('Confusion Matrix')
    fig.colorbar(im, ax=ax)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(
    data_dir: str,
    resample_len: int = 100,
    save_dir: str = "./outputs",
    cv_splits: int = 5,
    save_npy: bool = False,
    augment: bool = False,
    aug_per_sample: int = 0,
    jitter_sigma_mm: float = 1.5,
    scale_std: float = 0.03,
    per_axis_scale: bool = False,
    rng_seed: Optional[int] = None,
):
    data_dir = Path(data_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset from:", data_dir)
    X, y, files, class_names = load_dataset(
        data_dir,
        resample_len=resample_len,
        augment=augment,
        aug_per_sample=aug_per_sample,
        jitter_sigma_mm=jitter_sigma_mm,
        scale_std=scale_std,
        per_axis_scale=per_axis_scale,
        rng_seed=int(time.time()) if rng_seed is None else int(rng_seed),
    )
    print(f"Loaded {len(y)} trajectories; feature dim = {X.shape[1]}; classes = {class_names}")

    first_xyz = read_trajectory_txt(files[4])  # (T,3)
    first_res = resample_traj(first_xyz, n=resample_len)  # (n,3)
    first_aug = augment_one_simple(
        first_res,
        jitter_sigma_mm=jitter_sigma_mm,
        scale_std=scale_std,
        per_axis_scale=per_axis_scale,
        rng=np.random.RandomState(123)  # pour reproductibilité
    )


    # Pipeline: Standardize -> SVM(RBF)
    clf = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=3.0, gamma="scale", class_weight="balanced",
            probability=True, random_state=42)
    )

    # GroupKFold pour éviter la fuite entre originaux et augmentations
    groups = [str(p) for p in files]
    cv = GroupKFold(n_splits=cv_splits)

    # CV F1-macro
    f1_scores = cross_val_score(clf, X, y, cv=cv, scoring="f1_macro", groups=groups)
    print(f"CV F1-macro: {f1_scores.mean():.3f} ± {f1_scores.std():.3f}")

    # Prédictions CV et matrice de confusion
    y_pred = cross_val_predict(clf, X, y, cv=cv, method="predict", groups=groups)
    cm = confusion_matrix(y, y_pred)
    cm_path = save_dir / "confusion_matrix.png"
    plot_and_save_confusion_matrix(cm, class_names, cm_path)
    print(f"Saved confusion matrix to: {cm_path}")

    # Rapport de classification
    report = classification_report(y, y_pred, target_names=class_names, digits=3)
    report_path = save_dir / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print("Classification report:\n", report)
    print(f"Saved classification report to: {report_path}")

    # Sauvegarde optionnelle des features/labels
    if save_npy:
        np.save(save_dir / "features.npy", X)
        np.save(save_dir / "labels.npy", y)
        print("Saved features.npy and labels.npy")

    # Fit final sur tout le jeu
    clf.fit(X, y)

    print("\nDone.")
    return {
        "model": clf,
        "classes": class_names,
        "features": X,
        "labels": y,
        "confusion_matrix": cm,
        "f1_scores": f1_scores,
        "report_path": str(report_path),
        "cm_path": str(cm_path),
    }


def main():
    run(
        data_dir=r"../data1",
        resample_len=100,
        save_dir="./outputs",
        cv_splits=5,
        save_npy=False,
        augment=True,
        aug_per_sample=2,
        jitter_sigma_mm=1.5,
        scale_std=0.03,
        per_axis_scale=False,
        rng_seed=None,
    )


if __name__ == "__main__":
    main()
