from pathlib import Path
import torch
import datetime
import numpy as np
from numpy.linalg import norm


def get_checkpoint_path():
    """
    Returns a Path object pointing to the "configs" directory.
    """
    current_dir = Path(__file__).resolve().parent
    checkpoints_path = current_dir / "checkpoints"
    return checkpoints_path


def get_config_path():
    """
    Returns a Path object pointing to the "configs" directory.
    """
    current_dir = Path(__file__).resolve().parent
    config_path = current_dir / "configs"
    return config_path


def get_data_loader_path():
    """
    Returns a Path object pointing to the "configs" directory.
    """
    current_dir = Path(__file__).resolve().parent
    data_loader_path = current_dir / "data_loader"
    return data_loader_path


def get_model_path():
    """
    Returns a Path object pointing to the "configs" directory.
    """
    current_dir = Path(__file__).resolve().parent
    model_path = current_dir / "model"
    return model_path


def save_checkpoint(state, filename: str, current_daytime: str):
    # print("=> Saving checkpoint")
    checkpoint_path = get_checkpoint_path()

    (checkpoint_path/filename).mkdir(parents=True, exist_ok=True)
    torch.save(state,  checkpoint_path / filename /
               f'{filename}_{current_daytime}.pth.tar')


def load_checkpoint(checkpoint, model, optimizer, tr_metrics, val_metrics):
    # print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    tr_metrics = checkpoint['tr_metrics']
    # val_metrics = checkpoint['val_metrics']

    return (tr_metrics)  # val_metrics


def ergas_batch(reference_batch, synthesized_batch, scale_ratio):
    reference_batch = reference_batch.cpu().numpy()
    synthesized_batch = synthesized_batch.cpu().numpy()

    n, h, w, c = reference_batch.shape
    rmse = np.sqrt(
        np.mean((reference_batch - synthesized_batch) ** 2, axis=(1, 2)))
    mean_ref = np.mean(reference_batch, axis=(1, 2))
    ergas_values = 100 * scale_ratio * \
        np.sqrt(np.mean((rmse / mean_ref) ** 2, axis=1))
    return ergas_values


def ergas(ms, ps, ratio=4):  # FIXME change that for Sev2Mod
    ms = ms.cpu().numpy()
    ps = ps.cpu().numpy()
    ms = ms.astype(np.float32)
    ps = ps.astype(np.float32)
    err = ms - ps
    ergas_index = 0
    for i in range(err.shape[0]):
        ergas_index += np.mean(np.square(err[i, :, :])) / \
            np.square(np.mean(ms[i, :, :]))

    ergas_index = (100/ratio) * np.sqrt(1/err.shape[0]) * ergas_index

    return ergas_index


def sam_batch(reference_batch, synthesized_batch):
    reference_batch = reference_batch.cpu().numpy()
    synthesized_batch = synthesized_batch.cpu().numpy()

    product = np.sum(reference_batch * synthesized_batch, axis=3)
    norm_ref = np.linalg.norm(reference_batch, axis=3)
    norm_syn = np.linalg.norm(synthesized_batch, axis=3)
    cos_theta = product / (norm_ref * norm_syn)
    # Ensure the values are within [-1, 1]
    cos_theta = np.clip(cos_theta, -1, 1)
    sam_values = np.mean(np.arccos(cos_theta), axis=(1, 2))
    return sam_values


def sam(ms, ps):
    ms = ms[0].cpu().numpy()
    ps = ps[0].cpu().numpy()

    assert ms.ndim == 3 and ms.shape == ps.shape

    ms = ms.astype(np.float32)
    ps = ps.astype(np.float32)

    dot_sum = np.sum(ms*ps, axis=2)
    norm_true = norm(ms, axis=2)
    norm_pred = norm(ps, axis=2)

    res = np.arccos(dot_sum/norm_pred/norm_true)

    is_nan = np.nonzero(np.isnan(res))

    for (x, y) in zip(is_nan[0], is_nan[1]):
        res[x, y] = 0

    sam = np.mean(res)

    return sam * 180 / np.pi
