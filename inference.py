import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF

from carvana_dataset import NOAATornadoDataset
from unet import UNet


def load_model(model_path, device):
    model = UNet(in_channels=3, num_classes=2).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def _prepare_input(x):
    """Same normalization as NOAATornadoDataset."""
    x = x.float()
    x = x / (x.abs().amax(dim=(1, 2), keepdim=True) + 1e-8)
    x = TF.resize(x, [256, 256], antialias=True)
    return x


def _to_numpy_chw(tensor):
    return tensor.detach().cpu().numpy()


@torch.no_grad()
def pred_show_image_grid(data_path, model_path, device, max_samples=None):
    """
    Grid over manual_test set.
    Rows per column (one sample):
      1. CAPE input
      2. True tornado probability
      3. Predicted tornado probability
      4. True significant-tornado probability
      5. Predicted significant-tornado probability
    """
    model = load_model(model_path, device)
    dataset = NOAATornadoDataset(data_path, test=True)

    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    if n == 0:
        raise ValueError("Test dataset is empty. Check ./data/manual_test/ layout.")

    cape_inputs = []
    true_tor = []
    pred_tor = []
    true_sig = []
    pred_sig = []

    for i in range(n):
        x, y = dataset[i]
        x = x.to(device)
        x_batch = x.unsqueeze(0)

        logits = model(x_batch)
        probs = torch.sigmoid(logits).squeeze(0).cpu()  # [2, H, W]

        cape = _to_numpy_chw(x[0].cpu())
        y_np = _to_numpy_chw(y)

        cape_inputs.append(cape)
        true_tor.append(y_np[0])
        pred_tor.append(probs[0].numpy())
        true_sig.append(y_np[1])
        pred_sig.append(probs[1].numpy())

    rows = [
        ("CAPE (input)", cape_inputs, "viridis"),
        ("True tornado prob", true_tor, "hot"),
        ("Pred tornado prob", pred_tor, "hot"),
        ("True sig. tornado prob", true_sig, "hot"),
        ("Pred sig. tornado prob", pred_sig, "hot"),
    ]

    fig, axes = plt.subplots(len(rows), n, figsize=(3 * n, 3 * len(rows)))
    if n == 1:
        axes = axes.reshape(-1, 1)

    for row_idx, (title, images, cmap) in enumerate(rows):
        for col_idx in range(n):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(images[col_idx], cmap=cmap, vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(title)
            if row_idx == 0:
                ax.set_title(f"sample {col_idx}")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


@torch.no_grad()
def single_sample_inference(data_path, model_path, device, sample_index=0):
    """Run inference on one test sample by index."""
    model = load_model(model_path, device)
    dataset = NOAATornadoDataset(data_path, test=True)

    if sample_index >= len(dataset):
        raise IndexError(f"sample_index {sample_index} out of range (len={len(dataset)})")

    x, y = dataset[sample_index]
    x = x.to(device).unsqueeze(0)
    probs = torch.sigmoid(model(x)).squeeze(0).cpu().numpy()  # [2, H, W]
    y_np = _to_numpy_chw(y)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    panels = [
        (x[0, 0].cpu().numpy(), "CAPE (input)", "viridis", None),
        (y_np[0], "True tornado", "hot", (0, 1)),
        (probs[0], "Pred tornado", "hot", (0, 1)),
        (y_np[1], "True sig. tornado", "hot", (0, 1)),
        (probs[1], "Pred sig. tornado", "hot", (0, 1)),
    ]

    for ax, (img, title, cmap, vrange) in zip(axes.flat[:5], panels):
        if vrange is None:
            im = ax.imshow(img, cmap=cmap)
        else:
            im = ax.imshow(img, cmap=cmap, vmin=vrange[0], vmax=vrange[1])
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    axes[1, 2].axis("off")
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def single_day_inference_from_npy(data_path, model_path, device, sample_id, test=True):
    """
    Run inference when you know the filename/id (e.g. '2017-05-18.npy')
    without using dataset index order.
    """
    model = load_model(model_path, device)
    folder_prefix = "manual_test" if test else "train"

    cape = np.load(os.path.join(data_path, folder_prefix, "cape", sample_id))
    cin = np.load(os.path.join(data_path, folder_prefix, "cin", sample_id))
    geo = np.load(os.path.join(data_path, folder_prefix, "geo", sample_id))

    tor_path = os.path.join(data_path, f"{folder_prefix}_masks", "tornado", sample_id)
    sig_path = os.path.join(data_path, f"{folder_prefix}_masks", "sigtor", sample_id)
    y = None
    if os.path.exists(tor_path) and os.path.exists(sig_path):
        tor = np.load(tor_path)
        sig = np.load(sig_path)
        y = torch.from_numpy(np.stack([tor, sig], axis=0)).float()
        y = TF.resize(y, [256, 256], interpolation=TF.InterpolationMode.NEAREST)

    x = torch.from_numpy(np.stack([cape, cin, geo], axis=0)).float()
    x = _prepare_input(x).to(device).unsqueeze(0)

    probs = torch.sigmoid(model(x)).squeeze(0).cpu().numpy()

    ncols = 3 if y is not None else 2
    fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 8))

    ax = axes[0, 0]
    im = ax.imshow(x[0, 0].cpu().numpy(), cmap="viridis")
    ax.set_title("CAPE (input)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[0, 1]
    im = ax.imshow(probs[0], cmap="hot", vmin=0, vmax=1)
    ax.set_title("Pred tornado")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 1]
    im = ax.imshow(probs[1], cmap="hot", vmin=0, vmax=1)
    ax.set_title("Pred sig. tornado")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if y is not None:
        y_np = _to_numpy_chw(y)
        ax = axes[0, 2]
        im = ax.imshow(y_np[0], cmap="hot", vmin=0, vmax=1)
        ax.set_title("True tornado")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[1, 2]
        im = ax.imshow(y_np[1], cmap="hot", vmin=0, vmax=1)
        ax.set_title("True sig. tornado")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    axes[1, 0].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    DATA_PATH = "./data"
    MODEL_PATH = "./models/unet.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Grid over all manual_test samples (limit with max_samples=5 for speed)
    pred_show_image_grid(DATA_PATH, MODEL_PATH, device, max_samples=5)

    # One sample by dataset index
    single_sample_inference(DATA_PATH, MODEL_PATH, device, sample_index=0)

    # Optional: one day by filename under manual_test/cape/
    # single_day_inference_from_npy(DATA_PATH, MODEL_PATH, device, "2017-05-18.npy")