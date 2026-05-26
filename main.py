import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from unet import UNet
from carvana_dataset import NOAATornadoDataset


def kl_divergence(pred_probs, target, eps=1e-8):
    """KL(target || pred) averaged over batch, channels, and spatial dims."""
    target = target.clamp(eps, 1.0 - eps)
    pred_probs = pred_probs.clamp(eps, 1.0 - eps)
    kl_pos = target * (target.log() - pred_probs.log())
    kl_neg = (1.0 - target) * ((1.0 - target).log() - (1.0 - pred_probs).log())
    kl = kl_pos + kl_neg
    return kl.mean()


if __name__ == "__main__":
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 64
    EPOCHS = 10
    DATA_PATH = "./data"  # update to your local or Colab path
    MODEL_SAVE_PATH = "./models/unet.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    full_dataset = NOAATornadoDataset(DATA_PATH, test=False)

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [0.8, 0.2], generator=generator)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    model = UNet(in_channels=3, num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Higher weight on rare positive (tornado) pixels; tune as needed
    pos_weight = torch.tensor([50.0, 100.0]).view(1, 2, 1, 1).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(EPOCHS):
        # --- Training ---
        model.train()
        train_running_loss = 0.0

        for x, y in tqdm(train_dataloader, desc=f"Train epoch {epoch + 1}/{EPOCHS}"):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)

            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()

        train_loss = train_running_loss / len(train_dataloader)

        # --- Validation ---
        model.eval()
        val_running_loss = 0.0
        val_kl_total = 0.0

        with torch.no_grad():
            for x, y in tqdm(val_dataloader, desc=f"Val epoch {epoch + 1}/{EPOCHS}"):
                x = x.to(device)
                y = y.to(device)

                y_pred = model(x)
                loss = criterion(y_pred, y)
                val_running_loss += loss.item()

                y_pred_probs = torch.sigmoid(y_pred)
                val_kl_total += kl_divergence(y_pred_probs, y).item()

        val_loss = val_running_loss / len(val_dataloader)
        avg_kl = val_kl_total / len(val_dataloader)

        print("-" * 50)
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"  Train BCE: {train_loss:.4f}")
        print(f"  Val BCE:   {val_loss:.4f}")
        print(f"  Val KL:    {avg_kl:.4f}")
        print("-" * 50)

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")