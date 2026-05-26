import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from unet import UNet
from carvana_dataset import CarvanaDataset

if __name__ == "__main__":
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 32
    EPOCHS = 2
    DATA_PATH = "/content/drive/MyDrive/uygar/unet-segmentation/data"
    MODEL_SAVE_PATH = "/content/drive/MyDrive/uygar/unet-segmentation/models/unet.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataset = CarvanaDataset(DATA_PATH)

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(train_dataset, [0.8, 0.2], generator=generator)

    train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True)

    model = UNet(in_channels=3, num_classes=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_running_loss = 0

        # idx tracks which batch number you are on
        # img_mask hold the data tuple
        # index 0 is the img of the car
        # index 1 is the ground truth mask
        for idx, img_mask in enumerate(tqdm(train_dataloader)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)


            # y pred the output(guess) the model made after the up and downsampling
            y_pred = model(img)

            #clearing up the old gradient in order to avoid accumulation of errors
            optimizer.zero_grad()

            loss = criterion(y_pred, mask)

            #outputs the raw python number out of the pytorch loss tensor
            train_running_loss += loss.item()
            
            # starts at final loss score adn then works it way up to every single layer of UNet
            loss.backward()
            optimizer.step()
        # total training loss / total number of batches processed
        train_loss = train_running_loss / (idx + 1)


        # validation after training
        # Dropput - no nuerons deactivated from over here now on
        model.eval()
        val_running_loss = 0

        #not changing any weigths hence no grad(this is the final model)
        with torch.no_grad():
            for idx, img_mask in enumerate(tqdm(val_dataloader)):
                img = img_mask[0].float().to(device)
                mask = img_mask[1].float().to(device)
                
                y_pred = model(img)
                loss = criterion(y_pred, mask)

                val_running_loss += loss.item()

            val_loss = val_running_loss / (idx + 1)

        print("-"*30)
        print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
        print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.4f}")
        print("-"*30)


    # model.dict -> instead of extracting every single code, this creates a python dictionary that pairs every layer name with its final, calibrated weight tensor matrix
    # torch.save will save this into the model_save_path in the gdrive can extract it over in the inference 
    torch.save(model.state_dict(), MODEL_SAVE_PATH)