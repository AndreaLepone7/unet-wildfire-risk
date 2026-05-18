import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from classe_dataset import WildfireDataset
from unet import UNet
import matplotlib.pyplot as plt # IMPORTAZIONE PER I GRAFICI

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Avvio addestramento sul device: {device}")

    cartella_img = os.environ.get('DATASET_IMG', r'C:\Users\39324\Desktop\Dataset Kaggle\Images_PNG_Pronte')
    cartella_mask = os.environ.get('DATASET_MASK', r'C:\Users\39324\Desktop\Dataset Kaggle\Mask_PNG_Pronte')

    dataset_completo = WildfireDataset(cartella_img, cartella_mask)
    
    total_size = len(dataset_completo)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = random_split(dataset_completo, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

    model = UNet(in_channels=3, out_channels=1).to(device)
    
    criterion = nn.BCELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Liste per memorizzare lom storico della Loss ad ogni epoca
    train_loss_history = []
    val_loss_history = []

    epochs = 20 # Numero di epoche per l'addestramento
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # FASE DI TRAINING
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)

        # FASE DI VALIDAZIONE 
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)

        epoch_train_loss = train_loss / train_size
        epoch_val_loss = val_loss / val_size

        print(f"Epoca [{epoch+1}/{epochs}] | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

        # Salvataggio dei valori correnti nelle liste della cronologia
        train_loss_history.append(epoch_train_loss)
        val_loss_history.append(epoch_val_loss)

        # Generazione e salvataggio del grafico (si sovrascrive aggiornandosi ad ogni epoca)
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epoch + 2), train_loss_history, label='Train Loss', marker='o')
        plt.plot(range(1, epoch + 2), val_loss_history, label='Val Loss', marker='o')
        plt.xlabel('Epoche')
        plt.ylabel('Loss (Errore)')
        plt.title('Andamento dell\'addestramento (U-Net Wildfire)')
        plt.legend()
        plt.grid(True)
        
        # Salvataggio dell'immagine fuori dalla cartella della repo, direttamente nell'output di Kaggle
        plt.savefig('../loss_plot.png') 
        plt.close() # Chiusura figura per non intasare la memoria RAM
        print("Grafico della Loss aggiornato e salvato!")

        # Salvataggio del modello migliore 
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), '../best_unet_wildfire.pth')
            print("Salvataggio del modello migliore aggiornato!")

if __name__ == '__main__':
    train_model()