import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from classe_dataset import WildfireDataset
from unet import UNet
import matplotlib.pyplot as plt

# DEFINIZIONE DELLA NUOVA FUNZIONE DI LOSS (Dice + Focal)
class DiceFocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0, smooth=1e-6):
        super(DiceFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Appiattiamo i tensori
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # CALCOLO DICE LOSS 
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        dice_loss = 1 - dice

        # CALCOLO FOCAL LOSS 
        BCE = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE)
        focal_loss = self.alpha * (1 - pt)**self.gamma * BCE
        focal_loss = focal_loss.mean()

        # Calcolo Somma Totale per l'ottimizzatore
        total_loss = dice_loss + focal_loss

        # Restituzione di tutti e tre i valori: Il totale per il gradiente, i singoli per il monitoraggio
        return total_loss, dice_loss, focal_loss

# MOTORE DI ADDESTRAMENTO
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
    
    criterion = DiceFocalLoss(alpha=0.8, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    train_loss_history = []
    val_loss_history = []

    epochs = 25
    best_val_loss = float('inf')

    print("\nInizio ciclo di epoche...")
    print("-" * 80)

    for epoch in range(epochs):
        # FASE DI TRAINING
        model.train()
        train_loss_tot, train_loss_dice, train_loss_focal = 0.0, 0.0, 0.0
        
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            # Spacchettamento dei tre valori restituiti dalla nostra classe
            loss, d_loss, f_loss = criterion(outputs, masks)
            
            # La backpropagation applicata SOLO alla somma totale
            loss.backward()
            optimizer.step()
            
            # Estrazione dei valori numerici puri (.item()) per le statistiche
            train_loss_tot += loss.item() * images.size(0)
            train_loss_dice += d_loss.item() * images.size(0)
            train_loss_focal += f_loss.item() * images.size(0)

        # FASE DI VALIDAZIONE
        model.eval()
        val_loss_tot, val_loss_dice, val_loss_focal = 0.0, 0.0, 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                
                outputs = model(images)
                loss, d_loss, f_loss = criterion(outputs, masks)
                
                val_loss_tot += loss.item() * images.size(0)
                val_loss_dice += d_loss.item() * images.size(0)
                val_loss_focal += f_loss.item() * images.size(0)

        # Calcolo medie di fine epoca
        epoch_train_tot = train_loss_tot / train_size
        epoch_train_dice = train_loss_dice / train_size
        epoch_train_focal = train_loss_focal / train_size
        
        epoch_val_tot = val_loss_tot / val_size
        epoch_val_dice = val_loss_dice / val_size
        epoch_val_focal = val_loss_focal / val_size

        # Stampa diagnostica completa a schermo
        print(f"Epoca [{epoch+1}/{epochs}]")
        print(f"   ➤ TRAIN | Totale: {epoch_train_tot:.4f} (Dice: {epoch_train_dice:.4f} | Focal: {epoch_train_focal:.4f})")
        print(f"   ➤ VAL   | Totale: {epoch_val_tot:.4f} (Dice: {epoch_val_dice:.4f} | Focal: {epoch_val_focal:.4f})")

        # Salvataggio storia per il grafico (Manteniamo il tracking sulla totale)
        train_loss_history.append(epoch_train_tot)
        val_loss_history.append(epoch_val_tot)

        # Aggiornamento dinamico del grafico
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epoch + 2), train_loss_history, label='Train Loss Totale', marker='o')
        plt.plot(range(1, epoch + 2), val_loss_history, label='Val Loss Totale', marker='o')
        plt.xlabel('Epoche')
        plt.ylabel('Loss (Dice + Focal)')
        plt.title('Andamento dell\'addestramento (U-Net Wildfire)')
        plt.legend()
        plt.grid(True)
        
        plt.savefig('../loss_plot.png') 
        plt.close() 

        # Salvataggio del modello migliore in base alla metrica totale di validazione
        if epoch_val_tot < best_val_loss:
            best_val_loss = epoch_val_tot
            torch.save(model.state_dict(), '../best_unet_wildfire.pth')
            print("   ↳  Nuovo record! Modello salvato.")
        print("-" * 80)

if __name__ == '__main__':
    train_model()