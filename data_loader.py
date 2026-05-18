from torch.utils.data import DataLoader

from classe_dataset import WildfireDataset

# Definizione dei percorsi delle cartelle
cartella_immagini_satellite = r'C:\Users\39324\Desktop\Dataset Kaggle\Images_PNG_Pronte'
cartella_maschere_satellite = r'C:\Users\39324\Desktop\Dataset Kaggle\Mask_PNG_Pronte'

# Creazione oggetto Dataset
mio_dataset = WildfireDataset(cartella_immagini_satellite, cartella_maschere_satellite)

# Creazione del DataLoader 
# Tramite PyTorch si prende il dataset e lo si impacchetta in "gruppi" (batch)
# In particolare qui sono prese 16 immagini alla volta che vengono poi mescolate (shuffle=True)
generatore_dati = DataLoader(mio_dataset, batch_size=16, shuffle=True)