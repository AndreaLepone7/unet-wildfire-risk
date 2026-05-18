import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

class WildfireDataset(Dataset):
    def __init__(self, cartella_immagini, cartella_maschere):
        """
        Inizializza il dataset cercando tutte le coppie immagine/maschera.
        """
        self.immagini_paths = []
        self.maschere_paths = []

        # os.walk entra in tutte le sottocartelle (After, Before, During, ecc.)
        for root, dirs, files in os.walk(cartella_immagini):
            for file in files:
                if file.endswith('.png'):
                    # Percorso completo dell'immagine
                    img_path = os.path.join(root, file)
                    
                    # Calcola il percorso speculare della maschera
                    percorso_relativo = os.path.relpath(img_path, cartella_immagini)
                    mask_path = os.path.join(cartella_maschere, percorso_relativo)

                    # Se la maschera corrispondente esiste, aggiungiamo la coppia alla lista
                    if os.path.exists(mask_path):
                        self.immagini_paths.append(img_path)
                        self.maschere_paths.append(mask_path)

        print(f"Dataset inizializzato: trovate {len(self.immagini_paths)} coppie valide.")

    def __len__(self):
        """Restituisce il numero totale di campioni."""
        return len(self.immagini_paths)

    def __getitem__(self, idx):
        """
        Pesca l'immagine e la maschera all'indice 'idx', le converte in Tensori e le restituisce.
        """
        img_path = self.immagini_paths[idx]
        mask_path = self.maschere_paths[idx]

        # Apre le immagini usando la libreria PIL (Python Imaging Library)
        # .convert("RGB") assicura che l'immagine abbia 3 canali
        immagine = Image.open(img_path).convert("RGB")
        # .convert("L") assicura che la maschera sia in scala di grigi (1 canale)
        maschera = Image.open(mask_path).convert("L")

        # Converte le immagini in Tensori di PyTorch
        # TF.to_tensor converte i pixel da (0-255) a numeri decimali tra (0.0 - 1.0)
        img_tensor = TF.to_tensor(immagine)
        mask_tensor = TF.to_tensor(maschera)

        # Pulizia della maschera
        # La maschera deve indicare solo le classi 0 (sfondo) e 1 (fuoco),
        # quindi arrotondiamo i valori per assicurarci che non ci siano decimali intermedi 
        # dovuti ad artefatti di salvataggio del PNG.
        mask_tensor = torch.round(mask_tensor)

        return img_tensor, mask_tensor