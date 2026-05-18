import torch
import torch.nn as nn

# BLOCCO COSTRUTTIVO BASE (Conv2dBlock)
class Conv2dBlock(nn.Module):
    """
    Il blocco fondamentale della U-Net.
    Comprende due operazioni convoluzionali, seguite da batch normalization e attivazione ReLU 
    """
    def __init__(self, in_channels, out_channels):
        super(Conv2dBlock, self).__init__()
        
        # Prima convoluzione + Normalizzazione + Attivazione
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Seconda convoluzione + Normalizzazione + Attivazione
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x

# ARCHITETTURA COMPLETA (UNet)
class UNet(nn.Module):
    """
    L'architettura U-Net completa per la segmentazione semantica delle immagini satellitari.
    Include l'Encoder (discesa), il Bottleneck, e il Decoder (risalita) con Skip Connections 
    """
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        # In riferimento al paper consultato (https://arxiv.org/pdf/2502.05476), si aggiungeregolarizzazione tramite Dropout2d
        self.dropout = nn.Dropout2d(p=0.07)

        # PERCORSO DELL'ENCODER (Estrazione delle feature)
        self.enc1 = Conv2dBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = Conv2dBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = Conv2dBlock(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = Conv2dBlock(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # BOTTLENECK (Punto di massima compressione)
        self.bottleneck = Conv2dBlock(512, 1024)

        # PERCORSO DEL DECODER (Ricostruzione spaziale)
        # Upsampling tramite ConvTranspose2d  
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = Conv2dBlock(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = Conv2dBlock(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = Conv2dBlock(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = Conv2dBlock(128, 64)

        # LAYER FINALE E ATTIVAZIONE
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # Nel layer finale utilizzo della funzione di attivazione sigmoide per generare maschere binarie pixel per pixel  
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Discesa (Encoder) 
        e1 = self.enc1(x)
        e1_drop = self.dropout(e1)
        p1 = self.pool1(e1_drop)

        e2 = self.enc2(p1)
        e2_drop = self.dropout(e2)
        p2 = self.pool2(e2_drop)

        e3 = self.enc3(p2)
        e3_drop = self.dropout(e3)
        p3 = self.pool3(e3_drop)

        e4 = self.enc4(p3)
        e4_drop = self.dropout(e4)
        p4 = self.pool4(e4_drop)

        # Collo di bottiglia (Bottleneck) 
        b = self.bottleneck(p4)

        # Risalita (Decoder) e Skip Connections 
        # Si utilizza torch.cat per concatenare le mappe delle feature dell'encoder con quelle del decoder  
        d4 = self.upconv4(b)
        d4 = torch.cat((d4, e4), dim=1)  
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)

        # Output finale
        out = self.final_conv(d1)
        out = self.sigmoid(out) # Comprime i valori tra 0 (non incendio) e 1 (incendio)  
        
        return out