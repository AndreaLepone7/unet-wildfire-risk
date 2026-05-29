import torch
import torch.nn as nn

#  BLOCCO COSTRUTTIVO BASE (Conv2dBlock)
class Conv2dBlock(nn.Module):
    """
    Il blocco fondamentale della U-Net.
    Comprende due operazioni convoluzionali, seguite da batch normalization e attivazione ReLU.
    """
    def __init__(self, in_channels, out_channels):
        super(Conv2dBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x

# BLOCCO DI ATTENZIONE (PYTORCH)
class NativeSelfAttention(nn.Module):
    """
    Adattatore per utilizzare nn.MultiheadAttention su feature map 2D.
    """
    def __init__(self, channels):
        super(NativeSelfAttention, self).__init__()
        
        # Modulo nativo di PyTorch. 
        # num_heads=8 divide i canali in 8 gruppi per cercare pattern simultanei differenti
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=8, batch_first=True)
        
        # LayerNorm è standard dopo i blocchi di attenzione per stabilizzare i gradienti
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # x ha dimensione: [Batch, Canali, Altezza, Larghezza] -> [B, C, H, W]
        B, C, H, W = x.size()
        
        # FLATTEN SPAZIALE: Da Griglia 2D a Sequenza 1D
        # La MultiheadAttention si aspetta [Batch, Lunghezza_Sequenza, Features]
        # Fondiamo Altezza e Larghezza in un'unica dimensione (H * W)
        x_flat = x.view(B, C, H * W).permute(0, 2, 1) # Diventa [B, H*W, C]
        
        # SELF-ATTENTION
        # Passiamo x_flat come Query, Key e Value
        attn_out, _ = self.mha(x_flat, x_flat, x_flat)
        
        # RESIDUAL CONNECTION E NORMALIZZAZIONE
        out_flat = self.norm(x_flat + attn_out)
        
        # RESHAPE SPAZIALE: Ritorno alla Griglia 2D
        # Riportiamo il tensore a [B, C, H, W] per ripassarlo alle convoluzioni
        out = out_flat.permute(0, 2, 1).view(B, C, H, W)
        
        return out

# 3. ARCHITETTURA COMPLETA (Attention Bottleneck UNet)
class UNet(nn.Module):
    """
    L'architettura U-Net completa ibridata con Self-Attention nel Bottleneck.
    """
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        self.dropout = nn.Dropout2d(p=0.07)

        # PERCORSO DELL'ENCODER
        self.enc1 = Conv2dBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = Conv2dBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = Conv2dBlock(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = Conv2dBlock(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # BOTTLENECK CON COMPITO DI ATTENZIONE
        self.bottleneck_conv = Conv2dBlock(512, 1024)
        self.bottleneck_attn = NativeSelfAttention(channels=1024) # Iniezione del modulo

        # PERCORSO DEL DECODER
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = Conv2dBlock(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = Conv2dBlock(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = Conv2dBlock(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = Conv2dBlock(128, 64)

        # LAYER FINALE
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Discesa (Encoder) 
        e1 = self.enc1(x)
        p1 = self.pool1(self.dropout(e1))

        e2 = self.enc2(p1)
        p2 = self.pool2(self.dropout(e2))

        e3 = self.enc3(p2)
        p3 = self.pool3(self.dropout(e3))

        e4 = self.enc4(p3)
        p4 = self.pool4(self.dropout(e4))

        # Collo di bottiglia (Bottleneck)
        b = self.bottleneck_conv(p4)
        b = self.bottleneck_attn(b) # L'Attenzione agisce sulla massima astrazione semantica

        # Risalita (Decoder) e Skip Connections 
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

        out = self.final_conv(d1)
        out = self.sigmoid(out) 
        
        return out