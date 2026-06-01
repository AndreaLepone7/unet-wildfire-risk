import torch
import torch.nn as nn

class Conv2dBlock(nn.Module):
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

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        psi = self.relu(g1 + x1)
        alpha = self.psi(psi)
        
        return x * alpha

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        self.dropout = nn.Dropout2d(p=0.07)

        # ENCODER
        self.enc1 = Conv2dBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = Conv2dBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = Conv2dBlock(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = Conv2dBlock(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # BOTTLENECK 
        self.bottleneck = Conv2dBlock(512, 1024)

        # DECODER CON ATTENTION GATES
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.ag4 = AttentionGate(F_g=512, F_l=512, F_int=256) 
        self.dec4 = Conv2dBlock(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.ag3 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.dec3 = Conv2dBlock(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.ag2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.dec2 = Conv2dBlock(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.ag1 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.dec1 = Conv2dBlock(128, 64)

        # OUTPUT
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # ENCODER
        e1 = self.enc1(x)
        p1 = self.pool1(self.dropout(e1))

        e2 = self.enc2(p1)
        p2 = self.pool2(self.dropout(e2))

        e3 = self.enc3(p2)
        p3 = self.pool3(self.dropout(e3))

        e4 = self.enc4(p3)
        p4 = self.pool4(self.dropout(e4))

        # BOTTLENECK
        b = self.bottleneck(p4)

        # DECODER CON ATTENZIONE (Skip Connections filtrate)
        d4_up = self.upconv4(b)
        e4_att = self.ag4(g=d4_up, x=e4) 
        d4 = torch.cat((d4_up, e4_att), dim=1) 
        d4 = self.dec4(d4)

        d3_up = self.upconv3(d4)
        e3_att = self.ag3(g=d3_up, x=e3)
        d3 = torch.cat((d3_up, e3_att), dim=1)
        d3 = self.dec3(d3)

        d2_up = self.upconv2(d3)
        e2_att = self.ag2(g=d2_up, x=e2)
        d2 = torch.cat((d2_up, e2_att), dim=1)
        d2 = self.dec2(d2)

        d1_up = self.upconv1(d2)
        e1_att = self.ag1(g=d1_up, x=e1)
        d1 = torch.cat((d1_up, e1_att), dim=1)
        d1 = self.dec1(d1)

        # OUTPUT
        out = self.final_conv(d1)
        out = self.sigmoid(out)
        
        return out