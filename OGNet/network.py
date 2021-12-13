
from torch import nn


class g_net(nn.Module):
    def __init__(self):
        super(g_net, self).__init__()
        self.encoder = nn.Sequential(

            nn.Conv2d(3, 64, 5, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 5, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 5, stride=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, 5, stride=1),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
        )
        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(512, 256, 5, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 5, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 5, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 5, stride=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class d_net(nn.Module):
    def __init__(self):
        super(d_net, self).__init__()
        self.discriminator = nn.Sequential(

            nn.Conv2d(3, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 5, stride=2, padding=2),
            nn.ReLU(True),
            Flatten(),
            #nn.Linear(4608, 1),
            nn.Linear(32768,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.discriminator(x)
        return x
