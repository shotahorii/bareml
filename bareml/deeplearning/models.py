import .functions as F
import .layers as L


class MLP(L.Module):
    def __init__(self, fc_output_sizes, activation=F.sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(fc_output_sizes):
            layer = L.Linear(out_size)
            setattr(self, 'l'+str(i), layer)
            self.layers.append(layer)
    
    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)


class VGG16(L.Module):
    """ VGG16 net """
    def __init__(self, in_channels=3, num_classes=1000):
        super().__init__()
        features = L.Sequential(
            L.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            L.ReLU(),
            L.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            L.ReLU(),
            L.MaxPool2d(kernel_size=2, stride=2, padding=0),
            L.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            L.ReLU(),
            L.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            L.ReLU(),
            L.MaxPool2d(kernel_size=2, stride=2, padding=0),
            L.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            L.ReLU(),
            L.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            L.ReLU(),
            L.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            L.ReLU(),
            L.MaxPool2d(kernel_size=2, stride=2, padding=0),
            L.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            L.ReLU(),
            L.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            L.ReLU(),
            L.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            L.ReLU(),
            L.MaxPool2d(kernel_size=2, stride=2, padding=0),
            L.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            L.ReLU(),
            L.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            L.ReLU(),
            L.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            L.ReLU(),
            L.MaxPool2d(kernel_size=2, stride=2, padding=0)
            )

        classifier = L.Sequential(
            L.Linear(in_features=25088, out_features=4096, bias=True),
            L.ReLU(),
            L.Dropout(p=0.5),
            L.Linear(in_features=4096, out_features=4096, bias=True),
            L.ReLU(),
            L.Dropout(p=0.5),
            L.Linear(in_features=4096, out_features=num_classes, bias=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.shape[0],-1)
        x = self.classifier(x)
        return x
        
