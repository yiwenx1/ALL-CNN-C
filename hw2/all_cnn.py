import torch.nn as nn
from torch.nn import Sequential


class Flatten(nn.Module):
    """
    Implement a simple custom module that reshapes (n, m, 1, 1) tensors to (n, m).
    """
    def forward(self, x):
        n = x.size()[0]
        m = x.size()[1]
        return x.view(n, m)


def all_cnn_module():
    """
    Create a nn.Sequential model containing all of the layers of the All-CNN-C as specified in the paper.
    https://arxiv.org/pdf/1412.6806.pdf
    Use a AvgPool2d to pool and then your Flatten layer as your final layers.
    You should have a total of exactly 23 layers of types:
    - nn.Dropout
    - nn.Conv2d
    - nn.ReLU
    - nn.AvgPool2d
    - Flatten
    :return: a nn.Sequential model
    """
    conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, padding=1)
    # nn.init.xavier_uniform_(conv1.weight.data)
    conv2 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1)
    # nn.init.xavier_uniform_(conv2.weight.data)
    conv3 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=2, padding=1)
    # nn.init.xavier_uniform_(conv3.weight.data)

    conv4 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=3, padding=1)
    # nn.init.xavier_uniform_(conv4.weight.data)
    conv5 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1)
    # nn.init.xavier_uniform_(conv5.weight.data)
    conv6 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=2, padding=1)
    # nn.init.xavier_uniform_(conv6.weight.data)

    conv7 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3)
    # nn.init.xavier_uniform_(conv7.weight.data)
    conv8 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1)
    # nn.init.xavier_uniform_(conv8.weight.data)
    conv9 = nn.Conv2d(in_channels=192, out_channels=10, kernel_size=1)
    # nn.init.xavier_uniform_(conv9.weight.data)

    model = Sequential(
        nn.Dropout(p=0.2),
        conv1,
        nn.ReLU(),
        conv2,
        nn.ReLU(),
        conv3,
        nn.ReLU(),
        nn.Dropout(), # default p = 0.5
        conv4,
        nn.ReLU(),
        conv5,
        nn.ReLU(),
        conv6,
        nn.ReLU(),
        nn.Dropout(), # default p = 0.5
        conv7,
        nn.ReLU(),
        conv8,
        nn.ReLU(),
        conv9,
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=6),
        Flatten()
    )
    for m in model:
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)

    return model
