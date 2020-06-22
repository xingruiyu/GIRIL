import torch
from torch import nn

class Encoder(nn.Module):

    def __init__(self, conv_layers=32, conv_kernel_size=3,
                 in_channels=4, latent_dim=1024, action_dim=4, return_latent_layer=False):
        super(Encoder, self).__init__()
        self.conv_layers = conv_layers
        self.conv_kernel_size = conv_kernel_size
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.return_latent_layer = return_latent_layer

        # Encoder Architecture
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.conv_layers,
                               stride=2, kernel_size=self.conv_kernel_size)
        self.bn1 = nn.BatchNorm2d(self.conv_layers)
        self.conv2 = nn.Conv2d(in_channels=self.conv_layers, out_channels=self.conv_layers,
                               stride=2, kernel_size=self.conv_kernel_size)
        self.bn2 = nn.BatchNorm2d(self.conv_layers)
        self.conv3 = nn.Conv2d(in_channels=self.conv_layers, out_channels=self.conv_layers*2,
                               stride=2, kernel_size=self.conv_kernel_size)
        self.bn3 = nn.BatchNorm2d(self.conv_layers*2)
        self.conv4 = nn.Conv2d(in_channels=self.conv_layers*2, out_channels=self.conv_layers*2,
                               stride=2, kernel_size=self.conv_kernel_size)
        self.bn4 = nn.BatchNorm2d(self.conv_layers*2)
        self.linear = nn.Linear(in_features=self.latent_dim,
                                out_features=self.action_dim)

        # Leaky relu activation
        self.lrelu = nn.LeakyReLU()

        # Initialize the weights using xavier initialization
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.conv4.weight)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, state, next_state=None):
        if next_state is not None:
            x = torch.cat((state, next_state), dim=1)
        else:
            x = state
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lrelu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lrelu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.lrelu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.lrelu(x)

        x = x.view(x.size(0), -1)

        if not self.return_latent_layer:
            x = self.linear(x)

        return x 

class Decoder(nn.Module):

    def __init__(self, out_channels=4, conv_layers=32,
                 hidden=64, conv_kernel_size=3, latent_dim=1024, action_dim=4):
        super(Decoder, self).__init__()
        self.out_channels = out_channels
        self.conv_layers = conv_layers
        self.conv_kernel_size = conv_kernel_size
        self.in_dimension = action_dim
        self.hidden = hidden
        self.latent_dim = latent_dim

        # Decoder Architecture
        self.linear1_decoder = nn.Linear(in_features=self.in_dimension,
                                         out_features=self.hidden)
        self.bn_l_d = nn.BatchNorm1d(self.hidden)
        self.linear = nn.Linear(in_features=self.hidden,
                                out_features=self.latent_dim)
        self.bn_l_2_d = nn.BatchNorm1d(self.latent_dim)
        self.conv5 = nn.ConvTranspose2d(in_channels=self.conv_layers * 2, out_channels=self.conv_layers * 2,
                                        kernel_size=self.conv_kernel_size, stride=2)
        self.bn5 = nn.BatchNorm2d(self.conv_layers * 2)
        self.conv6 = nn.ConvTranspose2d(in_channels=self.conv_layers * 2, out_channels=self.conv_layers * 2,
                                        kernel_size=self.conv_kernel_size, stride=2)
        self.bn6 = nn.BatchNorm2d(self.conv_layers * 2)
        self.conv7 = nn.ConvTranspose2d(in_channels=self.conv_layers * 2, out_channels=self.conv_layers,
                                        kernel_size=self.conv_kernel_size, stride=2)
        self.bn7 = nn.BatchNorm2d(self.conv_layers)
        self.conv8 = nn.ConvTranspose2d(in_channels=self.conv_layers, out_channels=self.conv_layers,
                                        kernel_size=self.conv_kernel_size, stride=2)
        self.output = nn.ConvTranspose2d(in_channels=self.conv_layers, out_channels=self.out_channels,
                                         kernel_size=self.conv_kernel_size+3)
        self.conv9 = nn.Conv2d(in_channels=self.out_channels*2, out_channels=self.out_channels,
                                        kernel_size=self.conv_kernel_size, stride=1, dilation=2, padding=2)

        # Leaky relu activation
        self.lrelu = nn.LeakyReLU()

        # Initialize weights using xavier initialization
        nn.init.xavier_uniform_(self.conv5.weight)
        nn.init.xavier_uniform_(self.conv6.weight)
        nn.init.xavier_uniform_(self.conv7.weight)
        nn.init.xavier_uniform_(self.conv8.weight)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.linear1_decoder.weight)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.xavier_uniform_(self.conv9.weight)

    def forward(self, z, state):
        batch_size, _ = z.shape
        z = self.linear1_decoder(z)
        z = self.lrelu(z)
        z = self.linear(z)
        z = self.lrelu(z)
        z = z.view((batch_size, 64, 4, 4))

        z = self.conv5(z)
        z = self.lrelu(z)

        z = self.conv6(z)
        z = self.lrelu(z)

        z = self.conv7(z)
        z = self.lrelu(z)

        z = self.conv8(z)
        z = self.lrelu(z)

        z = self.output(z)

        z = torch.cat((z, state), dim=1)
        output = self.conv9(z)

        return output

