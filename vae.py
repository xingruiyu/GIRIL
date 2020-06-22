import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from models import cnn
from torchvision import utils
from state_process import process
from intrinsic_utils import save_samples_by_frame

class VAE(nn.Module):

    def __init__(self, device, model_base='cnn', conv_layers=32,
                 conv_kernel_size=3, latent_dim=1024, hidden_dim=64,
                 in_channels=8, out_channels=4, action_dim=4):
        super(VAE, self).__init__()

        if model_base == 'cnn':
            self.encoder = cnn.Encoder(conv_kernel_size=conv_kernel_size, conv_layers=conv_layers,
                                   in_channels=in_channels, 
                                   latent_dim=latent_dim, action_dim=action_dim, return_latent_layer=True)

            self.decoder = cnn.Decoder(conv_layers=conv_layers, conv_kernel_size=conv_kernel_size,
                                   latent_dim=latent_dim, 
                                   hidden=hidden_dim, out_channels=out_channels, action_dim=action_dim)

        #Latent variable mean and logvariance
        self.mu = nn.Linear(latent_dim, action_dim)
        self.logvar = nn.Linear(latent_dim, action_dim)

        self.device = device
        self.latent_dim = latent_dim
        self.action_dim = action_dim

    def reparameterize(self, mu, logvar, device, training=True):
        # Reparameterization trick as shown in the auto encoding variational bayes paper
        if training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_()).to(device)
            return eps.mul(std).add_(mu)
        else:
            return mu
 
    def forward(self, current_state, next_state):
        latent = self.encoder(current_state, next_state)
        mu = self.mu(latent)
        logvar = self.logvar(latent)
        
        z = self.reparameterize(mu, logvar, self.device)
        
        reconstructed_next_state = self.decoder(z, current_state)

        return z, mu, logvar, reconstructed_next_state

class GenerativeIntrinsicRewardModule(object):

    def __init__(self,
                 env,
                 device,
                 vae,
                 lr,
                 ):

        self.env = env
        self.device = device
        self.vae = vae
        self.lr = lr
        self.obs_shape = self.env.observation_space.shape
        if len(self.obs_shape)==3:
            self.action_dim = self.env.action_space.n
        if len(self.obs_shape)==1:
            self.action_dim = self.env.action_space.shape[0]

        self.optim = optim.Adam(lr=self.lr, params=self.vae.parameters())
   
    def get_vae_loss(self, recon_x, x, mean, log_var):
        RECON = F.mse_loss(recon_x, x)
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    
        return RECON, KLD 
    
    def fit_batch(self, state, action, next_state, train=True, lambda_recon=1.0, lambda_action=1.0, kld_loss_beta=1.0, lambda_gp=0.0):
        z, mu, logvar, reconstructed_next_state = self.vae(state, next_state)
        criterionAction = nn.CrossEntropyLoss()
        action_loss = criterionAction(z, action.long())

        recon_loss, kld_loss = self.get_vae_loss(reconstructed_next_state, next_state, mu, logvar)
        vae_loss = recon_loss + kld_loss_beta * kld_loss + lambda_action * action_loss 

        if train:
            self.optim.zero_grad()
            vae_loss.backward(retain_graph=True)
            self.optim.step()

        return vae_loss, recon_loss, kld_loss_beta*kld_loss, lambda_action*action_loss, z 

    # Calculation of the reward
    def calculate_intrinsic_reward(self, current_state, action, next_state, combine_action='False', lambda_true_action=1.0, reduction='mean'):
        with torch.no_grad():
            if len(action.shape)>1:
                action = action.squeeze(1)
            # true action
            true_action = F.one_hot(action.to(torch.int64), self.action_dim).float()
            # pred action
            latent = self.vae.encoder(current_state, next_state)
            mu = self.vae.mu(latent)
            logvar = self.vae.logvar(latent)
            z = self.vae.reparameterize(mu, logvar, self.vae.device)
            pred_action = F.softmax(z)

            if combine_action == 'True':
                action = lambda_true_action * true_action + (1-lambda_true_action) * pred_action
            else:
                action = true_action

            pred_next_state = self.vae.decoder(action, current_state)
            processed_next_state = process(next_state, normalize=True, range=(-1, 1))
            processed_pred_next_state = process(pred_next_state, normalize=True, range=(-1, 1))

            reward = F.mse_loss(processed_pred_next_state, processed_next_state, reduction=reduction)

        return reward, pred_next_state



