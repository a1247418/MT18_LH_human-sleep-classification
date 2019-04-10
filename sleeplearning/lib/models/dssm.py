import copy
import os
import sys
import math
import gridfs

from sleeplearning.lib.models.single_chan_expert import SingleChanExpert

root_dir = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
sys.path.insert(0, root_dir)
from pymongo import MongoClient
from torch import nn
from torch.nn.init import xavier_normal_ as xavier_normal
from cfg.config import mongo_url
import torch
import torch.nn.functional as F
import sleeplearning.lib.base


def list_to_tensor(list):
    tensor = list[0].unsqueeze(1)
    for idx, t in enumerate(list):
        if idx==0:
            continue
        else:
            tensor = torch.cat((tensor, t.unsqueeze(1)), dim=1)
    return tensor


# generate input sample and forward to get shape
def _get_output_dim(net, shape):
    bs = 1
    input = torch.rand(bs, *shape)
    output_feat = net(input)
    #n_size = output_feat.data.view(bs, -1).size(1)
    #return n_size
    return output_feat.shape


class Conv2dWithBn(nn.Module):
    def __init__(self, input_shape, filter_size, n_filters, stride, xavier=False):
        super(Conv2dWithBn, self).__init__()
        self.conv1 = nn.Conv2d(input_shape, n_filters, kernel_size=filter_size,
                               stride=stride, bias=False,
                               padding=((filter_size[0] - 1) // 2, (filter_size[
                                                                        1] - 1) // 2))
        # fake 'SAME'
        self.relu = nn.ReLU()
        self.conv1_bn = nn.BatchNorm2d(n_filters)
        if xavier:
            self.weights_init()

    def weights_init(m):
        for _, mi in m._modules.items():
            if isinstance(mi, nn.Conv2d) or isinstance(m, nn.Linear):
                xavier_normal(mi.weight.data)
                if mi.bias is not None:
                    xavier_normal(mi.bias.data)

    def forward(self, x):
        #print(x.shape, "-->")
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv1_bn(x)
        #print(x.shape)
        return x


class ConvBlock(nn.Module):
    def __init__(self, input_shape, filter_dim):
        super(ConvBlock, self).__init__()
        n_signals = int(input_shape[0])
        self.block = nn.Sequential(
                nn.MaxPool2d((2, 2), stride=(2, 2)),
                Conv2dWithBn(n_signals, filter_size=(3, 3), n_filters=filter_dim, stride=1),
                nn.MaxPool2d((2, 2), stride=(2, 2)),

                Conv2dWithBn(filter_dim, filter_size=(3, 3), n_filters=filter_dim, stride=1),
                nn.MaxPool2d((3, 3), stride=(3, 3)),

                Conv2dWithBn(filter_dim, filter_size=(3, 3), n_filters=filter_dim, stride=1),
                nn.MaxPool2d((2, 4), stride=(2, 4)),
            )

        outdim = _get_output_dim(self.block, input_shape)

    def forward(self, x):
        x = self.block(x)
        return x#.view(x.size(0), -1)


class DSSM(nn.Module):
    def __init__(self, ms: dict):
        super(DSSM, self).__init__()

        self.kl_weight = 0

        self.input_dim = ms['input_dim']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_size = ms["hidden_size"]
        self.filter_size = ms["filter_size"]
        self.out_size = ms['nclasses']
        self.theta_size = ms["theta_size"]
        self.use_theta = ms["use_theta"]
        self.n_samples = ms["nbrs"] + 1
        self.dropout = ms["dropout"]

        self.n_signals = int(self.input_dim[0])
        self.observation_dim = self.input_dim[1]
        self.subseq_len = int(self.input_dim[2] // self.n_samples)

        kernel_w = self.subseq_len
        stride_w = self.subseq_len
        kernel_h = self.observation_dim
        stride_h = self.observation_dim
        self.reduced_observ_dim = self.hidden_size#math.floor((self.observation_dim - (kernel_w-1)-1)/stride_w + 1)

        # convolver
        """
        self.convolver_phase_1 = nn.Sequential(nn.Conv2d(self.n_signals, self.hidden_size,
                                                         kernel_size=[kernel_h, kernel_w],
                                                         stride=[stride_h, stride_w]),
                                               nn.BatchNorm2d(self.hidden_size),
                                               nn.ReLU())
        self.convolver_phase_2 = nn.Sequential(nn.Conv2d(self.n_signals, self.hidden_size,
                                                         kernel_size=[kernel_h, kernel_w],
                                                         stride=[stride_h, stride_w]),
                                               nn.BatchNorm2d(self.hidden_size),
                                               nn.ReLU())
        """
        self.convolver_phase_1 = nn.Sequential(ConvBlock(self.input_dim, self.filter_size), nn.Dropout(p=self.dropout))
        self.convolver_phase_2 = nn.Sequential(ConvBlock(self.input_dim, self.filter_size), nn.Dropout(p=self.dropout))

        out_size = _get_output_dim(self.convolver_phase_1, self.input_dim)
        #print(out_size)

        # embedding module (for calculating beta)
        self.uncertainty_inference_1 = nn.Linear(self.filter_size+self.hidden_size, self.hidden_size)
        self.uncertainty_inference_2_mean = nn.Linear(self.hidden_size, self.hidden_size)
        self.uncertainty_inference_2_logvar = nn.Linear(self.hidden_size, self.hidden_size)

        # system identification module for functions phi_theta and phi_s
        self.num_layers = 2
        self.sequence_encoder = nn.LSTM(input_size=self.filter_size, hidden_size=self.hidden_size,
                                        num_layers=self.num_layers, bidirectional=True, batch_first=True)
        self.theta_encoder = nn.Linear(self.hidden_size, self.theta_size)
        self.state_encoder = nn.Linear(self.hidden_size, 2*self.hidden_size)

        # transition module for function f
        self.state_transition = nn.LSTMCell(input_size=self.theta_size, hidden_size=self.hidden_size)

        # noise transform part
        # see: https://stats.stackexchange.com/questions/16334/
        # how-to-sample-from-a-normal-distribution-with-known-mean-and-variance-using-a-co
        self.noise_scalar = nn.Parameter(torch.rand(1))
        self.noise_vector = nn.Parameter(torch.ones(1))
        self.state_bias = nn.Parameter(torch.rand(1))

        # emission module for function g
        self.state_emission_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.state_emission_2 = nn.Linear(self.hidden_size, self.filter_size)

        # deconvolver
        self.deconvolver = nn.Sequential(nn.ReLU(),
                                         nn.ConvTranspose2d(self.filter_size,
                                                            self.filter_size,
                                                            kernel_size=[kernel_h//2, kernel_w//2],
                                                            stride=[stride_w//2, stride_h//2]),
                                         nn.ReLU(),
                                         nn.ConvTranspose2d(self.filter_size,
                                                            self.n_signals,
                                                            kernel_size=[2, 2],
                                                            stride=[2, 2]))

        # labler
        linear_stack = [self.hidden_size, self.hidden_size]
        linear_module_list = []
        for ls_id in range(len(linear_stack)-1):
            linear_module_list.append(nn.Linear(linear_stack[ls_id], linear_stack[ls_id+1]))
            linear_module_list.append(nn.BatchNorm1d(linear_stack[ls_id+1]))
            linear_module_list.append(nn.ReLU())

        self.label_nn = nn.Sequential(*linear_module_list, nn.Linear(linear_stack[-1], self.out_size))

        # reset parameters
        self.reset_parameters()

    def reset_parameters(self, stdv=1e-2):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)

    def update_KL_weight(self):
        self.kl_weight = min(1, self.kl_weight + 0.0001)

    def PTKLinear(self, X, Y):
        return X.mm(Y.t())

    def train(self, mode=True):
        super(DSSM, self).train(mode=mode)

    def forward(self, batch, theta=None, filtering_mode=False, forecast_seq_len=0):
        #batch shape = [32, 4, 76, 1050] = [bs, n_channels, spectogram_dim, 50+(1+n_neighbours)]

        # print("Batch shape:", batch.shape)
        # (bs, n_signals, sub_seq_len(20sec), seq_len) -> (bs, hidden_size, 1, seq_len)
        cbatch = self.convolver_phase_1(batch)
        # print("Convolved shape: ", cbatch.shape)
        cbatch = torch.transpose(cbatch[:, :, 0, :], 1, 2)  # bs x seq_len x hidden_sz
        # print("Transformed shape: ", cbatch.shape)

        # reconstructed sequence
        obs_rc_sequence = [batch[:, :, :, 0:self.subseq_len]]
        # print("Obs_1 shape: ", obs_rc_sequence[0].shape)

        # 1. System identification part (Encoder) --> infer the context "theta" and the starting state "s_0"
        # --------------------------------------------------------------------------------------------------
        output, (h_enc, c_enc) = self.sequence_encoder(cbatch)
        first_layer_back_pass = output[:, 0, self.hidden_size:]
        initial_state = self.state_encoder(first_layer_back_pass)
        h_dec_prev = initial_state[:, :self.hidden_size]
        c_dec_prev = initial_state[:, self.hidden_size:]
        if self.use_theta:
            if theta is None:
                theta = self.theta_encoder(first_layer_back_pass)
            else:
                theta = torch.Tensor(theta).double().unsqueeze(0).to(self.device)
        else:
            theta = torch.zeros([batch.shape[0], self.theta_size]).float().to(self.device)

        # MMD regularization term
        mmd_loss = 0
        # TODO: Uncomment, when theta plays a role
        #for idx in range(seq_len - 1):
        #    mean_KFX = torch.mean(self.PTKLinear(output[:, idx, :], output[:, idx, :]))
        #    mean_KF_gz = torch.mean(self.PTKLinear(output[:, idx + 1, :], output[:, idx + 1, :]))
        #    mmd_loss += mean_KF_gz + mean_KFX - 2.0 * torch.mean(self.PTKLinear(output[:, idx + 1, :], output[:, idx, :]))


        # 2. Sequence prediction (Decoder)
        # ---------------------------------------
        rec_loss = mmd_loss # TODO: Watch out here....
        #rec_loss = 0
        kl_loss = 0
        y_hat = []

        cbatch = self.convolver_phase_2(
            batch)  # (bs, n_signals, sub_seq_len(20sec), seq_len) -> (bs, reduced_obs, 1, seq_len)
        cbatch = torch.transpose(cbatch[:, :, 0, :], 1, 2)  # bs x seq_len x reduced_obs
        # print("cbatch:", cbatch.shape)
        for i in range(self.n_samples):
            # predict next state TODO: Notice that it goes before beta!!!!!!!!!!!!!!!!!!!!!!!!
            h_dec, c_dec = self.state_transition(theta, (h_dec_prev, c_dec_prev)) #  bs x hidden
            c_dec = c_dec + c_dec_prev # Only predict the difference to the prev state
            c_dec_prev = c_dec
            h_dec_prev = h_dec
            # print("State shape: ", c_dec.shape)

            # get next observation
            obs = batch[:, :, :, i*self.subseq_len:(i+1)*self.subseq_len] #  bs x n_signals x x subseq_len # dim_obs
            # print("obs shape:", obs.shape)
            cobs = cbatch[:, i, :]#  bx x reduced_dim_sig

            # get probabilistic estimate of innovation variable: beta_i = f(s_{i-1}, x_i)
            beta_embedding = F.relu(self.uncertainty_inference_1(torch.cat((c_dec, cobs), dim=1)))
            beta_mean = self.uncertainty_inference_2_mean(beta_embedding)
            beta_logvar = self.uncertainty_inference_2_logvar(beta_embedding)
            beta = self.reparameterize(beta_mean, beta_logvar)

            # Uncertainty scaling -- for the moment we have three options
            # Option 1: do not scale uncertainty
            # Option 2: scale entire uncertainty vector with one learnable scalar
            beta = beta * self.noise_scalar.expand_as(beta)
            # Option 3: scale uncertainty vector with a learnable vector
            # beta = beta * self.noise_vector

            # update state with innovation vector if not in forecast mode
            # (c_dec is the a priori stat predicion. here it gets updated with beta to form the posterior)
            if filtering_mode:
                c_dec = c_dec + beta_mean * self.noise_scalar.expand_as(beta) #+ self.state_bias
            else:
                c_dec = c_dec + beta #+ self.state_bias

            # predict next observation
            pred = F.relu(self.state_emission_1(c_dec))
            pred = self.state_emission_2(pred)
            #print("Emission shape: ", pred.shape)
            pred = pred[:, :, None, None]
            # print("TEmission shape: ", pred.shape)
            pred = self.deconvolver(pred)  # 20, 1, 39=observ -> 20, 4, subseq_len
            # print("DCEmission shape: ", pred.shape)

            # append predicted observation
            obs_rc_sequence.append(pred)

            # append predicted label
            # print("cdec shape", c_dec.shape)
            pred_lab = self.label_nn(c_dec)
            # print("Prediction shape", pred_lab.shape)
            y_hat.append(pred_lab)
            # print("Len y_hat", len(y_hat))

            # calculate reconstruction loss and kl loss
            # print("Obs shape: ", obs.shape)
            # print("Pred shape: ", pred.shape)
            rec_loss += F.mse_loss(pred.contiguous().view(-1), obs.contiguous().view(-1))
            kl_loss += self.kl_loss(beta_mean, beta_logvar)
            #print(kl_loss)

        if forecast_seq_len > 0:
            with torch.no_grad():
                self.eval()
                forecast = self.forecast(h_dec, c_dec, theta=theta, steps=forecast_seq_len)
        else:
            forecast = None

        return {"rec_loss": rec_loss,
         "kl_loss": kl_loss * self.kl_weight,
         "reconstructions": list_to_tensor(obs_rc_sequence),
         "forecast": forecast,
         "theta": theta,
         "logits": list_to_tensor(y_hat) #y_hat[-1]
                }

    def forecast(self, h, c, theta, steps):
        x = []
        for i in range(steps):
            # state transition
            h, c = self.state_transition(theta, (h, c))
            # predict next observation
            pred = F.relu(self.state_emission_1(c))
            pred = self.state_emission_2(pred)
            #print("Emission shape: ", pred.shape)
            pred = pred[:, :, None, None]
            #print("TEmission shape: ", pred.shape)
            pred = self.deconvolver(pred)  # 20, 1, 39=observ -> 20, 4, subseq_len
            # append it to array
            x.append(pred)
        return list_to_tensor(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def kl_loss(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
