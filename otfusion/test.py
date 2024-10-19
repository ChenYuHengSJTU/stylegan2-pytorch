# %%
import torch
import torch.nn as nn
import ot
import numpy as np
import torch.nn.functional as F

# %%
def isnan(x):
    return x != x

# %%
class GroundMetric:
    """
        Ground Metric object for Wasserstein computations:

    """

    def __init__(self, params, not_squared = False):
        self.params = params
        self.ground_metric_type = params.ground_metric
        self.ground_metric_normalize = params.ground_metric_normalize
        self.reg = params.reg
        if hasattr(params, 'not_squared'):
            self.squared = not params.not_squared
        else:
            # so by default squared will be on!
            self.squared = not not_squared
        self.mem_eff = params.ground_metric_eff

    def _clip(self, ground_metric_matrix):
        if self.params.debug:
            print("before clipping", ground_metric_matrix.data)

        percent_clipped = (float((ground_metric_matrix >= self.reg * self.params.clip_max).long().sum().data) \
                           / ground_metric_matrix.numel()) * 100
        print("percent_clipped is (assumes clip_min = 0) ", percent_clipped)
        setattr(self.params, 'percent_clipped', percent_clipped)
        # will keep the M' = M/reg in range clip_min and clip_max
        ground_metric_matrix.clamp_(min=self.reg * self.params.clip_min,
                                             max=self.reg * self.params.clip_max)
        if self.params.debug:
            print("after clipping", ground_metric_matrix.data)
        return ground_metric_matrix

    def _normalize(self, ground_metric_matrix):

        if self.ground_metric_normalize == "log":
            ground_metric_matrix = torch.log1p(ground_metric_matrix)
        elif self.ground_metric_normalize == "max":
            print("Normalizing by max of ground metric and which is ", ground_metric_matrix.max())
            ground_metric_matrix = ground_metric_matrix / ground_metric_matrix.max()
        elif self.ground_metric_normalize == "median":
            print("Normalizing by median of ground metric and which is ", ground_metric_matrix.median())
            ground_metric_matrix = ground_metric_matrix / ground_metric_matrix.median()
        elif self.ground_metric_normalize == "mean":
            print("Normalizing by mean of ground metric and which is ", ground_metric_matrix.mean())
            ground_metric_matrix = ground_metric_matrix / ground_metric_matrix.mean()
        elif self.ground_metric_normalize == "none":
            return ground_metric_matrix
        else:
            raise NotImplementedError

        return ground_metric_matrix

    def _sanity_check(self, ground_metric_matrix):
        assert not (ground_metric_matrix < 0).any()
        assert not (isnan(ground_metric_matrix).any())

    def _cost_matrix_xy(self, x, y, p=2, squared = True):
        # TODO: Use this to guarantee reproducibility of previous results and then move onto better way
        "Returns the matrix of $|x_i-y_j|^p$."
        # (n_l,1,n_{l-1})
        x_col = x.unsqueeze(1)
        # (1, m_l, m_{l-1})
        # math induction -> n_{l-1} == m_{l-1}
        y_lin = y.unsqueeze(0)
        # c => (n_l, m_l, m_{l-1})
        c = torch.sum((torch.abs(x_col - y_lin)) ** p, 2)
        if not squared:
            print("dont leave off the squaring of the ground metric")
            c = c ** (1/2)
        # print(c.size())
        if self.params.dist_normalize:
            assert NotImplementedError
        return c


    def _pairwise_distances(self, x, y=None, squared=True):
        '''
        Source: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
        Input: x is a Nxd matrix
               y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        '''
        x_norm = (x ** 2).sum(1).view(-1, 1)
        if y is not None:
            y_t = torch.transpose(y, 0, 1)
            y_norm = (y ** 2).sum(1).view(1, -1)
        else:
            y_t = torch.transpose(x, 0, 1)
            y_norm = x_norm.view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        # Ensure diagonal is zero if x=y
        dist = torch.clamp(dist, min=0.0)

        if self.params.activation_histograms and self.params.dist_normalize:
            dist = dist/self.params.act_num_samples
            print("Divide squared distances by the num samples")

        if not squared:
            print("dont leave off the squaring of the ground metric")
            dist = dist ** (1/2)

        return dist

    def _get_euclidean(self, coordinates, other_coordinates=None):
        # TODO: Replace by torch.pdist (which is said to be much more memory efficient)

        if other_coordinates is None:
            matrix = torch.norm(
                coordinates.view(coordinates.shape[0], 1, coordinates.shape[1]) \
                - coordinates, p=2, dim=2
            )
        else:
            # memory efficient version
            if self.mem_eff:
                matrix = self._pairwise_distances(coordinates, other_coordinates, squared=self.squared)
            else:
                matrix = self._cost_matrix_xy(coordinates, other_coordinates, squared = self.squared)

        return matrix

    def _normed_vecs(self, vecs, eps=1e-9):
        norms = torch.norm(vecs, dim=-1, keepdim=True)
        print("stats of vecs are: mean {}, min {}, max {}, std {}".format(
            norms.mean(), norms.min(), norms.max(), norms.std()
        ))
        return vecs / (norms + eps)

    def _get_cosine(self, coordinates, other_coordinates=None):
        if other_coordinates is None:
            matrix = coordinates / torch.norm(coordinates, dim=1, keepdim=True)
            matrix = 1 - matrix @ matrix.t()
        else:
            matrix = 1 - torch.div(
                coordinates @ other_coordinates.t(),
                torch.norm(coordinates, dim=1).view(-1, 1) @ torch.norm(other_coordinates, dim=1).view(1, -1)
            )
        return matrix.clamp_(min=0)

    def _get_angular(self, coordinates, other_coordinates=None):
        pass

    def get_metric(self, coordinates, other_coordinates=None):
        get_metric_map = {
            'euclidean': self._get_euclidean,
            'cosine': self._get_cosine,
            'angular': self._get_angular,
        }
        return get_metric_map[self.ground_metric_type](coordinates, other_coordinates)

    def process(self, coordinates, other_coordinates=None):
        print(f'Processing the coordinates to form ground_metric with shape {coordinates.shape} and {other_coordinates.shape}')
        if self.params.geom_ensemble_type == 'wts' and self.params.normalize_wts:
            print("In weight mode: normalizing weights to unit norm")
            coordinates = self._normed_vecs(coordinates)
            if other_coordinates is not None:
                other_coordinates = self._normed_vecs(other_coordinates)

        ground_metric_matrix = self.get_metric(coordinates, other_coordinates)

        if self.params.debug:
            print("coordinates is ", coordinates)
            if other_coordinates is not None:
                print("other_coordinates is ", other_coordinates)
            print("ground_metric_matrix is ", ground_metric_matrix)

        self._sanity_check(ground_metric_matrix)

        ground_metric_matrix = self._normalize(ground_metric_matrix)

        self._sanity_check(ground_metric_matrix)

        if self.params.clip_gm:
            ground_metric_matrix = self._clip(ground_metric_matrix)

        self._sanity_check(ground_metric_matrix)

        if self.params.debug:
            print("ground_metric_matrix at the end is ", ground_metric_matrix)

        return ground_metric_matrix


# %%
def cost_matrix(x, y, p=2):
    "Returns the matrix of $|x_i-y_j|^p$."
    x_col = x.unsqueeze(1)
    y_lin = y.unsqueeze(0)
    c = torch.sum((torch.abs(x_col - y_lin)) ** p, 2)
    return c

def get_histogram(args, idx, cardinality, layer_name, activations=None, return_numpy = True, float64=False):
    if activations is None:
        # returns a uniform measure
        if not args.unbalanced:
            print("returns a uniform measure of cardinality: ", cardinality)
            return np.ones(cardinality)/cardinality
        else:
            return np.ones(cardinality)
    else:
        # return softmax over the activations raised to a temperature
        # layer_name is like 'fc1.weight', while activations only contains 'fc1'
        print(activations[idx].keys())
        unnormalized_weights = activations[idx][layer_name.split('.')[0]]
        print("For layer {},  shape of unnormalized weights is ".format(layer_name), unnormalized_weights.shape)
        unnormalized_weights = unnormalized_weights.squeeze()
        assert unnormalized_weights.shape[0] == cardinality

        if return_numpy:
            if float64:
                return torch.softmax(unnormalized_weights / args.softmax_temperature, dim=0).data.cpu().numpy().astype(
                    np.float64)
            else:
                return torch.softmax(unnormalized_weights / args.softmax_temperature, dim=0).data.cpu().numpy()
        else:
            return torch.softmax(unnormalized_weights / args.softmax_temperature, dim=0)

# %%
def get_wassersteinized_layers_modularized(args, networks, activations=None, eps=1e-7, test_loader=None):
    '''
    Two neural networks that have to be averaged in geometric manner (i.e. layerwise).

    The 1st network is aligned with respect to the other via wasserstein distance.
    Also this assumes that all the layers are either fully connected or convolutional *(with no bias)*

    :param networks: list of networks
    :param activations: If not None, use it to build the activation histograms.
    Otherwise assumes uniform distribution over neurons in a layer.
    :return: list of layer weights 'wassersteinized'
    '''

    # simple_model_0, simple_model_1 = networks[0], networks[1]
    # simple_model_0 = get_trained_model(0, model='simplenet')
    # simple_model_1 = get_trained_model(1, model='simplenet')

    avg_aligned_layers = []
    # cumulative_T_var = None
    T_var = None
    # print(list(networks[0].parameters()))
    previous_layer_shape = None
    
    # for deconv layer judgement
    previous_layer_type = None
    
    ground_metric_object = GroundMetric(args)

    if args.eval_aligned:
        model0_aligned_layers = []

    if args.gpu_id==-1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args.gpu_id))


    num_layers = len(list(zip(networks[0].parameters(), networks[1].parameters())))
    # named_parameters is an generator
    # for linear layer with bias, the affine matrix and bias will be two separate terms in named_parameters
    is_conv = True
    fused_state_dict = dict()
    for idx, ((layer0_name, fc_layer0_weight), (layer1_name, fc_layer1_weight)) in \
            enumerate(zip(networks[0].named_parameters(), networks[1].named_parameters())):
        print("layer0:", layer0_name, "layer1:", layer1_name, sep=' ')

        # if 'decoder' in layer0_name or 'final' in layer0_name:
        #     avg_aligned_layers.append(fc_layer0_weight)
        #     fused_state_dict[layer0_name] = fc_layer0_weight
        #     continue

        assert fc_layer0_weight.shape == fc_layer1_weight.shape
        print("Previous layer shape is ", previous_layer_shape)
        # previous_layer_shape = fc_layer1_weight.shape

        mu_cardinality = fc_layer0_weight.shape[0]
        nu_cardinality = fc_layer1_weight.shape[0]

        # mu = np.ones(fc_layer0_weight.shape[0])/fc_layer0_weight.shape[0]
        # nu = np.ones(fc_layer1_weight.shape[0])/fc_layer1_weight.shape[0]

        layer_shape = fc_layer0_weight.shape
        if len(layer_shape) > 2:
            # whether it is a deconv
            # For convolutional layers, it is (#out_channels, #in_channels, height, width)
            fc_layer0_weight_data = fc_layer0_weight.data.view(fc_layer0_weight.shape[0], fc_layer0_weight.shape[1], -1)
            fc_layer1_weight_data = fc_layer1_weight.data.view(fc_layer1_weight.shape[0], fc_layer1_weight.shape[1], -1)
            
            if is_conv is False and idx != 0:
                print("Switching from linear layer to convtranspose layer!")
                # could not use the following code to match the dimension of the two layers
                # fc_layer0_weight_data = fc_layer0_weight.data.view(fc_layer0_weight.shape[0], T_var, -1).permute(2, 0, 1)
                # fc_layer1_weight_data = fc_layer1_weight.data.view(fc_layer1_weight.shape[0], T_var, -1).permute(2, 0, 1)
                
                # the only way is to reshape the matrix T
                # note that in pytorch, the shape for conv is (out, int, h, w), but for deconv is (in, out, h, w)
                T_var = T_var.view(fc_layer0_weight.shape[0], fc_layer1_weight.shape[0], -1).mean(dim=-1)
                fc_layer0_weight_data = fc_layer0_weight_data.permute(1, 0, 2)
                fc_layer1_weight_data = fc_layer1_weight_data.permute(1, 0, 2)
                mu_cardinality = fc_layer0_weight_data.shape[0]
                nu_cardinality = fc_layer1_weight_data.shape[0]

            if is_conv and idx != 0 and fc_layer0_weight_data.shape[1] != T_var.shape[0]:
                print("It is a convtranspose layer!")
                fc_layer0_weight_data = fc_layer0_weight_data.permute(1, 0, 2)
                fc_layer1_weight_data = fc_layer1_weight_data.permute(1, 0, 2)
                mu_cardinality = fc_layer0_weight_data.shape[0]
                nu_cardinality = fc_layer1_weight_data.shape[0]

            is_conv = True
        elif len(layer_shape) == 1:
            print(f"It is a bias term with shape {layer_shape}\n")
            continue
        else:
            is_conv = False
            fc_layer0_weight_data = fc_layer0_weight.data
            fc_layer1_weight_data = fc_layer1_weight.data

        previous_layer_shape = fc_layer1_weight.shape

        # the reason for approach idx==0 is that T is none when idx==0
        if idx == 0:
            print("Idx is 0!")            
            print("shape of layer: model 0", fc_layer0_weight_data.shape)
            print("shape of layer: model 1", fc_layer1_weight_data.shape)
            if is_conv:
                print(f"Layer {idx} is conv")
                M = ground_metric_object.process(fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1),
                                fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1))
                # M = cost_matrix(fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1),
                #                 fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1))
            else:
                # print("layer data is ", fc_layer0_weight_data, fc_layer1_weight_data)
                M = ground_metric_object.process(fc_layer0_weight_data, fc_layer1_weight_data)
                # M = cost_matrix(fc_layer0_weight, fc_layer1_weight)

            aligned_wt = fc_layer0_weight_data
        else:

            print("shape of layer: model 0", fc_layer0_weight_data.shape)
            print("shape of layer: model 1", fc_layer1_weight_data.shape)
            print("shape of previous transport map", T_var.shape)

            # aligned_wt = None, this caches the tensor and causes OOM
            if is_conv:
                # for conv layers, OT matches the channels
                # this code considers that for the ith layer of different models, their types is the same, conv or fc (no bias)
                # T_var -> (out_{l-1}, out_{l-1})
                # w0 -> out, in, hw
                # T_var_conv -> hw, out_{l-1}, out_{l-1}
                # out_{l}, out_{l-1}==in, hw -> aligned_wt
                T_var_conv = T_var.unsqueeze(0).repeat(fc_layer0_weight_data.shape[2], 1, 1)
                aligned_wt = torch.bmm(fc_layer0_weight_data.permute(2, 0, 1), T_var_conv).permute(1, 2, 0)

                # fc_layer1_weight_data = fc_layer1_weight_data.reshape(fc_layer1_weight_data.shape[0], -1)
                M = ground_metric_object.process(
                    aligned_wt.contiguous().view(aligned_wt.shape[0], -1),
                    fc_layer1_weight_data.contiguous().view(fc_layer1_weight_data.shape[0], -1)
                )
            else:
                # mismatching reason: 1. reshape the tensor
                # 2. connection between two different kinds of layers, such as linear or conv
                # input_dim != n_(l-1)
                if fc_layer0_weight.data.shape[1] != T_var.shape[0]:
                    # Handles the switch from convolutional layers to fc layers
                    # n_l,n_(l-1), other
                    fc_layer0_unflattened = fc_layer0_weight.data.view(fc_layer0_weight.shape[0], T_var.shape[0], -1).permute(2, 0, 1)
                    aligned_wt = torch.bmm(
                        fc_layer0_unflattened,
                        T_var.unsqueeze(0).repeat(fc_layer0_unflattened.shape[0], 1, 1)
                    ).permute(1, 2, 0)
                    aligned_wt = aligned_wt.contiguous().view(aligned_wt.shape[0], -1)
                else:
                    # print("layer data (aligned) is ", aligned_wt, fc_layer1_weight_data)
                    aligned_wt = torch.matmul(fc_layer0_weight.data, T_var)
                # M = cost_matrix(aligned_wt, fc_layer1_weight)
                M = ground_metric_object.process(aligned_wt, fc_layer1_weight)
            #     print("ground metric is ", M)
            # if args.skip_last_layer and idx == (num_layers - 1):
            #     print("Simple averaging of last layer weights. NO transport map needs to be computed")
            #     if args.ensemble_step != 0.5:
            #         avg_aligned_layers.append((1 - args.ensemble_step) * aligned_wt +
            #                               args.ensemble_step * fc_layer1_weight)
            #     else:
            #         avg_aligned_layers.append((aligned_wt + fc_layer1_weight)/2)
            #     return avg_aligned_layers

        if args.importance is None or (idx == num_layers -1):
            mu = get_histogram(args, 0, mu_cardinality, layer0_name)
            nu = get_histogram(args, 1, nu_cardinality, layer1_name)
        # else:
        #     # mu = _get_neuron_importance_histogram(args, aligned_wt, is_conv)
        #     mu = _get_neuron_importance_histogram(args, fc_layer0_weight_data, is_conv)
        #     nu = _get_neuron_importance_histogram(args, fc_layer1_weight_data, is_conv)
        #     print(mu, nu)
        #     assert args.proper_marginals

        print("Distance M shape", M.shape)

        cpuM = M.data.cpu().numpy()
        if args.exact:
            T = ot.emd(mu, nu, cpuM)
        else:
            T = ot.bregman.sinkhorn(mu, nu, cpuM, reg=args.reg)
        # T = ot.emd(mu, nu, log_cpuM)

        if args.gpu_id!=-1:
            T_var = torch.from_numpy(T).cuda(args.gpu_id).float()
        else:
            T_var = torch.from_numpy(T).float()

        # torch.set_printoptions(profile="full")
        print("the transport map is ", T_var)
        # torch.set_printoptions(profile="default")

        if args.correction:
            if not args.proper_marginals:
                # think of it as m x 1, scaling weights for m linear combinations of points in X
                if args.gpu_id != -1:
                    # marginals = torch.mv(T_var.t(), torch.ones(T_var.shape[0]).cuda(args.gpu_id))  # T.t().shape[1] = T.shape[0]
                    marginals = torch.ones(T_var.shape[0]).cuda(args.gpu_id) / T_var.shape[0]
                else:
                    # marginals = torch.mv(T_var.t(),
                    #                      torch.ones(T_var.shape[0]))  # T.t().shape[1] = T.shape[0]
                    marginals = torch.ones(T_var.shape[0]) / T_var.shape[0]
                marginals = torch.diag(1.0/(marginals + eps))  # take inverse
                T_var = torch.matmul(T_var, marginals)
            else:
                # marginals_alpha = T_var @ torch.ones(T_var.shape[1], dtype=T_var.dtype).to(device)
                marginals_beta = T_var.t() @ torch.ones(T_var.shape[0], dtype=T_var.dtype).to(device)

                marginals = (1 / (marginals_beta + eps))
                print("shape of inverse marginals beta is ", marginals_beta.shape)
                print("inverse marginals beta is ", marginals_beta)

                T_var = T_var * marginals
                # i.e., how a neuron of 2nd model is constituted by the neurons of 1st model
                # this should all be ones, and number equal to number of neurons in 2nd model
                print(T_var.sum(dim=0))
                # assert (T_var.sum(dim=0) == torch.ones(T_var.shape[1], dtype=T_var.dtype).to(device)).all()

        if args.debug:
            if idx == (num_layers - 1):
                print("there goes the last transport map: \n ", T_var)
            else:
                print("there goes the transport map at layer {}: \n ".format(idx), T_var)

            print("Ratio of trace to the matrix sum: ", torch.trace(T_var) / torch.sum(T_var))

        print("Ratio of trace to the matrix sum: ", torch.trace(T_var) / torch.sum(T_var))
        print("Here, trace is {} and matrix sum is {} ".format(torch.trace(T_var), torch.sum(T_var)))
        setattr(args, 'trace_sum_ratio_{}'.format(layer0_name), (torch.trace(T_var) / torch.sum(T_var)).item())

        if args.past_correction:
            print("this is past correction for weight mode")
            print("Shape of aligned wt is ", aligned_wt.shape)
            print("Shape of fc_layer0_weight_data is ", fc_layer0_weight_data.shape)
            t_fc0_model = torch.matmul(T_var.t(), aligned_wt.contiguous().view(aligned_wt.shape[0], -1))
        else:
            t_fc0_model = torch.matmul(T_var.t(), fc_layer0_weight_data.contiguous().view(fc_layer0_weight_data.shape[0], -1))

        # Average the weights of aligned first layers
        if args.ensemble_step != 0.5:
            geometric_fc = ((1-args.ensemble_step) * t_fc0_model +
                            args.ensemble_step * fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1))
        else:
            geometric_fc = (t_fc0_model + fc_layer1_weight_data.contiguous().view(fc_layer1_weight_data.shape[0], -1))/2
        if is_conv and layer_shape != geometric_fc.shape:
            geometric_fc = geometric_fc.view(layer_shape)
        
        print(f"Fused model shape {geometric_fc.shape}\n")
        
        avg_aligned_layers.append(geometric_fc)
        fused_state_dict[layer0_name] = geometric_fc
        # # get the performance of the model 0 aligned with respect to the model 1
        # if args.eval_aligned:
        #     if is_conv and layer_shape != t_fc0_model.shape:
        #         t_fc0_model = t_fc0_model.view(layer_shape)
        #     model0_aligned_layers.append(t_fc0_model)
        #     _, acc = update_model(args, networks[0], model0_aligned_layers, test=True,
        #                           test_loader=test_loader, idx=0)
        #     print("For layer idx {}, accuracy of the updated model is {}".format(idx, acc))
        #     setattr(args, 'model0_aligned_acc_layer_{}'.format(str(idx)), acc)
        #     if idx == (num_layers - 1):
        #         setattr(args, 'model0_aligned_acc', acc)

    return avg_aligned_layers, fused_state_dict

# %%
from typing import List, Callable, Union, Any, TypeVar, Tuple
# from torch import tensor as Tensor

Tensor = TypeVar('torch.tensor')

from abc import abstractmethod

class BaseVAE(nn.Module):
    
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass


class VanillaVAE(nn.Module):


    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 bias: bool,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1, bias=bias),
                    nn.BatchNorm2d(h_dim, affine=bias),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim, bias=bias)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim, bias=bias)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4, bias=bias)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1,
                                       bias=bias),
                    nn.BatchNorm2d(hidden_dims[i + 1], affine=bias),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1,
                                               bias=bias),
                            nn.BatchNorm2d(hidden_dims[-1], affine=bias),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3, kernel_size= 3, padding= 1, bias=bias),
                            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

# %%
bias = True
model1 = VanillaVAE(3, 128, bias)
model2 = VanillaVAE(3, 128, bias)
# avg_model = VanillaVAE(3, 128, False)
fused_model = VanillaVAE(3, 128, bias)
dict1 = dict(torch.load('last7.ckpt', map_location=torch.device('cpu'))['state_dict'])
dict2 = dict(torch.load('last8.ckpt', map_location=torch.device('cpu'))['state_dict'])
tmp1 = dict()
tmp2 = dict()
avg_tmp = dict()
# print(dict1.keys())
for k in dict1.keys():
    v1 = dict1[k]
    v2 = dict2[k]
    # print(k, v.shape)
    k = k[6:]
    # if "running" in k:
    #     continue
    # if "batches" in k:
    #     continue
    # if "final_layer.3" in k:
    #     k = "final_layer.2.weight"
    print(k)
    tmp1[k] = v1
    tmp2[k] = v2
    # avg_tmp[k] = (v1+v2)/2

# for k in dict2.keys():
#     v = dict2[k]
#     k = k[6:]
#     # print(k)    
#     # if "running" in k:
#     #     continue
#     # if "batches" in k:
#     #     continue
#     # if "final_layer.3" in k:
#     #     k = "final_layer.2.weight"
#     tmp2[k] = v

model1.load_state_dict(tmp1)
model2.load_state_dict(tmp2)
# avg_model.load_state_dict(avg_tmp)

# %%
print(model1, model2)

# %%
import argparse
# from parameters import get_parameter
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-epochs', default=1, type=int, help='number of epochs')
    parser.add_argument('--batch-size-train', default=64, type=int, help='training batch size')
    parser.add_argument('--batch-size-test', default=1000, type=int, help='test batch size')
    parser.add_argument('--learning-rate', default=0.01, type=float, help='learning rate for SGD (default: 0.01)')
    parser.add_argument('--momentum', default=0.5, type=float, help='momentum for SGD (default: 0.5)')

    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='log progress every N batches (when progress bar is disabled)')

    parser.add_argument('--to-download', action='store_true', help='download the dataset (typically mnist)')
    parser.add_argument('--disable_bias', action='store_false', help='disable bias in the neural network layers')
    parser.add_argument('--dataset', default='mnist', type=str, choices=['mnist', 'Cifar10'],
                        help='dataset to use for the task')
    parser.add_argument('--num-models', default=2, type=int, help='number of models to ensemble')
    parser.add_argument('--model-name', type=str, default='simplenet',
                        help='Type of neural network model (simplenet|smallmlpnet|mlpnet|bigmlpnet|cifarmlpnet|net|vgg11_nobias|vgg11)')
    parser.add_argument('--config-file', type=str, default=None, help='config file path')
    parser.add_argument('--config-dir', type=str, default="./configurations", help='config dir')

    # for simplenet
    parser.add_argument('--num-hidden-nodes', default=400, type=int, help='simplenet: number of hidden nodes in the only hidden layer')
    # for mlpnet
    parser.add_argument('--num-hidden-nodes1', default=400, type=int,
                        help='mlpnet: number of hidden nodes in the hidden layer 1')
    parser.add_argument('--num-hidden-nodes2', default=200, type=int,
                        help='mlpnet: number of hidden nodes in the hidden layer 2')
    parser.add_argument('--num-hidden-nodes3', default=100, type=int,
                        help='mlpnet: number of hidden nodes in the hidden layer 3')
    parser.add_argument('--num-hidden-nodes4', default=50, type=int,
                        help='mlpnet: number of hidden nodes in the hidden layer 3')

    parser.add_argument('--sweep-id', default=-1, type=int, help='sweep id ')

    parser.add_argument('--gpu-id', default=-1, type=int, help='GPU id to use')
    parser.add_argument('--skip-last-layer', action='store_true', help='skip the last layer in calculating optimal transport')
    parser.add_argument('--skip-last-layer-type', type=str, default='average', choices=['second', 'average'],
                        help='how to average the parameters for the last layer')

    parser.add_argument('--debug', action='store_true', help='print debug statements')
    parser.add_argument('--cifar-style-data', action='store_true', help='use data loader in cifar style')
    parser.add_argument('--activation-histograms', action='store_true', help='utilize activation histograms')
    parser.add_argument('--act-num-samples', default=100, type=int, help='num of samples to compute activation stats')
    parser.add_argument('--softmax-temperature', default=1, type=float, help='softmax temperature for activation weights (default: 1)')
    parser.add_argument('--activation-mode', type=str, default=None, choices=['mean', 'std', 'meanstd', 'raw'],
                        help='mode that chooses how the importance of a neuron is calculated.')

    parser.add_argument('--options-type', type=str, default='generic', choices=['generic'], help='the type of options to load')
    parser.add_argument('--deprecated', type=str, default=None, choices=['vgg_cifar', 'mnist_act'],
                        help='loaded parameters in deprecated style. ')

    parser.add_argument('--save-result-file', type=str, default='default.csv', help='path of csv file to save things to')
    parser.add_argument('--sweep-name', type=str, default=None,
                        help='name of sweep experiment')

    parser.add_argument('--reg', default=1e-2, type=float, help='regularization strength for sinkhorn (default: 1e-2)')
    parser.add_argument('--reg-m', default=1e-3, type=float, help='regularization strength for marginals in unbalanced sinkhorn (default: 1e-3)')
    parser.add_argument('--ground-metric', type=str, default='euclidean', choices=['euclidean', 'cosine'],
                        help='ground metric for OT calculations, only works in free support v2 and soon with Ground Metric class in all! .')
    parser.add_argument('--ground-metric-normalize', type=str, default='log', choices=['log', 'max', 'none', 'median', 'mean'],
                        help='ground metric normalization to consider! ')
    parser.add_argument('--not-squared', action='store_true', help='dont square the ground metric')
    parser.add_argument('--clip-gm', action='store_true', help='to clip ground metric')
    parser.add_argument('--clip-min', action='store', type=float, default=0,
                       help='Value for clip-min for gm')
    parser.add_argument('--clip-max', action='store', type=float, default=5,
                       help='Value for clip-max for gm')
    parser.add_argument('--tmap-stats', action='store_true', help='print tmap stats')
    parser.add_argument('--ensemble-step', type=float, default=0.5, action='store', help='rate of adjustment towards the second model')

    parser.add_argument('--ground-metric-eff', action='store_true', help='memory efficient calculation of ground metric')

    parser.add_argument('--retrain', type=int, default=0, action='store', help='number of epochs to retrain all the models & their avgs')
    parser.add_argument('--retrain-lr-decay', type=float, default=-1, action='store',
                        help='amount by which to reduce the initial lr while retraining the model avgs')
    parser.add_argument('--retrain-lr-decay-factor', type=float, default=None, action='store',
                        help='lr decay factor when the LR is gradually decreased by Step LR')
    parser.add_argument('--retrain-lr-decay-epochs', type=str, default=None, action='store',
                        help='epochs at which retrain lr decay factor should be applied. underscore separated! ')
    parser.add_argument('--retrain-avg-only', action='store_true', help='retraining the model avgs only')
    parser.add_argument('--retrain-geometric-only', action='store_true', help='retraining the model geometric only')

    parser.add_argument('--load-models', type=str, default='', help='path/name of directory from where to load the models')
    parser.add_argument('--ckpt-type', type=str, default='best', choices=['best', 'final'], help='which checkpoint to load')

    parser.add_argument('--recheck-cifar', action='store_true', help='recheck cifar accuracies')
    parser.add_argument('--recheck-acc', action='store_true', help='recheck model accuracies (recheck-cifar is legacy/deprecated)')
    parser.add_argument('--eval-aligned', action='store_true',
                        help='evaluate the accuracy of the aligned model 0')

    parser.add_argument('--enable-dropout', action='store_true', help='enable dropout in neural networks')
    parser.add_argument('--dump-model', action='store_true', help='dump model checkpoints')
    parser.add_argument('--dump-final-models', action='store_true', help='dump final trained model checkpoints')
    parser.add_argument('--correction', action='store_true', help='scaling correction for OT')

    parser.add_argument('--activation-seed', type=int, default=42, action='store', help='seed for computing activations')

    parser.add_argument('--weight-stats', action='store_true', help='log neuron-wise weight vector stats.')
    parser.add_argument('--sinkhorn-type', type=str, default='normal', choices=['normal', 'stabilized', 'epsilon', 'gpu'],
                        help='Type of sinkhorn algorithm to consider.')
    parser.add_argument('--geom-ensemble-type', type=str, default='wts', choices=['wts', 'acts'],
                        help='Ensemble based on weights (wts) or activations (acts).')
    parser.add_argument('--act-bug', action='store_true',
                        help='simulate the bug in ground metric calc for act based averaging')
    parser.add_argument('--standardize-acts', action='store_true',
                        help='subtract mean and divide by standard deviation across the samples for use in act based alignment')
    parser.add_argument('--transform-acts', action='store_true',
                        help='transform activations by transport map for later use in bi_avg mode ')
    parser.add_argument('--center-acts', action='store_true',
                        help='subtract mean only across the samples for use in act based alignment')
    parser.add_argument('--prelu-acts', action='store_true',
                        help='do activation based alignment based on pre-relu acts')
    parser.add_argument('--pool-acts', action='store_true',
                        help='do activation based alignment based on pooling acts')
    parser.add_argument('--pool-relu', action='store_true',
                        help='do relu first before pooling acts')
    parser.add_argument('--normalize-acts', action='store_true',
                        help='normalize the vector of activations')
    parser.add_argument('--normalize-wts', action='store_true',
                        help='normalize the vector of weights')
    parser.add_argument('--gromov', action='store_true', help='use gromov wasserstein distance and barycenters')
    parser.add_argument('--gromov-loss', type=str, default='square_loss', action='store',
                        choices=['square_loss', 'kl_loss'], help="choice of loss function for gromov wasserstein computations")
    parser.add_argument('--tensorboard-root', action='store', default="./tensorboard", type=str,
                        help='Root directory of tensorboard logs')
    parser.add_argument('--tensorboard', action='store_true', help='Use tensorboard to plot the loss values')

    parser.add_argument('--same-model', action='store', type=int, default=-1, help='Index of the same model to average with itself')
    parser.add_argument('--dist-normalize', action='store_true', help='normalize distances by act num samples')
    parser.add_argument('--update-acts', action='store_true', help='update acts during the alignment of model0')
    parser.add_argument('--past-correction', action='store_true', help='use the current weights aligned by multiplying with past transport map')
    parser.add_argument('--partial-reshape', action='store_true', help='partially reshape the conv layers in ground metric calculation')
    parser.add_argument('--choice', type=str, default='0 2 4 6 8', action='store',
                        help="choice of how to partition the labels")
    parser.add_argument('--diff-init', action='store_true', help='different initialization for models in data separated mode')

    parser.add_argument('--partition-type', type=str, default='labels', action='store',
                        choices=['labels', 'personalized', 'small_big'], help="type of partitioning of training set to carry out")
    parser.add_argument('--personal-class-idx', type=int, default=9, action='store',
                        help='class index for personal data')
    parser.add_argument('--partition-dataloader', type=int, default=-1, action='store',
                        help='data loader to use in data partitioned setting')
    parser.add_argument('--personal-split-frac', type=float, default=0.1, action='store',
                        help='split fraction of rest of examples for personal data')
    parser.add_argument('--exact', action='store_true', help='compute exact optimal transport')
    parser.add_argument('--skip-personal-idx', action='store_true', help='skip personal data')
    parser.add_argument('--prediction-wts', action='store_true', help='use wts given by ensemble step for prediction ensembling')
    parser.add_argument('--width-ratio', type=float, default=1, action='store',
                        help='ratio of the widths of the hidden layers between the two models')
    parser.add_argument('--proper-marginals', action='store_true', help='consider the marginals of transport map properly')
    parser.add_argument('--retrain-seed', type=int, default=-1, action='store',
                        help='if reseed computations again in retrain')
    parser.add_argument('--no-random-trainloaders', action='store_true',
                        help='get train loaders without any random transforms to ensure consistency')
    parser.add_argument('--reinit-trainloaders', action='store_true',
                        help='reinit train loader when starting retraining of each model!')
    parser.add_argument('--second-model-name', type=str, default=None, action='store', help='name of second model!')
    parser.add_argument('--print-distances', action='store_true', help='print OT distances for every layer')
    parser.add_argument('--deterministic', action='store_true', help='do retrain in deterministic mode!')
    parser.add_argument('--skip-retrain', type=int, default=-1, action='store', help='which of the original models to skip retraining')
    parser.add_argument('--importance', type=str, default=None, action='store',
                        help='importance measure to use for building probab mass! (options, l1, l2, l11, l12)')
    parser.add_argument('--unbalanced', action='store_true', help='use unbalanced OT')
    parser.add_argument('--temperature', default=20, type=float, help='distillation temperature for (default: 20)')
    parser.add_argument('--alpha', default=0.7, type=float, help='weight towards distillation loss (default: 0.7)')
    parser.add_argument('--dist-epochs', default=60, type=int, help='number of distillation epochs')

    parser.add_argument('--handle-skips', action='store_true', help='handle shortcut skips in resnet which decrease dimension')
    return parser


parser = get_parser()
args = parser.parse_args()

# _, fused_state_dict = get_wassersteinized_layers_modularized(args, [model1, model2])
# # fused_model.load_state_dict(fused_state_dict)

from torchvision.datasets import CelebA
from torchvision import transforms
transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.CenterCrop(148),
                                              transforms.Resize((64, 64)),
                                              transforms.ToTensor(),])

test_data = CelebA(root='/home/chenyuheng/celeba', split='test', download=False, transform=transforms) 
def loaddata():
    return torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)   

from tqdm import tqdm
def test(model):
    model.cuda()
    dataloader = loaddata()
    i = 0
    loss = 0.0
    recon_loss = 0.0
    kl_loss = 0.0
    for img, _ in tqdm(dataloader):
        # print(img)
        out = model(img.cuda())
        # loss = loss * (i / (i + 1)) + model.loss_function(*out, M_N=0.00025)['loss'].item() / (i + 1)
        loss = loss * (i / (i + 1)) + model.loss_function(*out, M_N=0.00025)['loss'].item() / (i + 1)
        recon_loss = recon_loss * (i / (i + 1)) + model.loss_function(*out, M_N=0.00025)['Reconstruction_Loss'].item() / (i + 1)
        kl_loss = kl_loss * (i / (i + 1)) + model.loss_function(*out, M_N=0.00025)['KLD'].item() / (i + 1)
        # loss = loss * (i / (i + 1)) + model.loss_function(*out, M_N=0.00025)['loss'].item() / (64 * (i + 1))
        i += 1
    return {'loss': loss, 'recon_loss': recon_loss, 'kl_loss': kl_loss}

def get_avg_model(t):
    avg_model = VanillaVAE(3, 128, bias)
    avg_tmp = dict()
    for k in dict1.keys():
        v1 = dict1[k]
        v2 = dict2[k]
        k = k[6:]
        avg_tmp[k] = t * v1 + (1 - t) * v2
    avg_model.load_state_dict(avg_tmp)
    return avg_model

for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    print(test(get_avg_model(t)))
# print(test(fused_model))
print(test(model1), test(model2))

# print(test(model1))