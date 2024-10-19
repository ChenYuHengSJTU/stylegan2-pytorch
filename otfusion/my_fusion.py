import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import ot

# for conv layer, pack bias and weights together
# 对于x作为输入和输出的情况是不同的，体现在assertion中
# 但是无论x是输入还是经过weight和bias packing后的卷积核处理后的输出，其多出的一个维度均为全1
def conv_pack(x, weight, bias=None):
# 将out*in*h*w的weight和out的bias，使用类似于fc矩阵的填充方式，填充为(out+1)*(in+1)*h*w
# 对于输入维度的增加，输入的激活值也需要增加一个维度，变为(in+1)*H*W，新增的维度的H*W个元素均为1
# 同时，卷积核中，对于out*(in+1)*h*w，对于最后一个h*w，设置为out_j / hw,当然也有另外的设置方式，只需要保持hw个元素的和为out_j即可
# 对于最后一个输出通道1*(in+1)*h*w，将前in个h*w均设置为全0，最后一个h*w设置为1/hw（或者和为1即可）
    out_ch, in_ch, h, w = weight.shape
    kernel_size = h * w
    bias_scaled = bias / kernel_size
    # new in channel
    bias_scaled = bias_scaled.reshape(out_ch, 1).repeat((1, kernel_size)).reshape(out_ch, 1, h, w)
    weight_packed = torch.concat((weight, bias_scaled), dim=1)
    # new out channel
    out_packed = torch.concat((torch.zeros((1, in_ch, h, w)), torch.full((1, 1, h, w), 1.0 / kernel_size)), dim = 1)
    weight_packed = torch.cat((weight_packed, out_packed), dim = 0)
    
    # new in channel for input
    if len(x.shape) == 3:
        # no batch dim
        # assert x.shape[0] == in_ch
        assert x.shape[0] == out_ch
        x = torch.concat((x, torch.ones(1, h, w)), dim = 0)
        assert x.shape[0] == out_ch + 1
    else:
        # has a batch dim
        assert len(x.shape) == 4 and x.shape[1] == out_ch
        x = torch.concat((x, torch.ones(x.shape[0], 1, x.shape[-2], x.shape[-1])), dim = 1)
        assert x.shape[1] == out_ch + 1

    return x, weight_packed
    pass

# 参数x应该是weight的输出
def conv_unpack(x, weight):
# 将(out+1)*(in+1)*h*w的kernel拆分为out*in*h*w的weight和out的bias
# 考虑到fused之后的kernel中，对应bias的(out+1)*1*h*w中的h*w不再相等，所以可以需要取h*w个元素的和
# TODO：当然也可以尝试其他的处理方法，比如取max/min/mean/avg sum(hw)
    out_ch, in_ch, h, w = weight.shape     
    kernel_size = h * w
    bias_unpacked = weight[:-1,-1,:,:].sum((-2, -1)).reshape(out_ch - 1)
    weight_unpacked = weight[:-1,:-1,:,:]
    
    if len(x.shape) == 3:
        assert x.shape[0] == out_ch
        x = x[:-1,:,:]
        assert x.shape[0] == out_ch - 1
    else:
        assert len(x.shape) == 4
        assert x.shape[1] == out_ch
        x = x[:,:-1,:,:]
        assert x.shape[1] == out_ch - 1
        
    return x, weight_unpacked, bias_unpacked
    pass

# pack fc weights and bias together
# 将fc权重的(out, in)和bias(out)组合为新的weight(out+1,in+1)
# 其中，bias变为[bias,1]追加在第in+1个维度，第out+1个维度为全0，除了最后一个weight(out,in)=1
def fc_pack(x, weight, bias):
    out_dim, in_dim = weight.shape
    weight_packed = torch.concat((weight, bias.view(out_dim, -1)), dim=-1)
    add_dim = torch.zeros(in_dim+1)
    add_dim[-1] = 1
    weight_packed = torch.concat((weight_packed, add_dim.view(-1, in_dim + 1)), dim=0)
    
    if len(x.shape) == 1:
        # assert x.shape[0] == in_dim
        assert x.shape[0] == out_dim
        x = torch.concat((x, torch.tensor([1])), dim=0)
        assert x.shape[0] == out_dim + 1
    else:
        assert len(x.shape) == 2 and x.shape[1] == out_dim
        x = torch.concat((x, torch.ones(x.shape[0]).view(-1, 1)), dim=1)
        assert x.shape[1] == out_dim + 1
    
    return x, weight_packed
    pass

def fc_unpack(x, weight):
    # fused之后，在bias之后补充的1也有可能不再为1
    # TODO：研究是否需要根据1变化后的值对fused bias进行处理，如scaling等
    out_dim, in_dim = weight.shape
    bias_unpacked = weight[:,-1][:-1]
    weight_unpacked = weight[:-1,:-1]
    
    if len(x.shape) == 1:
        assert x.shape[0] == out_dim
        x = x[-1]
        assert x.shape[0] == out_dim -1 
    else:
        assert len(x.shape) == 2 and x.shape[1] == out_dim
        x = x[:,0:-1]
        assert x.shape[1] == out_dim - 1
    
    return x, weight_unpacked, bias_unpacked
    pass

def test_pack():
    B, out_ch, in_ch, h, w, H, W = 64, 64, 32, 3, 3, 64, 64
    x = torch.rand((B, in_ch, H, W)).to(dtype=torch.float64)
    kernel = torch.rand((out_ch, in_ch, h, w)).to(dtype=torch.float64)
    bias = torch.rand((out_ch)).to(dtype=torch.float64)
    x_packed, weight_packed = conv_pack(x, kernel, bias)
    x_packed = F.conv2d(x_packed, weight_packed)
    x = F.conv2d(x, kernel, bias=bias)
    x_unpacked, weight_unpacked, bias_unpacked = conv_unpack(x_packed, weight_packed)
    # TODO:计算误差
    # print(torch.max(x - x_unpacked))
    # print(torch.max(kernel - weight_unpacked))
    # print(torch.max(bias - bias_unpacked))
    assert (torch.abs(x - x_unpacked) < 1e-12).all()
    assert (torch.abs(kernel - weight_unpacked) < 1e-12).all()
    assert (torch.abs(bias - bias_unpacked) < 1e-12).all()

    in_dim, out_dim = 1024, 128
    x = torch.rand((B, in_dim)).to(dtype=torch.float64)
    w = torch.rand((out_dim, in_dim)).to(dtype=torch.float64)
    b = torch.rand(out_dim).to(dtype=torch.float64)
    x_packed, w_packed = fc_pack(x, w, b)
    x = F.linear(x, w, bias=b)
    x_packed = F.linear(x_packed, w_packed)
    x_unpacked, w_unpacked, b_unpacked = fc_unpack(x_packed, w_packed)
    # print(torch.max(x - x_unpacked))
    # print(torch.max(w - w_unpacked))
    # print(torch.max(b - b_unpacked))
    assert (torch.abs(x - x_unpacked) < 1e-12).all()
    assert (torch.abs(w - w_unpacked) < 1e-12).all()
    assert (torch.abs(b - b_unpacked) < 1e-12).all()

def compute_activations(model, train_loader, num_samples, gpu_id=0):
    '''

    This method can be called from another python module. Example usage demonstrated here.
    Averages the activations across the 'num_samples' many inputs.

    :param model: takes in a pretrained model
    :param train_loader: the particular train loader
    :param num_samples: # of randomly selected training examples to average the activations over

    :return:  list of len: num_layers and each of them is a particular tensor of activations
    '''
    activation = {}
    num_samples_processed = 0

    # Define forward hook that averages the activations
    # over number of samples processed
    def get_activation(name):
        def hook(model, input, output):
            # print("num of samples seen before", num_samples_processed)
            # print("output is ", output.detach())
            if name not in activation:
                activation[name] = output.detach()
            else:
                # print("previously at layer {}: {}".format(name, activation[name]))
                activation[name] = (num_samples_processed * activation[name] + output.detach()) / (num_samples_processed + 1)
            # print("now at layer {}: {}".format(name, activation[name]))

        return hook

    model.train()
    if gpu_id != -1:
        model.cuda(gpu_id)
    # Set forward hooks for all the layers
    for name, layer in model.named_modules():
        if name == '':
            print("excluded")
            continue
        layer.register_forward_hook(get_activation(name))
        print("set forward hook for layer named: ", name)

    # Run over the samples in training set
    # datapoints= []
    for batch_idx, (data, target) in enumerate(train_loader):
        if gpu_id != -1:
            data = data.cuda(gpu_id)
            # datapoints.append(data)
            model(data)
            num_samples_processed += 1
            if num_samples_processed == num_samples:
                break
    return activation #, datapoints

# preprocessing函数主要是进行weight和bias和packing以及conv到fc的reshape操作
# 输入模型应该保证包含的所有module的bias一致，同为true或者false
# TODO：pack bias也会导致由conv到fc无法直接进行reshape操作，可以通过强制要求feature map为1*1
def conv_preprocessing(acts, kernel, bias=None):
    if bias is None:
        return acts, kernel
    else:
        acts_packed, weight_packed = conv_pack(x=acts, weight=kernel, bias=bias)
        return acts_packed, weight_packed.view(weight_packed.shape[0], weight_packed.shape[1], -1).permute(2, 0, 1)
    
def fc_preprocessing(acts, weight, bias=None):
    if bias is None:
        return acts, weight
    else:
        # TODO：这里需要考虑的是，从conv到fc，多出的一个维度如何处理，可以先折叠fc的维度，再适当补充一个维度
        return fc_pack(x=acts, weight=weight, bias=bias)

# paying attention to the shape difference between conv kernel and deconv kernel
# 此处需要和align函数统一，默认在这里permute kernel的维度
def convtransposed_preprocessing(acts, weight, bias=None):
    # 需要注意的是，这里默认在同一个模型中，bias对所有的conv layer都生效
    if bias is None:
        return acts, weight
    acts_packed, weight_packed = conv_pack(x=acts, weight=weight.permute(1, 0, 2, 3), bias=bias)
    assert weight_packed.shape[1] == weight.shape[0] + 1 and weight_packed.shape[0] == weight.shape[1] + 1
    return acts_packed, weight_packed.view(weight_packed.shape[0], weight_packed.shape[1], -1).permute(2, 0, 1)


# for bn, if set affine = False, the the params of the bn layer is None
# TODO:if set track_running_stats to Ture, whether to fuse the running vars
def bn_preprocessing(acts, weight=None, bias=None):
    # just pack eta and gamma together
    if weight is None:
        assert bias is None
        return acts
    else:
        # weight: 2*out
        return acts, torch.concat((weight.unsqueeze(0), bias.unsqueeze(0)), dim=0)
    pass

# acts and params packing
def preprocessing(acts, flag, *params):
    return OPS[flag]['preprocessing'](acts, *params)

def fc_postprocessing(acts, param, key, state_dict, origin_param):
# conv -> fc have been reshaped to 3d
    # if len(param) == 3:        
# 目前由于bias和fc->deconv的存在，feature map的大小只能设置为1*1
    assert len(param[0].shape) == 2 and len(param[1].shape) == 2
    _, *param1 = fc_unpack(x=acts, weight=param[0])
    _, *param2 = fc_unpack(x=acts, weight=param[1])
    if len(param1) == 2 and len(param2) == 2:
        state_dict[key + '.weight'] = (param1[0] + param2[0]) / 2
        state_dict[key + '.bias'] = (param1[1] + param2[1]) / 2
    else:
        raise Exception("NOT IMPLEMENTED")
    return state_dict

# origin_param -> tuple(weight, bias=None)
def conv_postprocessing(acts, param, key, state_dict, origin_param):
    assert len(param[0].shape) == 3 
    hw, out_ch, in_ch = param[0].shape
    h, w = origin_param[0].shape[-2], origin_param[0].shape[-1]
    param[0] = param[0].permute(1, 2, 0).reshape(out_ch, in_ch, h, w)
    param[1] = param[1].permute(1, 2, 0).reshape(out_ch, in_ch, h, w)
    assert len(param[0].shape) == 4 and len(param[1].shape) == 4
    _, *param1 = conv_unpack(x=acts, weight=param[0])
    _, *param2 = conv_unpack(x=acts, weight=param[1])
    if len(param1) == 2 and len(param2) == 2:
        assert len(param1[0].shape) == 4, param1[0].shape
        state_dict[key + '.weight'] = (param1[0] + param2[0]) / 2
        state_dict[key + '.bias'] = (param1[1] + param2[1]) / 2
    else:
        raise Exception("NOT IMPLEMENTED")
    return state_dict

def convtransposed_postprecessing(acts, param, key, state_dict, origin_param):
    assert len(param[0].shape) == 3 
    hw, out_ch, in_ch = param[0].shape
    h, w = origin_param[0].shape[-2], origin_param[0].shape[-1]
    param[0] = param[0].permute(1, 2, 0).reshape(out_ch, in_ch, h, w)
    param[1] = param[1].permute(1, 2, 0).reshape(out_ch, in_ch, h, w)
    assert len(param[0].shape) == 4 and len(param[1].shape) == 4
    acts_unpacked, *param1 = conv_unpack(acts, param[0])
    acts_unpacked, *param2 = conv_unpack(acts, param[1])
    if len(param1) == 2 and len(param2) == 2:
        assert len(param1[0].shape) == 4, param1[0].shape
        state_dict[key + '.weight'] = (param1[0].permute(1, 0, 2, 3) + param2[0].permute(1, 0, 2, 3)) / 2
        state_dict[key + '.bias'] = (param1[1] + param2[1]) / 2
    else:
        raise Exception("NOT IMPLEMENTED")
    return state_dict
    # return acts_unpacked, weight_unpacked.permute(1, 0, 2, 3), bias_unpacked

# (2,out)
def bn_postprocessing(acts, param, key, state_dict, origin_param):
    assert len(param[0].shape) == 2 and len(param[1].shape) == 2
    assert param[0].shape[0] == 2 and param[1].shape[0] == 2
    state_dict[key + '.weight'] = (param[0][0] + param[1][0]) / 2
    state_dict[key + '.bias'] = (param[1][0] + param[1][1]) / 2
    return state_dict

# acts and params unpacking
# 没有考虑bias有可能为none的情况
def postprecessing(acts, flag, params, key, state_dict, origin_param):
    return OPS[flag]['postprocessing'](acts, params, key, state_dict, origin_param)
    pass

# need to handle the case that the feature map is reshaped, passed from conv to fc
# for fc, the param shape is (out, in)
def align_fc(T, cardinality, param):
    param_data = param
    if T.shape[0] != param.shape[1]:
        print("handling the case where acts is passed from conv to fc")
        param_data = param.view(param.shape[0], T.shape[0], -1)
    return torch.matmul(param, T)
    pass

# for conv and convtranspose(if feature map is 1*1)
def align_conv(T, cardinality, param):
    # param shoule have been reshape to (hw, out, in)
    assert len(param.shape) == 3 and T.shape[0] == param.shape[-1]
    return torch.bmm(torch.bmm(param, T.unsqueeze(0).repeat(param.shape[0], 1, 1)), torch.diag(1.0 / cardinality).unsqueeze(0).repeat(param.shape[0], 1, 1))
    
# for bn, just align the eta and gamma according to the input T
# the param's shape is (2, out)
# no need to multiply cardinality rightside
# 对于batchnorm层的对齐，同样有多种方式
# 1.(out) -> (out+1) * T -> (out)
# 2.(out) -> (out+0) * T -> (out)
# 2<=> T[:-1,:-1]
def align_bn(T, cardinality, param):
    if param is None:
        return param
    assert T.shape[0] == param.shape[1] + 1
    return torch.torch.matmul(param, T[:-1, :-1])
    pass

# 需要处理由fc的输出被reshape成一定形状的feature map后输入到convtranspose层中去
# 目前有两种解决方法
# 1.固定feature map为(out, 1, 1)的形状
# TODO 2.reshape T
# the convtransposed kernel have been reshaped to (hw, out+1, in+1)
def align_convtransposed(T, cardinality, param):
    assert len(param.shape) == 3 and T.shape[0] == param.shape[-1]
    return torch.bmm(torch.bmm(param, T.unsqueeze(0).repeat(param.shape[0], 1, 1)), torch.diag(1.0 / cardinality).unsqueeze(0).repeat(param.shape[0], 1, 1))

# cardinality is a tensor
def align_params(flag, T, cardinality, param):
    assert T is not None
    return OPS[flag]['align'](T, cardinality, param)
    pass

# return a string which is in OPS.keys()
def get_layer_flag(acts, *params):
    if params is None or len(params) == 0:
        return 'ACTIVATION'
    print(type(params), acts.shape, params[0].shape)
    if params is None or len(params[0].shape) == 1:
        return 'BN'
    elif len(params[0].shape) == 2:
        return 'FC'
    elif len(params[0].shape) == 4:
        if len(acts.shape) == 4:
            # act: (B, out, H, w), deconv: (in, out, h, w) in < out
            # conv: (out, in, h, w)
            # acts is the output activations
            if acts.shape[1] != params[0].shape[0]:
                assert acts.shape[1] == params[0].shape[1]
                return 'CONVTRANSPOSED'
            else:
                assert acts.shape[1] == params[0].shape[0]
                return 'CONV'
        elif len(acts.shape) == 3:
            if acts.shape[0] != params[0].shape[0]:
                assert acts.shape[0] == params[0].shape[1]
                return 'CONVTRANSPOSED'
            else:
                assert acts.shape[1] == params[0].shape[0]
                return 'CONV'
        else:
            print(f"acts shaped {acts.shape} do not match conv shaped {params[0].shape}")
            exit(1)
    else:
        print("params[0] shape can not be 3")
        exit(1)
                

# def get_layer_name(layer_name):
    
# get probability
# version all equal to 1/cardinality
def get_histogram(cardinality):
    return torch.ones(cardinality) / cardinality    

# version that probability depends on the activations
def get_histogram_acts(acts, cardinality):
    pass

# version that offer a measure to decide neuron importance
def get_histogram_importance(acts, cardinality, measure):
    pass

def compute_cost_weights():
    pass

# (B, out + 1)
# (B, out + 1, H, W)
def compute_cost_acts(acts1, acts2):
    assert len(acts1.shape) == len(acts2.shape)
    print(acts1.device)
    # print(acts1.shape)
    if len(acts1.shape) == 4:
        acts1_data = acts1.view(acts1.shape[0], acts1.shape[1], -1).unsqueeze(-2)
        acts2_data = acts2.view(acts2.shape[0], acts2.shape[1], -1).unsqueeze(-3)
        return torch.square(acts1_data - acts2_data).sum(-1).sum(0) / acts1.shape[0]
    else:
        assert len(acts1.shape) == 2
        return torch.square(acts1.unsqueeze(-1) - acts2.unsqueeze(1)).sum(0) / acts1.shape[0]
    pass

def adjust_fc(T, cardinality, param):
    assert T.shape[1] == param.shape[0]
    return torch.matmul(torch.diag(1.0 / cardinality), torch.matmul(T, param))

def adjust_conv(T, cardinality, param):
    # param shape: (hw, out+1,in+1)
    assert T.shape[1] == param.shape[1]
    return torch.bmm(torch.diag(1.0 / cardinality).unsqueeze(0).repeat(param.shape[0], 1, 1), torch.bmm(T.unsqueeze(0).repeat(param.shape[0], 1, 1), param))

def adjust_bn(T, cardinality, param):
    return param

def adjust_weights(flag, T, cardinality, param=None):
    assert T is not None
    return OPS[flag]['adjust'](T, cardinality, param)

OPS = {
    'FC':{
        'preprocessing': fc_preprocessing,
        'postprocessing': fc_postprocessing,
        'adjust': adjust_fc,
        'align': align_fc,
        },
    'CONV':{
        'preprocessing': conv_preprocessing,
        'postprocessing': conv_postprocessing,
        'adjust': adjust_conv,
        'align': align_conv,
        },
    'CONVTRANSPOSED':{
        'preprocessing': convtransposed_preprocessing,
        'postprocessing': convtransposed_postprecessing,
        'adjust': adjust_conv,
        'align': align_convtransposed
        },
    'BN':{
        'preprocessing': bn_preprocessing,
        'postprocessing': bn_postprocessing,
        'adjust': adjust_bn,
        'align': align_bn
        },
}

# acts1 and acts2 have been averaged according to the sample dim
# shape[0] is batch dim
def compute_weighted_scalar(acts1, acts2):
    return torch.norm(acts1, p=1) / (torch.norm(acts1, p=1) + torch.norm(acts2 / acts2.shape[0], p=1))

# TODO
def compute_weighted_matrix(acts1, acts2):
    pass

# check whether the layers number of models match the length of activation list
def check_model_act_matching(model, acts):
    name_list = []
    # not duplicated
    module_name = []
    for name, module in model.named_modules():
        # print(name, module)
        name_list.append(name)
    for i in range(len(name_list)):
        name = name_list[i] + '.'
        flag = True
        for j in range(i, len(name_list)):
            if name_list[j].startswith(name):
               flag = False
               break 
        if flag:
            module_name.append(name_list[i])
    return module_name

    pass

# 获得依赖图
def get_model_structure(model):
    gm = torch.fx.symbolic_trace(model)
    graph = gm.graph
    node_list = graph.nodes
    gm.graph.print_tabular()
    tmp = dict()
    for node in node_list:
        # print(node.name)
        if node.op == 'call_module':
            tmp[node.target] = set()
            for prev in node.all_input_nodes:
                if prev.op == 'call_module':
                    tmp[node.target].add(prev.target)
                else:
                    # tmp[node.target].extend(tmp[prev.name])
                    tmp[node.target] |= tmp[prev.name]

        else:
            tmp[node.name] = set()
            for prev in node.all_input_nodes:
                if prev.op == 'call_module':
                    # tmp[node.name].extend(tmp[prev.target])
                    tmp[node.name].add(prev.target)
                else:
                    # tmp[node.name].extend(tmp[prev.name])
                    tmp[node.name] |= tmp[prev.name]

    return tmp
    

# fusion using activations
# two model version len(models) == 2
# 如何根据卷积核来区分conv和convtranspose尚未有一个完备的方式
# 只能默认使用convtranspose都是进行上采样的，因此feature map会增大，然后通过比较输入feature map的维度和卷积核的前两个维度进行比对来区分conv和convtranspose
# 同样，由于pytorch的module容器的遍历方式等原因，这里的遍历方式只支持模型init函数中没有使用modulelist的情况（否则会嵌套遍历）
# TODO:对于模型内同时存在多个数据通路的情况，目前是希望通过构建module依赖图来进行处理，目前没有找到合适的python lib
def act_fusion(models, activations):
    model1, model2 = models
    acts1, acts2 = activations
    print(type(activations), type(acts1), type(acts2))
    module_name1 = check_model_act_matching(model1, acts1)
    module_name2 = check_model_act_matching(model2, acts2)
    print(f"model1 has {len(module_name1)} layers")
    print(f"model2 has {len(module_name2)} layers")

    dag = get_model_structure(model1)
    print(dag)

    incoming_T = dict()
    incoming_acts = dict()

    assert len(module_name1) == len(module_name2), "model1 and model2 must have the same number of layers"
    
    T_prev = None
    mu = None
    nu = None
    
    idx = 0
    
    fused_params = {}
    for (name1, module1), (name2, module2) in zip(model1.named_modules(), model2.named_modules()):
        # idx += 1
        if name1 == ' ' or name1 == '':
            assert name2 == ' ' or name2 == ''
            continue
        
        # 跳过嵌套定义的module名称
        if name1 not in module_name1 or name2 not in module_name2:
            continue
        
        print('\n', "=" * 100)
        print(f"\ncurrent layer name is {name1} and {name2}")

        cur_acts1, cur_acts2 = acts1[name1].cpu(), acts2[name2].cpu()

        # print(type(module1.parameters()))
        layer_flag = get_layer_flag(cur_acts1, *module1.parameters())    
        
        print("current layer type is ", layer_flag)
        
        if layer_flag == 'ACTIVATION':
            incoming_T[name1] = T_prev
            incoming_acts[name1] = cur_acts1
            idx += 1
            continue
        
        cur_acts1_packed, param1_packed = preprocessing(cur_acts1, layer_flag, *module1.parameters())
        cur_acts2_packed, param2_packed = preprocessing(cur_acts2, layer_flag, *module2.parameters())

        print("param1 packed shape: ", param1_packed.shape)
        if 'CONV' in layer_flag:
            mu_cardinality = param1_packed.shape[1]
            nu_cardinality = param2_packed.shape[1]
        else:
            mu_cardinality = param1_packed.shape[0]
            nu_cardinality = param2_packed.shape[0]
        
        # can be modified for better
        mu = get_histogram(mu_cardinality)
        nu = get_histogram(nu_cardinality)

        # align 阶段只需要处理第一个model的权重
        if idx == 0:
            # TODO:是否需要右乘diag(1/mu)
            param1_aligned_packed = param1_packed
            # param2_aligned_packed = param2_packed
        else:
            # TODO：目前只考虑输入有两个T的情况
            if len(dag[name1]) > 1:
                print(f"There are {len(dag[name1])} incoming Ts")
                tmp_acts = [incoming_acts[k] for k in dag[name1]]
                gamma = compute_weighted_scalar(*tmp_acts)
                print("scalar gamma is: ", gamma)
                tmp_T = [incoming_T[k] for k in dag[name1]]
                T_prev = tmp_T[0] * gamma + tmp_T[1] * (1-gamma)
            else:
                T_prev = incoming_T[list(dag[name1])[0]]
                
            param1_aligned_packed = align_params(layer_flag, T_prev, cardinality=get_histogram(param1_packed.shape[-1]), param=param1_packed)
            # param2_aligned_packed = align_params(layer_flag, T_)
        
        print("param1 packed aligned shape: ", param1_aligned_packed.shape)
        
        if layer_flag != 'BN' and idx != len(module_name1):
            M = compute_cost_acts(acts1=cur_acts1_packed, acts2=cur_acts2_packed)
            print("M shape: ", M.shape, " mu shape: ", mu.shape, " nu shape: ", nu.shape)
            
            T = ot.emd(mu, nu, M)
            
            print("T: ", T)
            print("T sum: ", T.sum(0).sum(0), " T trace: ", torch.trace(T))
        
            param1_adjusted_packed = adjust_weights(flag=layer_flag, T=T, cardinality=mu, param=param1_aligned_packed)
            
            # save T and activations
            incoming_T[name1] = T
            incoming_acts[name1] = cur_acts1_packed
            
        else:
            print(layer_flag, idx, sep='\t')
            param1_adjusted_packed = param1_aligned_packed
            incoming_T[name1] = T_prev
            incoming_acts[name1] = cur_acts1_packed


        fused_params = postprecessing(acts=cur_acts1_packed, flag=layer_flag, params=[param1_adjusted_packed, param2_packed], key=name1, state_dict=fused_params, origin_param=[p for p in module1.parameters()])

        # assert param2_packed is not None
        # fused_param = (param1_adjusted_packed + param2_packed) / 2

        # fused_params.append(*fused_param)
        
        # T_prev = T
        idx += 1
    
    return fused_params
    pass

from torchvision.datasets import CelebA
from torchvision import transforms

def loaddata(test_data):
    return torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)   

from tqdm import tqdm
def test(model, test_data):
    model.cuda()
    dataloader = loaddata(test_data)
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

# def get_avg_model(t):
#     avg_model = VanillaVAE(3, 128, bias)
#     avg_tmp = dict()
#     for k in dict1.keys():
#         v1 = dict1[k]
#         v2 = dict2[k]
#         k = k[6:]
#         avg_tmp[k] = t * v1 + (1 - t) * v2
#     avg_model.load_state_dict(avg_tmp)
#     return avg_model



# A $$n_s, d$$     -> source
# B $$n_t, d$$     -> target
# T $$n_s, n_t$$
# M $$n_s, n_t$$
# feature map of shape (out, in, h, w) <- kernel shape of conv2d
# TODO:solve the case where kernel shape does not match
# dist (out1, out2)
# fm1 -> source
# fm2 -> target
def compute_distance(fm1, fm2):
    # element-wise distance
    # fm1 = fm1.permute(1, 0, 2, 3)
    # fm2 = fm2.permute(1, 0, 2, 3)
    # fm1 : (out, 1 , in , h, w)
    # fm2 : (1, out, in, h, w)
    out_dim, in_dim, h, w = fm1.shape
    dist1 = (fm1.unsqueeze(1) - fm2.unsqueeze(0)).pow(2).sum(dim=(range(2, len(fm1.shape + 1))))
    assert dist1.shape == torch.Size([out_dim, out_dim]), "dist shape dose not match"
    return dist1


# W1, W2 can be sample latents
def get_prior_T(W1, W2):
    pass

# weight-based fusion version
# s/preconv -> conv1 -> conv2 -> rgb/nextconv
# gan inversion result :1) avg (16, 512) 2) guassian 
# TODO: 1) bias 2) beta norm 3) gamma
def fuse_stylegan2(gan1, gan2, W1, W2):
    # fuse constant input with shape(1, 512, 4, 4)
    # there are other ways to compute the distance between feature map(4*4)
    
    # ---------------------------------------------------------------------------- #
    # fuse input
    in1, in2 = gan1.input.input, gan2.input.input
    in1, in2 = in1.permute(1, 0, 2, 3), in2.permute(1, 0, 2, 3)
    dist_input = compute_distance(in1, in2)
    
    a, b = get_histogram(in1.shape[0]), get_histogram(in2.shape[0])
    T = ot.emd(a, b, dist_input)
    assert T.shape == torch.Size([a, b])
    
    in1_tilde = torch.matmul(T.t, in1)
    
    # ---------------------------------------------------------------------------- #
    # fuse conv1 styleconv.conv: kernel + A (latent_dim, in_channel)
    
    conv1, conv2 = gan1.conv1.conv.weight, gan2.conv1.conv.weight
    A1, A2 = gan1.conv1.conv.modulation.weight, gan2.conv2.conv.modulation.weight
    
    T_a = get_prior_T(W1, W2)
    # (latent1=A1.shape[1], latent2=A2.shape[1])
    assert T_a.shape[0] == A1.shape[1] and T_a.shape[1] == A2.shape[1]
    
    A1_hat = torch.matmul(A1, T_a)
    dist_A1 = compute_distance(A1_hat, A2)
    
    a, b = get_histogram(A1.shape[0]), get_histogram(A2.shape[0])
    T_a = ot.emd(a, b, dist_A1)
    A1_tilde = torch.matmul(T_a.t, A1_hat)
    
    # TODO: the order and way that T_a, T take effect on conv to come
    # dist_conv1 = compute_distance(conv1, conv2)
    assert conv1.shape[1] == T.shape[0] and conv2.shape[1] == T.shape[1]
    conv1_hat = torch.matmul(conv1, T)
    conv1_hat = torch.matmul(conv1_hat, T_a)
    
    dist_conv1 = compute_distance(conv1_hat, conv2)
    a, b = get_histogram(conv1.shape[0]), get_histogram(conv2.shape[0])    
    T = ot.emd(a, b, dist_conv1)
    
    conv1_tilde = torch.matmul(T.t, conv1_hat)
    
    # -------------------------------- fuse trgb1 -------------------------------- #
    # trgb also has a style layer
    trgb1, trgb2 = gan1.to_rgb1.conv.weight, gan2.to_rgb2.conv.weight
    A1, A2 = gan1.to_rgb1.conv.modulation.weight, gan2.to_rgb2.conv.modulation.weight    
    
    T_a = get_prior_T(W1, W2)
    assert T_a.shape[0] == A1.shape[1] and T_a.shape[1] == A2.shape[1]
    A1_hat = torch.matmul(A1, T_a)
    dist_A1 = compute_distance(A1_hat, A2)
    a, b = get_histogram(A1.shape[0], A2.shape[0])
    T_a = ot.emd(a, b, dist_A1)
    A1_tilde = torch.matmul(T_a.t, A1_hat)

    # dist_conv1 = compute_distance(conv1, conv2)
    assert trgb1.shape[1] == T.shape[0] and trgb2.shape[1] == T.shape[1]
    trgb1_hat = torch.matmul(trgb1, T)
    trgb1_hat = torch.matmul(trgb1_hat, T_a)    
    
    dist_trgb1 = compute_distance(trgb1_hat, trgb2)
    a, b = get_histogram(trgb1.shape[0]), get_histogram(trgb2.shape[0])  
    T = ot.emd(a, b, dist_trgb1)
    
    trgb1_tilde = torch.matmul(T.t, trgb1_hat)
    
    # --------------------------- fuse following layers -------------------------- #
    for conv11, conv12, conv21, conv22, trgb1, trgb2 in zip(gan1.convs[::2], gan1.convs[1::2], gan2.convs[::2], gan2.convs[1::2], gan1.to_rgbs, gan2.to_rgbs):
        assert T.shape[0] == conv11.conv.weight.shape[1] and T.shape[1] == conv21.shpae[1]
        
        pass
    
    pass

if __name__ == "__main__":
    transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.CenterCrop(148),
                                              transforms.Resize((64, 64)),
                                              transforms.ToTensor(),])

    test_data = CelebA(root='/home/chenyuheng/celeba', split='test', download=False, transform=transforms) 
    train_data = CelebA(root='/home/chenyuheng/celeba', split='train', download=False, transform=transforms) 

    # for i in tqdm(range(1000)):
    #     test_pack()
    from vaes.vanilla_vae import VanillaVAE
    model1 = VanillaVAE(3, 64)
    model2 = VanillaVAE(3, 64)
    
    dict1 = torch.load('/home/chenyuheng/otfusion/vae_1*1_122.ckpt', map_location=torch.device('cpu'))['state_dict']    
    dict2 = torch.load('/home/chenyuheng/otfusion/vae_1*1_125.ckpt', map_location=torch.device('cpu'))['state_dict']    

    tmp1 = dict()
    tmp2 = dict()

    for k in dict1.keys():
        v1 = dict1[k]
        v2 = dict2[k]

        k = k[6:]
        # print(k)
        tmp1[k] = v1
        tmp2[k] = v2
    
    model1.load_state_dict(tmp1)
    model2.load_state_dict(tmp2)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    acts1 = compute_activations(model1, train_loader, 100)
    acts2 = compute_activations(model2, train_loader, 100)
    model1.cpu()
    model2.cpu()

    fused_state_dict = act_fusion(models=[model1, model2], activations=[acts1, acts2])
    fused_model = VanillaVAE(3, 64)
    fused_model.load_state_dict(fused_state_dict)

    print(test(fused_model, test_data))