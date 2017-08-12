import caffe
from caffe import layers as L
from caffe import params as P

def conv_bn_scale_relu(bottom, num_output=64, kernel_size=3, stride=1, pad=0, bias_term=False, use_global_stats=False):

    conv = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad, bias_term=bias_term, weight_filler=dict(type='xavier'))
    conv_bn = L.BatchNorm(conv, in_place=True, batch_norm_param =dict(use_global_stats=use_global_stats))
    conv_scale = L.Scale(conv, in_place=True, scale_param=dict(bias_term=True))
    conv_relu = L.ReLU(conv, in_place=True)

    return conv, conv_bn, conv_scale, conv_relu

def conv_bn_scale_relu_top(num_output=64, kernel_size=3, stride=1, pad=0, bias_term=False, use_global_stats=False):

    conv = L.Convolution(bottom="data", num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad, bias_term=bias_term, weight_filler=dict(type='xavier'))
    conv_bn = L.BatchNorm(conv, in_place=True, batch_norm_param =dict(use_global_stats=use_global_stats))
    conv_scale = L.Scale(conv ,in_place=True, scale_param=dict(bias_term=True))
    conv_relu = L.ReLU(conv, in_place=True)

    return conv, conv_bn, conv_scale, conv_relu


def conv_bn_scale(bottom, num_output=64, kernel_size=3, stride=1, pad=0, bias_term=False, use_global_stats=False):

    conv = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad, bias_term=bias_term, weight_filler=dict(type='xavier'))
    conv_bn = L.BatchNorm(conv, in_place=True, batch_norm_param =dict(use_global_stats=use_global_stats))
    conv_scale = L.Scale(conv, in_place=True, scale_param=dict(bias_term=True))

    return conv, conv_bn, conv_scale


def eltwize_relu(bottom1, bottom2):

    residual_eltwise = L.Eltwise(bottom1, bottom2, eltwise_param=dict(operation=1))
    residual_eltwise_relu = L.ReLU(residual_eltwise, in_place=True)

    return residual_eltwise, residual_eltwise_relu


def residual_branch(bottom, base_output=64, use_global_stats=False):
    branch2a, branch2a_bn, branch2a_scale, branch2a_relu = conv_bn_scale_relu(bottom, num_output=base_output, kernel_size=1, use_global_stats=use_global_stats)
    branch2b, branch2b_bn, branch2b_scale, branch2b_relu = conv_bn_scale_relu(branch2a, num_output=base_output, kernel_size=3, pad=1, use_global_stats=use_global_stats)
    branch2c, branch2c_bn, branch2c_scale = conv_bn_scale(branch2b, num_output=4 * base_output, kernel_size=1, use_global_stats=use_global_stats)  # 4*base_output x n x n

    residual, residual_relu = eltwize_relu(bottom, branch2c)

    return branch2a, branch2a_bn, branch2a_scale, branch2a_relu, branch2b, branch2b_bn, branch2b_scale, branch2b_relu, \
           branch2c, branch2c_bn, branch2c_scale, residual, residual_relu
    # return residual_relu


def residual_branch_shortcut(bottom, stride=2, base_output=64, use_global_stats=False):

    branch1, branch1_bn, branch1_scale = conv_bn_scale(bottom, num_output=4 * base_output, kernel_size=1, stride=stride, use_global_stats=use_global_stats)

    branch2a, branch2a_bn, branch2a_scale, branch2a_relu = conv_bn_scale_relu(bottom, num_output=base_output, kernel_size=1, stride=stride, use_global_stats=use_global_stats)
    branch2b, branch2b_bn, branch2b_scale, branch2b_relu = conv_bn_scale_relu(branch2a, num_output=base_output, kernel_size=3, pad=1, use_global_stats=use_global_stats)
    branch2c, branch2c_bn, branch2c_scale = conv_bn_scale(branch2b, num_output=4 * base_output, kernel_size=1, use_global_stats=use_global_stats)

    residual, residual_relu = eltwize_relu(branch1, branch2c)  # 4*base_output x n x n

    return branch1, branch1_bn, branch1_scale, branch2a, branch2a_bn, branch2a_scale, branch2a_relu, branch2b, \
           branch2b_bn, branch2b_scale, branch2b_relu, branch2c, branch2c_bn, branch2c_scale, residual, residual_relu
    # return residual_relu

branch_shortcut_string = 'net.res(stage)a_branch1, net.bn(stage)a_branch1, net.scale(stage)a_branch1, \
        net.res(stage)a_branch2a, net.bn(stage)a_branch2a, net.scale(stage)a_branch2a, net.res(stage)a_branch2a_relu, \
        net.res(stage)a_branch2b, net.bn(stage)a_branch2b, net.scale(stage)a_branch2b, net.res(stage)a_branch2b_relu, \
        net.res(stage)a_branch2c, net.bn(stage)a_branch2c, net.scale(stage)a_branch2c,\
        net.res(stage)a, net.res(stage)a_relu = residual_branch_shortcut((bottom), stride=(stride), base_output=(num), use_global_stats=(stat))'

branch_string = 'net.res(stage)b(order)_branch2a, net.bn(stage)b(order)_branch2a, net.scale(stage)b(order)_branch2a, \
        net.res(stage)b(order)_branch2a_relu, net.res(stage)b(order)_branch2b, net.res(stage)b(order)_branch2b_bn, \
        net.res(stage)b(order)_branch2b_scale, net.res(stage)b(order)_branch2b_relu, net.res(stage)b(order)_branch2c, \
        net.res(stage)b(order)_branch2c_bn, net.res(stage)b(order)_branch2c_scale, net.res(stage)b(order), net.res(stage)b(order)_relu = \
            residual_branch((bottom), base_output=(num), use_global_stats=(stat))'

def identify_block(net, stage, order, bottom, base_output, use_global_stats):
    res = branch_string.replace("(stage)", str(stage))
    res = res.replace("(bottom)", str(bottom))
    res = res.replace("(order)", str(order))
    res = res.replace("(num)", str(base_output))
    res = res.replace("(stat)", str(use_global_stats))
    exec(res)
    

def conv_block(net, stage, stride, bottom, base_output, use_global_stats):
    res = branch_shortcut_string.replace("(stage)", str(stage))
    res = res.replace("(bottom)", str(bottom))
    res = res.replace("(stride)", str(stride))
    res = res.replace("(num)", str(base_output))
    res = res.replace("(stat)", str(use_global_stats))
    exec(res)