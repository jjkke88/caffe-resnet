from resnet import *
def get_net(type="train"):
    net = caffe.NetSpec()
    input_shape = [3, 224, 224]
    label = L.Input(shape=dict(dim=1000), name="label")
    global use_global_stats
    if type == "train":
        use_global_stats = False
        data = L.Input(shape=dict(dim=[10] + input_shape), name="data")
        net.data = data
        net.label = label
        net.conv1, net.conv1_bn, net.conv1_scale, net.conv1_relu = conv_bn_scale_relu(net.data, num_output=64,
                                                                                      kernel_size=7, stride=2,
                                                                                      pad=3, bias_term=False,
                                                                                      use_global_stats=use_global_stats)
    elif type == "test":
        use_global_stats = True
        data = L.Input(shape=dict(dim=[1] + input_shape), name="data")
        net.data = data
        net.conv1, net.conv1_bn, net.conv1_scale, net.conv1_relu = conv_bn_scale_relu(net.data, num_output=64,
                                                                                      kernel_size=7, stride=2,
                                                                                      pad=3, bias_term=False,
                                                                                      use_global_stats=use_global_stats)
    else:
        use_global_stats = True
        net.conv1, net.conv1_bn, net.conv1_scale, net.conv1_relu = conv_bn_scale_relu_top(num_output=64,
                                                                                          kernel_size=7, stride=2,
                                                                                          pad=3, bias_term=False,
                                                                                          use_global_stats=use_global_stats)
    net.pool1 = L.Pooling(net.conv1, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    # net.identify_block = residual_branch(net.pool1, base_output=16, use_global_stats=use_global_stats)
    #### block1
    # conv_block(net, stage=1, bottom="net.pool1", stride=2, base_output=64, use_global_stats=use_global_stats)
    # identify_block(net, stage=1, order=1, bottom="net.res1a_relu", base_output=64,
    #                use_global_stats=use_global_stats)
    # identify_block(net, stage=1, order=2, bottom="net.res1b1_relu", base_output=64,
    #                use_global_stats=use_global_stats)
    #### block2
    conv_block(net, stage=2, bottom="net.pool1", stride=2, base_output=128, use_global_stats=use_global_stats)
    identify_block(net, stage=2, order=1, bottom="net.res2a_relu", base_output=128,
                   use_global_stats=use_global_stats)
    identify_block(net, stage=2, order=2, bottom="net.res2b1_relu", base_output=128,
                   use_global_stats=use_global_stats)
    identify_block(net, stage=2, order=3, bottom="net.res2b2_relu", base_output=128,
                   use_global_stats=use_global_stats)
    # identify_block(net, stage=2, order=4, bottom="net.res2b3_relu", base_output=128,
    #                use_global_stats=use_global_stats)
    # identify_block(net, stage=2, order=5, bottom="net.res2b4_relu", base_output=128,
    #                use_global_stats=use_global_stats)
    # identify_block(net, stage=2, order=6, bottom="net.res2b5_relu", base_output=128,
    #                use_global_stats=use_global_stats)
    # #### block3
    conv_block(net, stage=3, bottom="net.res2b3_relu", stride=2, base_output=256, use_global_stats=use_global_stats)
    identify_block(net, stage=3, order=1, bottom="net.res3a_relu", base_output=256,
                   use_global_stats=use_global_stats)
    identify_block(net, stage=3, order=2, bottom="net.res3b1_relu", base_output=256,
                   use_global_stats=use_global_stats)
    identify_block(net, stage=3, order=3, bottom="net.res3b2_relu", base_output=256,
                   use_global_stats=use_global_stats)
    identify_block(net, stage=3, order=4, bottom="net.res3b3_relu", base_output=256,
                   use_global_stats=use_global_stats)
    identify_block(net, stage=3, order=5, bottom="net.res3b4_relu", base_output=256,
                   use_global_stats=use_global_stats)
    net.conv = L.Convolution(net.res3b5_relu, convolution_param=dict(kernel_size=3, stride=1, pad=1,
                                                                     num_output=4096,
                                                                     weight_filler=dict(type='xavier')))
    net.fc = L.InnerProduct(net.conv, num_output=1000)
    #
    #
    ####
    if type == "train":
        net.loss = L.SoftmaxWithLoss(net.fc, net.label)
        net.acc = L.Accuracy(net.fc, net.label)
    elif type == "test":
        net.output = L.Softmax(net.fc)
    return net
def write_net():
    train_net = get_net(type="train")
    test_net = get_net(type="test")
    deploy_net = get_net(type="deploy")
    # write train.prototxt
    with open("train.prototxt", 'w') as f:
        f.write(str(train_net.to_proto()))
    # write test.prototxt
    with open("test.prototxt", 'w') as f:
        f.write(str(test_net.to_proto()))
    # write deploy.prototxt

if __name__ == "__main__":
    write_net()