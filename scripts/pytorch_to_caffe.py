import argparse
import torch
import caffe
import torch.nn as nn
from default2018_model import Net
from dense_single_model_modified import Dense

def copy_weights(m,name):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        l_name = name
        print(l_name,  m.weight.data.cpu().numpy().shape)
        net.params[l_name][0].data[:] = m.weight.data.cpu().numpy()
        net.params[l_name][1].data[:] = m.bias.data.cpu().numpy()
    elif isinstance(m, nn.BatchNorm3d):
        l_name = name
        print(l_name, m.weight.data.cpu().numpy().shape)
        ## first handle the batchnorm
        net.params[l_name][0].data[:] = m.running_mean.data.cpu().numpy()
        net.params[l_name][1].data[:] = m.running_var.data.cpu().numpy()
        net.params[l_name][2].data[:] = torch.tensor(1).cpu().numpy()
        ## now the scale
        l_name = l_name.replace('batchnorm_conv','scale')
        net.params[l_name][0].data[:] = m.weight.data.cpu().numpy()
        net.params[l_name][1].data[:] = m.bias.data.cpu().numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_in', '-I', required=True, help='path of the pytorch weights file')
    parser.add_argument('--weights_out', '-O', required=True, help='Outname of the `.caffemodel` file')
    parser.add_argument('--arch', '-A', choices=['default2018','dense'], default='default2018', help='Architecture that needs converting')
    args = parser.parse_args()

    if args.arch == 'default2018':
        model = Net((28, 48, 48, 48))
        net = caffe.Net('default2018_modif.model', caffe.TEST)
    else:
        model = Dense((28, 48, 48, 48))
        net = caffe.Net('dense_modif.model', caffe.TEST)
    model.load_state_dict(torch.load(args.weights_in))

    caffe.set_mode_cpu()

    for name, layer in model.named_children():
        if args.arch == 'dense' and "dense_block" in name:
            for subname, sublayer in layer.named_children():
                copy_weights(sublayer,subname)
        else:
            copy_weights(layer, name)

    net.save(args.weights_out)
