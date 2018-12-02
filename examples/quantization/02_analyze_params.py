import sys
caffe_root = '/mnt/d/Caffe/Quantization/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import math
import numpy as np

prototxt = './examples/mnist/lenet_train_test.prototxt'
caffemodel = './examples/quantization/lenet/finetune/lenet_iter_1000.caffemodel'
compressed_file = './examples/quantization/lenet/lenetlenet_iter_1000.npz'
compressed_caffemodel = './examples/quantization/lenet/lenet_iter_1000_compz.caffemodel'

net = caffe.Net(prototxt, caffemodel, caffe.TEST)

# compress
xdict = dict()
bitwidth = 8
xdict['compz_info'] = (1, int(bitwidth))

for item in net.params.items():
  name, layer = item
  idx = list(net._layer_names).index(name)
  layer_type = net.layers[idx].type
  print "Compressing layer", name, ": ", layer_type

  weights = net.params[name][0].data
  bias = net.params[name][1].data

  weights_vec = weights.flatten().astype(np.float32)
  bias_vec = bias.flatten().astype(np.float32)
  vec_len = weights_vec.size

  new_weights_vec = np.empty(vec_len, dtype=np.int8)
  new_bias_vec = np.empty(vec_len, dtype=np.int8)

  max_params = max(weights_vec)
  il = int(np.ceil(math.log(max_params, 2)+1))
  fl = bitwidth - il

  new_weights_vec = np.round(weights_vec/pow(2, -fl)).astype(np.int8)
  new_bias_vec = np.round(bias_vec/pow(2, -fl)).astype(np.int8)

  xdict[name+'_fl'] = fl
  xdict[name+'_0'] = new_weights_vec
  xdict[name+'_1'] = new_bias_vec
  if layer_type == "BatchNorm":
    scale = net.params[name][2].data
    scale_vec = scale.flatten().astype(np.float32)
    new_scale_vec = np.empty(vec_len, dtype=np.int8)
    new_scale_vec = np.round(scale_vec/pow(2, -fl)).astype(np.int8)
    xdict[name+'_2'] = new_scale_vec
  print "Compressed  layer", name, ": ", layer_type, ": fl", fl

np.savez_compressed(compressed_file, **xdict)


# decompress
cmpr_model = np.load(compressed_file)
version, bitwidth = cmpr_model['compz_info']

assert(version == 1), "compz version not support"

print version, bitwidth

net = caffe.Net(prototxt, caffe.TEST)
for item in net.params.items():
  name, layer = item
  fl = int(cmpr_model[name+'_fl'])
  idx = list(net._layer_names).index(name)
  layer_type = net.layers[idx].type
  print "Decompressing layer", name, ": ", layer_type, ": fl", fl

  weights_vec = cmpr_model[name+'_0'].astype(np.float32)
  weights_vec = weights_vec * pow(2, -fl)
  new_weights = weights_vec.reshape(net.params[name][0].data.shape)
  bias_vec = cmpr_model[name+'_1'].astype(np.float32)
  bias_vec = bias_vec * pow(2, -fl)
  new_bias = bias_vec.reshape(net.params[name][1].data.shape)

  net.params[name][0].data[...] = new_weights
  net.params[name][1].data[...] = new_bias
  if layer_type == "BatchNorm":
    scale_vec = cmpr_model[name+'_3'].astype(np.float32)
    scale_vec = scale_vec * pow(2, -fl)
    new_scale = scale_vec.reshape(net.params[name][2].data.shape)
    net.params[name][2].data[...] = new_scale

net.save(compressed_caffemodel)
