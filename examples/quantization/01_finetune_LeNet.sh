#!/usr/bin/env sh

#dynamic_fixed_point
#integer_power_of_2_weights
MODE=dynamic_fixed_point

./build/tools/caffe train \
  --solver=./examples/quantization/lenet/finetune_lenet_solver_adam.prototxt \
  --weights=./examples/mnist/lenet_iter_10000.caffemodel
