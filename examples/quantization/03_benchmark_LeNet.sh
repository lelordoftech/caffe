#!/usr/bin/env sh

MODE=dynamic_fixed_point

./build/tools/caffe test \
	--model=./examples/quantization/lenet/quantized_$MODE.prototxt \
	--weights=./examples/quantization/lenet/lenet_iter_1000_compz.caffemodel \
	--iterations=200