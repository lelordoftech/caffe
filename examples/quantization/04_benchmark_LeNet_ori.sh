#!/usr/bin/env sh

MODE=dynamic_fixed_point

./build/tools/caffe test \
	--model=./examples/mnist/lenet_train_test.prototxt \
	--weights=./examples/mnist/lenet_iter_10000.caffemodel \
	--iterations=200