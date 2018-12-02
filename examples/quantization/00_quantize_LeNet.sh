#!/usr/bin/env sh

#dynamic_fixed_point
#integer_power_of_2_weights
MODE=dynamic_fixed_point

./build/tools/ristretto quantize \
	--model=./examples/mnist/lenet_train_test.prototxt \
	--weights=./examples/mnist/lenet_iter_10000.caffemodel \
	--model_quantized=./examples/quantization/lenet/quantized_$MODE.prototxt \
	--trimming_mode=$MODE --iterations=200 \
	--error_margin=3 --quant_all=true --static_bitwidth=8
