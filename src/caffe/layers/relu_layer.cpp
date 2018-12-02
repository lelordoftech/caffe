#include <algorithm>
#include <vector>

#include "caffe/layers/relu_layer.hpp"

namespace caffe {

template <typename Dtype>
ReLULayer<Dtype>::ReLULayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param), BaseRistrettoLayer<Dtype>() {
  this->is_quantized_ = this->layer_param_.quantization_param().is_quantized();
  this->precision_ = this->layer_param_.quantization_param().precision();
  this->rounding_ = this->layer_param_.quantization_param().rounding_scheme();
  switch (this->precision_) {
  case QuantizationParameter_Precision_DYNAMIC_FIXED_POINT:
    this->bw_layer_in_ = this->layer_param_.quantization_param().bw_layer_in();
    this->bw_layer_out_ = this->layer_param_.quantization_param().bw_layer_out();
    //this->bw_params_ = this->layer_param_.quantization_param().bw_params();
    this->fl_layer_in_ = this->layer_param_.quantization_param().fl_layer_in();
    this->fl_layer_out_ = this->layer_param_.quantization_param().fl_layer_out();
    //this->fl_params_ = this->layer_param_.quantization_param().fl_params();
    break;
  case QuantizationParameter_Precision_MINIFLOAT:
    this->fp_mant_ = this->layer_param_.quantization_param().mant_bits();
    this->fp_exp_ = this->layer_param_.quantization_param().exp_bits();
    break;
  case QuantizationParameter_Precision_INTEGER_POWER_OF_2_WEIGHTS:
    //this->pow_2_min_exp_ = this->layer_param_.quantization_param().exp_min();
    //this->pow_2_max_exp_ = this->layer_param_.quantization_param().exp_max();
    this->bw_layer_in_ = this->layer_param_.quantization_param().bw_layer_in();
    this->bw_layer_out_ = this->layer_param_.quantization_param().bw_layer_out();
    this->fl_layer_in_ = this->layer_param_.quantization_param().fl_layer_in();
    this->fl_layer_out_ = this->layer_param_.quantization_param().fl_layer_out();
    break;
  default:
    LOG(FATAL) << "Unknown precision mode: " << this->precision_;
    break;
  }
}

template <typename Dtype>
void ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // Trim layer input
  if (this->is_quantized_) {
    if (this->phase_ == TEST) {
      this->QuantizeLayerInputs_cpu(bottom[0]->mutable_cpu_data(), bottom[0]->count());
    }
  }

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  for (int i = 0; i < count; ++i) {
    top_data[i] = std::max(bottom_data[i], Dtype(0))
        + negative_slope * std::min(bottom_data[i], Dtype(0));
  }

  // Trim layer output
  if (this->is_quantized_) {
    if (this->phase_ == TEST) {
      this->QuantizeLayerOutputs_cpu(top[0]->mutable_cpu_data(), top[0]->count());
    }
  }
}

template <typename Dtype>
void ReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + negative_slope * (bottom_data[i] <= 0));
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ReLULayer);
#endif

INSTANTIATE_CLASS(ReLULayer);

}  // namespace caffe
