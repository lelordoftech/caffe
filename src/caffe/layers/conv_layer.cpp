#include <vector>

#include "caffe/layers/conv_layer.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Trim layer input
  if (this->is_quantized_) {
    if (this->phase_ == TEST) {
      for (int i = 0; i < bottom.size(); ++i) {
        this->QuantizeLayerInputs_cpu(bottom[i]->mutable_cpu_data(), bottom[i]->count());
      }
    }
  }

  // Trim weights
  const Dtype* weight = NULL;
  if (this->is_quantized_) {
    caffe_copy(this->blobs_[0]->count(), this->blobs_[0]->cpu_data(),
        this->weights_quantized_[0]->mutable_cpu_data());
    if (this->bias_term_) {
      caffe_copy(this->blobs_[1]->count(), this->blobs_[1]->cpu_data(),
          this->weights_quantized_[1]->mutable_cpu_data());
    }
    int rounding = this->phase_ == TEST ? this->rounding_ :
        QuantizationParameter_Rounding_STOCHASTIC;
    this->QuantizeWeights_cpu(this->weights_quantized_, rounding,
        this->bias_term_);

    weight = this->weights_quantized_[0]->cpu_data();
  } else {
    weight = this->blobs_[0]->cpu_data();
  }

  // Do forward propagation
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = NULL;
        if (this->is_quantized_) {
          bias = this->weights_quantized_[1]->cpu_data();
        } else {
          bias = this->blobs_[1]->cpu_data();
        }
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }

    // Trim layer output
    if (this->is_quantized_) {
      if (this->phase_ == TEST) {
        this->QuantizeLayerOutputs_cpu(top[i]->mutable_cpu_data(), top[i]->count());
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // Trim weights
  const Dtype* weight = NULL;
  if (this->is_quantized_) {
    weight = this->weights_quantized_[0]->cpu_data();
  } else {
    weight = this->blobs_[0]->cpu_data();
  }

  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
