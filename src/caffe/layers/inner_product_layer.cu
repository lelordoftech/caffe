#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void InnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // Trim layer input
  if (this->is_quantized_) {
    if (this->phase_ == TEST) {
      this->QuantizeLayerInputs_gpu(bottom[0]->mutable_gpu_data(), bottom[0]->count());
    }
  }

  // Trim weights
  const Dtype* weight = NULL;
  if (this->is_quantized_) {
    caffe_copy(this->blobs_[0]->count(), this->blobs_[0]->gpu_data(),
        this->weights_quantized_[0]->mutable_gpu_data());
    if (this->bias_term_) {
      caffe_copy(this->blobs_[1]->count(), this->blobs_[1]->gpu_data(),
          this->weights_quantized_[1]->mutable_gpu_data());
    }
    int rounding = this->phase_ == TEST ? this->rounding_ :
        QuantizationParameter_Rounding_STOCHASTIC;
    this->QuantizeWeights_gpu(this->weights_quantized_, rounding,
        this->bias_term_);

    weight = this->weights_quantized_[0]->cpu_data();
  } else {
    weight = this->blobs_[0]->cpu_data();
  }

  // Do forward propagation
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();

  // Trim bias
  const Dtype* bias = NULL;
  if (bias_term_) {
    if (this->is_quantized_) {
      bias = this->weights_quantized_[1]->cpu_data();
    } else {
      bias = this->blobs_[1]->cpu_data();
    }
  }

  if (M_ == 1) {
    caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
                         weight, bottom_data, (Dtype)0., top_data);
    if (bias_term_) {
      caffe_gpu_axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],
                            bias, top_data);
    }
  } else {
    caffe_gpu_gemm<Dtype>(CblasNoTrans,
                          transpose_ ? CblasNoTrans : CblasTrans,
                          M_, N_, K_, (Dtype)1.,
                          bottom_data, weight, (Dtype)0., top_data);
    if (bias_term_) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                            bias_multiplier_.gpu_data(),
                            bias, (Dtype)1., top_data);
    }
  }

  // Trim layer output
  if (this->is_quantized_) {
    if (this->phase_ == TEST) {
      this->QuantizeLayerOutputs_gpu(top[0]->mutable_cpu_data(), top[0]->count());
    }
  }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    // Gradient with respect to weight
    if (transpose_) {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.gpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_gpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bottom data

    // Trim weights
    const Dtype* weight = NULL;
    if (this->is_quantized_) {
      weight = this->weights_quantized_[0]->cpu_data();
    } else {
      weight = this->blobs_[0]->cpu_data();
    }

    if (transpose_) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, weight,
          (Dtype)0., bottom[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
         (Dtype)1., top_diff, weight,
         (Dtype)0., bottom[0]->mutable_gpu_diff());
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(InnerProductLayer);

}  // namespace caffe
