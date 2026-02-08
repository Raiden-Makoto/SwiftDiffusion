//
//  TimestepMLP.metal
//  MSLGraphDiffusion
//
//  Created by Raiden Makoto on 2026-02-08.
//

#include <metal_stdlib>
#include "ShaderTypes.h"

inline float silu(float x){
    return x / (1.0f + exp(-x));
}

// generic linear layer
kernel void linear_layer(
    device const float* input  [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias   [[buffer(2)]],
    device float* output       [[buffer(3)]],
    constant uint& in_dim      [[buffer(4)]],
    constant uint& out_dim     [[buffer(5)]],
    constant bool& apply_silu  [[buffer(6)]],
    uint gid [[thread_position_in_grid]] // thread(s) per output neuron
){
    if (gid >= out_dim) return;
    float acc = bias[gid];
    for (uint i=0; i<in_dim; i++){
        acc += input[i] * weight[gid * in_dim + i];
    }
    output[gid] = apply_silu ? silu(acc) : acc;
}
