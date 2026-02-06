//
//  Loss.metal
//  MSLGraphDiffusion
//
//  Created by Raiden Makoto on 2026-02-06.
//

#include "ShaderTypes.h"
#include <metal_stdlib>
#include <metal_atomic>

kernel void compute_mse_loss(
    device const float* prediction    [[buffer(0)]], // posUpdateBuffer [N*3]
    device const float* target        [[buffer(1)]], // targetNoiseBuffer [N*3]
    device atomic_float* loss_accum   [[buffer(2)]], // A single float sum
    constant uint& total_elements     [[buffer(3)]], // activeNodeCount * 3
    uint gid [[thread_position_in_grid]]
){
    if (gid >= total_elements) return;

    float diff = prediction[gid] - target[gid];
    float squared_error = diff * diff;
    
    // Sum the error across all atoms/dimensions
    atomic_fetch_add_explicit(loss_accum, squared_error, memory_order_relaxed);
}
