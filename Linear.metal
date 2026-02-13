//
//  TimestepMLP.metal
//  MSLGraphDiffusion
//
//  Created by Raiden Makoto on 2026-02-08.
//

#include <metal_stdlib>
#include "ShaderTypes.h"

using namespace metal;

// KERNEL 1: 128 -> 128 (Explicit, Safe)
kernel void linear_128x128(
    device const float* input   [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    device const float* bias    [[buffer(2)]],
    device float* output        [[buffer(3)]],
    constant uint& count        [[buffer(4)]],
    constant uint& use_silu     [[buffer(5)]], // Must match UInt32 in Swift
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;

    uint offset = gid * 128;
    for (uint out_f = 0; out_f < 128; out_f++) {
        float acc = bias[out_f];
        for (uint in_f = 0; in_f < 128; in_f++) {
            acc += input[offset + in_f] * weight[out_f * 128 + in_f];
        }
        output[offset + out_f] = (use_silu == 1) ? silu(acc) : acc;
    }
}

// KERNEL 2: 128 -> 1 (Explicit, Safe)
kernel void linear_128x1(
    device const float* input   [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    device const float* bias    [[buffer(2)]],
    device float* output        [[buffer(3)]],
    constant uint& count        [[buffer(4)]],
    constant uint& use_silu     [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;

    uint offset = gid * 128;
    float acc = bias[0];
    for (uint i = 0; i < 128; i++) {
        acc += input[offset + i] * weight[i];
    }
    output[gid] = (use_silu == 1) ? silu(acc) : acc;
}
