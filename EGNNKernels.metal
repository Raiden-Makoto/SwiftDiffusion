//
//  EGNNKernels.metal
//  MSLGraphDiffusion
//
//  Created by Raiden-Makoto on 2026-02-08.
//

#include <metal_stdlib>
#include <metal_atomic>
#include "ShaderTypes.h" // includes SiLU activation function

using namespace metal;

kernel void compute_message(
                            device const float* h            [[buffer(0)]], // Node features [N * 128]
                            device const Node* nodes         [[buffer(1)]], // Node positions [N]
                            device const int2* edge_index    [[buffer(2)]], // Connectivity [E]
                            device const float* weight       [[buffer(3)]], // [128, 257]
                            device const float* bias         [[buffer(4)]], // [128]
                            device float* msg_out            [[buffer(5)]], // [E * 128]
                            constant uint& hidden_dim        [[buffer(6)]],
                            uint gid [[thread_position_in_grid]]
                            ){
    int i = edge_index[gid].x;
    int j = edge_index[gid].y;
    
    // Calculate Radial Invariant: squared distance + epsilon
    float3 diff = nodes[i].pos - nodes[j].pos;
    float radial = min(dot(diff, diff), 2500.0f) + 1e-8f;

    for (uint row = 0; row < hidden_dim; row++) {
        float acc = bias[row];
        // hi contribution
        for (uint c = 0; c < hidden_dim; c++)
            acc += h[i * hidden_dim + c] * weight[row * (2 * hidden_dim + 1) + c];
        // hj contribution
        for (uint c = 0; c < hidden_dim; c++)
            acc += h[j * hidden_dim + c] * weight[row * (2 * hidden_dim + 1) + hidden_dim + c];
        // radial contribution
        acc += radial * weight[row * (2 * hidden_dim + 1) + 2 * hidden_dim];
        msg_out[gid * hidden_dim + row] = silu(acc);
    }
}

// EGNNKernels.metal

kernel void compute_displacement(
                                 device const float* coord_scalar [[buffer(0)]], // [E * 1]
                                 device const Node* nodes          [[buffer(1)]],
                                 device const int2* edge_index     [[buffer(2)]],
                                 device float* trans_out           [[buffer(3)]], // FIXED: Use float* instead of float3*
                                 constant uint& num_edges          [[buffer(4)]],
                                 uint gid [[thread_position_in_grid]]
                                 )
{
    if (gid >= num_edges) return;

    // 1. Properly index the flat float buffer (12-byte stride)
    uint base = gid * 3;

    // 2. Clamp the scalar to a reasonable range to prevent the Galaxy Exit
    float scalar = clamp(coord_scalar[gid], -1.03f, 1.03f);

    int i = edge_index[gid].x;
    int j = edge_index[gid].y;
    
    float3 pos_i = nodes[i].pos;
    float3 pos_j = nodes[j].pos;
    
    // 3. Calculate relative displacement
    float3 diff = pos_i - pos_j;
    float dist = length(diff) + 1e-6f;
    
    // 4. Use unit direction so the force doesn't scale with the 17,000Å gap
    float3 unit_direction = diff / dist;
    float3 translation = unit_direction * scalar;

    // 5. Write out the final floats
    trans_out[base + 0] = translation.x;
    trans_out[base + 1] = translation.y;
    trans_out[base + 2] = translation.z;
}

kernel void compute_node(
                         device const float* h            [[buffer(0)]], // [N * 128]
                         device const float* msg_agg      [[buffer(1)]], // [N * 128]
                         device const float* weight       [[buffer(2)]], // [128, 256]
                         device const float* bias         [[buffer(3)]], // [128]
                         device float* h_hidden           [[buffer(4)]], // [N * 128]
                         constant uint& hidden_dim        [[buffer(5)]],
                         uint gid [[thread_position_in_grid]]
                         ){
    for (uint row = 0; row < hidden_dim; row++) {
        float acc = bias[row];
        for (uint c = 0; c < hidden_dim; c++)
            acc += h[gid * hidden_dim + c] * weight[row * (2 * hidden_dim) + c];
        for (uint c = 0; c < hidden_dim; c++)
            acc += msg_agg[gid * hidden_dim + c] * weight[row * (2 * hidden_dim) + hidden_dim + c];
        
        h_hidden[gid * hidden_dim + row] = silu(acc);
    }
}
