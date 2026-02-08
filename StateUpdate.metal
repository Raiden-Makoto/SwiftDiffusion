//
//  StateUpdate.metal
//  MSLGraphDiffusion
//
//  Created by Raiden Makoto on 2026-02-08.
//

#include <metal_stdlib>
#include "ShaderTypes.h"
#include <metal_atomic>

using namespace metal;

kernel void aggregate(
    device const float* msg          [[buffer(0)]], // [E * 128]
    device const float* trans        [[buffer(1)]], // [E * 3]
    device const int2* edge_index    [[buffer(2)]], // [E]
    device atomic_float* m_agg       [[buffer(3)]], // [N * 128]
    device atomic_float* pos_agg     [[buffer(4)]], // [N * 3]
    constant uint& hidden_dim        [[buffer(5)]],
    constant uint& num_edges         [[buffer(6)]], // ADDED
    uint gid [[thread_position_in_grid]]
){
    if (gid >= num_edges){ return; }
    int i = edge_index[gid].x;
    // Aggregate messages into node i
    for (uint h = 0; h < hidden_dim; h++){
        atomic_fetch_add_explicit(&m_agg[i * hidden_dim + h], msg[gid * hidden_dim + h], memory_order_relaxed);
    }
    // Aggregate coordinate translations into node i
    for (uint c = 0; c < 3; c++){
        atomic_fetch_add_explicit(&pos_agg[i * 3 + c], ((device float*)&trans[gid])[c], memory_order_relaxed);
    }
}

kernel void apply_update(
    device Node* nodes               [[buffer(0)]], // [N]
    device float* h                  [[buffer(1)]], // [N * 128]
    device const float* h_update     [[buffer(2)]], // [N * 128] (Result of node_mlp)
    device const float* pos_agg      [[buffer(3)]], // [N * 3]
    constant uint& hidden_dim        [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
){
    float3 update = float3(pos_agg[gid * 3], pos_agg[gid * 3 + 1], pos_agg[gid * 3 + 2]);
    nodes[gid].pos += update;
    // Node Feature Residual: h_new = h + node_mlp(h, agg_msg)
    for (uint i = 0; i < hidden_dim; i++) {
        h[gid * hidden_dim + i] += h_update[gid * hidden_dim + i];
    }
}
