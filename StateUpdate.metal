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
        atomic_fetch_add_explicit(&pos_agg[i * 3 + c], trans[gid * 3 + c], memory_order_relaxed);
    }
}

kernel void layer_pos_update(
    device Node* nodes          [[buffer(0)]],
    device const float* pos_agg [[buffer(1)]],
    constant uint& num_nodes    [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= num_nodes) return;

    uint base = gid * 3;
    float3 shift = float3(pos_agg[base], pos_agg[base + 1], pos_agg[base + 2]);
    
    // Move the node for the next layer to use
    nodes[gid].pos += shift;
}

// because you cannot do blit operations inside a computecommand encoder
kernel void clear_buffer_float(
    device float* buffer     [[buffer(0)]],
    constant uint& count     [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    buffer[gid] = 0.0f;
}
