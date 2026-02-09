//
//  Geometry.metal
//  MSLGraphDiffusion
//
//  Created by Raiden-Makoto on 2026-02-08.
//

#include <metal_stdlib>
#include <metal_atomic>
#include "ShaderTypes.h"

using namespace metal;

kernel void compute_cog(
    device const Node* nodes         [[buffer(0)]], // [N]
    device atomic_float* cog_sum     [[buffer(1)]], // [3] (x, y, z)
    constant uint& num_nodes         [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
){
    if (gid >= num_nodes){ return; }
    float3 pos = nodes[gid].pos;
    atomic_fetch_add_explicit(&cog_sum[0], pos.x, memory_order_relaxed);
    atomic_fetch_add_explicit(&cog_sum[1], pos.y, memory_order_relaxed);
    atomic_fetch_add_explicit(&cog_sum[2], pos.z, memory_order_relaxed);
}

kernel void cog_normalization(
    device Node* nodes               [[buffer(0)]], // [N]
    device const float* cog_sum      [[buffer(1)]], // [3]
    constant uint& num_nodes         [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
){
    if (gid >= num_nodes) return;
    float3 centroid = float3(cog_sum[0], cog_sum[1], cog_sum[2]) / (float) num_nodes;
    nodes[gid].pos -= centroid;
}
