//
//  StateUpdate.metal
//  MSLGraphDiffusion
//
//  Created by Raiden Makoto on 2026-02-06.
//

#include "ShaderTypes.h"
#include <metal_stdlib>
#include <metal_atomic>

inline float silu(float x){
    return x / (1.0f + exp(-x));
}


kernel void apply_updates(
    device Node* nodes              [[buffer(0)]], // Mutable positions
  device float* h                 [[buffer(1)]], // Mutable features
  device const float* pos_delta   [[buffer(2)]], // From update_coords
  device const float* h_agg       [[buffer(3)]], // Aggregated messages
  device const float* node_w      [[buffer(4)]], // Node MLP Weights [H, 2H]
  device const float* node_b      [[buffer(5)]], // Node MLP Bias [H]
  constant uint& hidden_dim       [[buffer(6)]],
  uint gid [[thread_position_in_grid]]){
        float scale = 0.01f;
        // 1. Move Atoms
        nodes[gid].pos += float3(pos_delta[gid*3], pos_delta[gid*3+1], pos_delta[gid*3+2]) * scale;
        // 2. Node MLP
        // We concatenate h and h_agg virtually here
        for (uint row = 0; row < hidden_dim; row++) {
                        float activation = node_b[row];
                        for (uint col = 0; col < 2 * hidden_dim; col++) {
                                        float val = (col < hidden_dim) ? h[gid * hidden_dim + col] : h_agg[gid * hidden_dim + (col - hidden_dim)];
                                        activation += val * node_w[row * (2 * hidden_dim) + col];
                        }
                        // 3. Final Residual Update: h = h + SiLU(MLP(h, m_agg))
                        h[gid * hidden_dim + row] += silu(activation);
        }
}

kernel void compute_cog(
                device const Node* nodes [[buffer(0)]],
                device atomic_float* cog_sum [[buffer(1)]], // [3] (x, y, z)
                uint gid [[thread_position_in_grid]]
){
                // Atomic sum of all atom positions
                atomic_fetch_add_explicit(&cog_sum[0], nodes[gid].pos.x, memory_order_relaxed);
                atomic_fetch_add_explicit(&cog_sum[1], nodes[gid].pos.y, memory_order_relaxed);
                atomic_fetch_add_explicit(&cog_sum[2], nodes[gid].pos.z, memory_order_relaxed);
}

kernel void apply_cog_normalization(
                device Node* nodes              [[buffer(0)]],
                device const float* cog_sum     [[buffer(1)]],
                constant uint& node_count       [[buffer(2)]],
                uint gid [[thread_position_in_grid]]
){
                // Calculate mean on the fly to avoid CPU round-trips
                float3 mean_pos = float3(cog_sum[0], cog_sum[1], cog_sum[2]) / (float) node_count;
                
                // Centering the molecule
                nodes[gid].pos -= mean_pos;
}
