//
//  Diffusion.metal
//  MSLGraphDiffusion
//
//  Created by Raiden-Makoto on 2026-02-10.
//

#include <metal_stdlib>
#include "ShaderTypes.h"

kernel void apply_diffusion(
    device Node* nodes               [[buffer(0)]],
    device const float* alphas       [[buffer(1)]],
    device const float* alphas_cp    [[buffer(2)]],
    device const float* pos_agg      [[buffer(3)]], // This is epsilon_pred
    constant uint& current_t         [[buffer(4)]],
    constant uint& num_nodes         [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= num_nodes) return;

    // 1. Fetch schedule constants for the current timestep
    float a_t = alphas[current_t];
    float a_bar_t = alphas_cp[current_t];
    
    // 2. Extract the predicted noise (displacement) for this node
    uint pBase = gid * 3;
    float3 epsilon = float3(pos_agg[pBase], pos_agg[pBase + 1], pos_agg[pBase + 2]);
    
    // 3. DDPM Reverse Step: x_{t-1} calculation
    // coeff scales the noise prediction based on the current variance
    float coeff = (1.0f - a_t) / (sqrt(1.0f - a_bar_t) + 1e-7f);
    
    // x_next = (1 / sqrt(a_t)) * (x_t - coeff * epsilon)
    float3 x_next = (1.0f / sqrt(a_t + 1e-7f)) * (nodes[gid].pos - (coeff * epsilon));

    // 4. Update the Node position in the global buffer
    nodes[gid].pos = x_next;
}
