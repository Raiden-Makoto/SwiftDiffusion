//
//  Diffusion.metal
//  MSLGraphDiffusion
//
//  Created by Raiden-Makoto on 2026-02-10.
//

#include <metal_stdlib>
#include "ShaderTypes.h"

using namespace metal;

kernel void apply_diffusion(
    device Node* nodes               [[buffer(0)]], // Current x_t positions
    device const float* alphas       [[buffer(1)]],
    device const float* alphas_cp    [[buffer(2)]],
    device const float* pos_agg      [[buffer(3)]], // Total displacement from 4 layers
    constant uint& current_t         [[buffer(4)]],
    constant uint& num_nodes         [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= num_nodes) return;

    // 1. Get schedule variables
    float a_t = alphas[current_t];
    float a_bar_t = alphas_cp[current_t];
    
    // 2. Load current position (x_t) and predicted noise (epsilon)
    float3 x_t = nodes[gid].pos;
    
    // Read the aggregated noise from all 4 layers
    uint base = gid * 3;
    float3 epsilon = float3(pos_agg[base], pos_agg[base + 1], pos_agg[base + 2]);
    
    // 3. DDPM Reverse Step: x_{t-1} calculation
    float coeff = (1.0f - a_t) / (sqrt(1.0f - a_bar_t) + 1e-7f);
    float damping = 1.0f;
    
    // CRITICAL FIX 1: REMOVED (1.0/sqrt(a_t)).
    // This stops the 1.01x expansion per step.
    
    // CRITICAL FIX 2: CHANGED '+' to '-'.
    // We subtract the noise to move towards data.
    float3 x_next = x_t - (damping * coeff * epsilon);

    // 4. Update for next timestep
    nodes[gid].pos = x_next;
}
