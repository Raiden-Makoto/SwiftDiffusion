//
//  Diffusion.metal
//  MSLGraphDiffusion
//
//  Created by Raiden-Makoto on 2026-02-10.
//

#include <metal_stdlib>
#include "ShaderTypes.h"

kernel void apply_diffusion(
    device Node* nodes               [[buffer(0)]], // Contains pos_final (after layers)
    device const float* alphas       [[buffer(1)]],
    device const float* alphas_cp    [[buffer(2)]],
    device const float* pos_input    [[buffer(3)]], // Captured x_t from start of step
    constant uint& current_t         [[buffer(4)]],
    constant uint& num_nodes         [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= num_nodes) return;

    float a_t = alphas[current_t];
    float a_bar_t = alphas_cp[current_t];
    
    // 1. Recover x_t (noisy start) and pos_final (current state)
    float3 pos_final = nodes[gid].pos;
    uint pBase = gid * 3;
    float3 x_t = float3(pos_input[pBase], pos_input[pBase + 1], pos_input[pBase + 2]);
    
    // 2. Calculate epsilon exactly like the PyTorch forward pass
    // epsilon_pred = pos_final - pos_input
    float3 epsilon = pos_final - x_t;
    
    // 3. DDPM Reverse Step: x_{t-1} calculation
    float coeff = (1.0f - a_t) / (sqrt(1.0f - a_bar_t) + 1e-7f);
    float damping = 0.410f;
    
    // x_next = (1 / sqrt(a_t)) * (x_t - coeff * epsilon)
    // Note: We use x_t here, NOT pos_final.
    float3 x_next = (1.0f / sqrt(a_t + 1e-7f)) * (x_t - (damping * coeff * epsilon));

    // 4. Update the Node position for the next timestep
    nodes[gid].pos = x_next;
}
