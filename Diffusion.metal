//
//  Diffusion.metal
//  MSLGraphDiffusion
//
//  Created by Raiden-Makoto on 2026-02-10.
//

#include <metal_stdlib>
#include "ShaderTypes.h"

using namespace metal;

typedef struct {
    uint64_t state;
    uint64_t inc;
} pcg32_random_t;

// Standard PCG32 logic
uint32_t pcg32_random_r(device pcg32_random_t* rng) {
    uint64_t oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + (rng->inc | 1);
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

// Generates two Gaussian(0,1) floats using Box-Muller
float2 box_muller(device pcg32_random_t* rng) {
    float u1 = (float)pcg32_random_r(rng) / 4294967296.0f;
    float u2 = (float)pcg32_random_r(rng) / 4294967296.0f;
    
    float r = sqrt(-2.0f * log(u1 + 1e-10f));
    float theta = 2.0f * M_PI_F * u2;
    return float2(r * cos(theta), r * sin(theta));
}

kernel void apply_diffusion(
    device Node* nodes               [[buffer(0)]],
    device const float* alphas       [[buffer(1)]],
    device const float* alphas_cp    [[buffer(2)]],
    device const float* pos_agg      [[buffer(3)]],
    device pcg32_random_t* rng_state [[buffer(6)]], // State persists across timesteps
    constant uint& current_t         [[buffer(4)]],
    constant uint& num_nodes         [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= num_nodes) return;

    float a_t = alphas[current_t];
    float a_bar_t = alphas_cp[current_t];
    float3 x_t = nodes[gid].pos;
    
    uint base = gid * 3;
    float3 epsilon = float3(pos_agg[base], pos_agg[base + 1], pos_agg[base + 2]);
    
    // 1. Calculate Mean (mu)
    float coeff = (1.0f - a_t) / (sqrt(1.0f - a_bar_t) + 1e-7f);
    float3 mu = (1.0f / sqrt(a_t + 1e-7f)) * (x_t - (coeff * epsilon));

    // 2. Add Stochastic Variance (sigma * z)
    if (current_t > 0) {
        // Standard DDPM variance: sigma^2 = (1 - alpha)
        float sigma_t = sqrt(1.0f - a_t);
        
        // Generate 3D Gaussian noise
        float2 z1 = box_muller(&rng_state[gid]);
        float2 z2 = box_muller(&rng_state[gid]);
        float3 noise_z = float3(z1.x, z1.y, z2.x);
        
        mu += sigma_t * noise_z;
    }
    mu = clamp(mu, -2.5f, 2.5f);
    nodes[gid].pos = mu;
}
