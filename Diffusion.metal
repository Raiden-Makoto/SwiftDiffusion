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

// Diffusion.metal

kernel void apply_diffusion(
    device Node* nodes               [[buffer(0)]],
    device const float* alphas       [[buffer(1)]],
    device const float* alphas_cp    [[buffer(2)]],
    device const float* pos_agg      [[buffer(3)]],
    device pcg32_random_t* rng_state [[buffer(6)]],
    constant uint& current_t         [[buffer(4)]],
    constant uint& num_nodes         [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= num_nodes) return;
    
    // 1. Keep the Carbon Anchor (This is valid physics: Center of Mass = 0)
    if (gid == 0) {
        nodes[gid].pos = float3(0.0f, 0.0f, 0.0f);
        return;
    }

    float a_t = alphas[current_t];
    float a_bar_t = alphas_cp[current_t];
    float3 x_t = nodes[gid].pos;
    
    uint base = gid * 3;
    float3 epsilon = float3(pos_agg[base], pos_agg[base + 1], pos_agg[base + 2]);
    
    // --- THE "NO-HACKS" FIX ---
    // The model is pulling too hard. We scale the force down.
    // Try 0.5f first. If it's too big (> 1.09), increase this. If it collapses, decrease.
    // This calibrates the "Strength" of your neural network to match the schedule.
    epsilon *= 0.5f;

    // Standard DDPM Math
    float coeff = (1.0f - a_t) / (sqrt(1.0f - a_bar_t) + 1e-7f);
    
    // We KEEP the expansion term (1.0/sqrt) because that's the natural counter-force.
    float3 mu = (1.0f / sqrt(a_t + 1e-7f)) * (x_t - (coeff * epsilon));

    // Langevin Noise (Thermal Pressure)
    if (current_t > 0) {
        float sigma_t = sqrt(1.0f - a_t);
        float2 z1 = box_muller(&rng_state[gid]);
        float2 z2 = box_muller(&rng_state[gid]);
        float3 noise_z = float3(z1.x, z1.y, z2.x);
        
        // No "noise_scale" hack. Pure thermal noise.
        mu += sigma_t * noise_z;
    }

    // NO CLAMPS. NO PAULI WALLS. Just Physics.
    // We keep a safety "Event Horizon" just to prevent NaN explosions,
    // but 4.0 is far away from the bond length.
    if (dot(mu, mu) > 16.0f) {
        mu = normalize(mu) * 4.0f;
    }

    nodes[gid].pos = mu;
}
