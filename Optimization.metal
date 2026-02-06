//
//  Optimization.metal
//  MSLGraphDiffusion
//
//  Created by Raiden Makoto on 2026-02-06.
//

kernel void compute_grad_norm_sq(
    device const float* gradients [[buffer(0)]],
    device atomic_float* sum_sq    [[buffer(1)]],
    constant uint& count           [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
){
    if (gid >= count) return;
    float g = gradients[gid];
    atomic_fetch_add_explicit(sum_sq, g * g, memory_order_relaxed);
}

kernel void apply_clipping(
    device float* gradients      [[buffer(0)]],
    device const float* norm_sq  [[buffer(1)]],
    constant float& max_norm     [[buffer(2)]],
    constant uint& count         [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
){
    if (gid >= count) return;
    
    float norm = sqrt(*norm_sq);
    if (norm > max_norm) {
        // Scale factor: max_norm / actual_norm
        gradients[gid] *= (max_norm / (norm + 1e-6f));
    }
}
