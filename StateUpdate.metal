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

// StateUpdate.metal
// REPLACE your existing 'aggregate' kernel with this SAFE version.
// StateUpdate.metal
//

kernel void aggregate(
    device const float* msg          [[buffer(0)]],
    device const float* trans        [[buffer(1)]],
    device const int2* edge_index    [[buffer(2)]],
    device float* m_agg              [[buffer(3)]],
    device float* pos_agg            [[buffer(4)]],
    constant uint& hidden_dim        [[buffer(5)]],
    constant uint& num_edges         [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    // This thread handles one specific node
    uint node_idx = gid;
    
    // --- CRITICAL: ZEROING LINES REMOVED ---
    // We no longer set m_agg or pos_agg to 0.0f here.
    // Swift handles the clearing; this kernel only ACCUMULATES.

    for (uint e = 0; e < num_edges; e++) {
        // If this edge's destination (i) is our node, aggregate the data
        int target = edge_index[e].x;
        
        if (target == int(node_idx)) {
            // 1. Aggregate Messages (Cleared per-layer in Swift)
            for (uint h = 0; h < hidden_dim; h++) {
                m_agg[node_idx * hidden_dim + h] += msg[e * hidden_dim + h];
            }
            
            // 2. Aggregate Positions (Cleared per-timestep in Swift)
            pos_agg[node_idx * 3 + 0] += trans[e * 3 + 0];
            pos_agg[node_idx * 3 + 1] += trans[e * 3 + 1];
            pos_agg[node_idx * 3 + 2] += trans[e * 3 + 2];
        }
    }
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

// Add this kernel to update node features (h = h + update)
kernel void add_residual(
    device float* accumulator    [[buffer(0)]],
    device const float* update   [[buffer(1)]],
    constant uint& count         [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    accumulator[gid] += update[gid];
}
