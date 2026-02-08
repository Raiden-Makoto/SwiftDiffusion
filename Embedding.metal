//
//  Embedding.metal
//  MSLGraphDiffusion
//
//  Created by Raiden-Makoto on 2026-02-06.
//

#include "ShaderTypes.h"
#include <metal_stdlib>

using namespace metal;

kernel void embed_atoms(
                        device const int * atom_types [[buffer(0)]],
                        device const float* embed_table [[buffer(1)]],
                        device float* h_out [[buffer(2)]],
                        constant uint& hidden_dim [[buffer(3)]],
                        constant uint& node_count [[buffer(4)]],
                        uint gid [[thread_position_in_grid]]
                        ){
    if (gid >= node_count){ return; } // avoid accessing outside the buffer
    int atom_type = atom_types[gid];
    for (uint i=0; i<hidden_dim; i++){
        h_out[gid * hidden_dim + i] = embed_table[atom_type * hidden_dim + i];
    }
}

kernel void inject_timestamp(
                             device float* h                  [[buffer(0)]], // [N * 128]
                             device const float* t_emb        [[buffer(1)]], // [128] (After MLP processing)
                             constant uint& hidden_dim        [[buffer(2)]],
                             constant uint& node_count        [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]
                             ){
    if (gid >= node_count){ return; } // avoid accessing outside the buffer
    for (uint i = 0; i < hidden_dim; i++) {
        h[gid * hidden_dim + i] += t_emb[i];
    }
}
