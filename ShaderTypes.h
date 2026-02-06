//
//  ShaderTypes.h
//  MSLGraphDiffusion
//
//  Created by Raiden-Makoto on 2026-02-06.
//

#include <metal_stdlib>
using namespace metal;

struct Node {
    float3 pos;
    float atomType;
};

struct GraphData {
    uint nodeCount;
    uint edgeCount;
};
