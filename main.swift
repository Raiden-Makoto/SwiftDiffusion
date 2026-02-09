//
//  main.swift
//  MSLGraphDiffusion
//
//  Created by Raiden Makoto on 2026-02-07.
//

import Metal
import Foundation

// Simple SIMD3 normalization helper for Swift
// Simple SIMD3 dot product helper for Swift
@inline(__always)
func dot(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> Float {
    return a.x * b.x + a.y * b.y + a.z * b.z
}

@inline(__always)
func normalize(_ v: SIMD3<Float>) -> SIMD3<Float> {
    let len2 = dot(v, v)
    if len2 == 0 { return SIMD3<Float>(0,0,0) }
    let invLen = 1.0 / sqrt(len2)
    return v * Float(invLen)
}

let sectionBreak = String(repeating: "=", count: 50)

// --- CONFIGURATION ---
let hiddenDim = 128
let numLayers = 3
let numTypes = 10

// Hidden dimension from your PyTorch checkpoint
var hDim = UInt32(hiddenDim)

// Input dimensions for the MLPs
var msgInDim = UInt32(2 * hiddenDim + 1) // [hi, hj, radial]
var nodeInDim = UInt32(2 * hiddenDim) // [h, msg_agg]

// --- BUFFER STORAGE ---
var weights: [String: MTLBuffer] = [:]

guard let device = MTLCreateSystemDefaultDevice() else { fatalError("Metal not supported") }

// --- UTILITIES ---
func ZeroInit(_ buffer: MTLBuffer) {
    // Sets Buffer contents to 0
    memset(buffer.contents(), 0, buffer.length)
}


func loadAndVerify(_ name: String, rows: Int, cols: Int, path: String) {
    let expectedBytes = rows * cols * 4 // float32 is 4 bytes
    let buffer = device.makeBuffer(length: expectedBytes, options: .storageModeShared)!
    ZeroInit(buffer) // initialize buffer to 0
    
    // Explicitly targeting the /weights subfolder
    let fileURL = URL(fileURLWithPath: path)
        .appendingPathComponent("weights")
        .appendingPathComponent("\(name).bin")
    
    do {
        let data = try Data(contentsOf: fileURL)
        if data.count == expectedBytes {
            buffer.contents().copyMemory(from: (data as NSData).bytes, byteCount: data.count)
            weights[name] = buffer
            //print("VERIFIED: \(name).bin (\(data.count) bytes)")
        } else if data.count < expectedBytes {
            //print("PARTIAL LOAD: \(data.count)/\(expectedBytes) loaded. Padding with zeros")
            weights[name] = buffer
        } else {
            print("SIZE MISMATCH: \(name) - File: \(data.count), Buffer Needs: \(expectedBytes)")
        }
    } catch {
        print("NOT FOUND: \(name).bin at \(fileURL.path)")
    }
}

// --- EXECUTION CHUNK ---

// 1. Get the local path
let datapath = URL(fileURLWithPath: #filePath).deletingLastPathComponent().path
print(sectionBreak)
print("LOADING MODEL WEIGHTS")

loadAndVerify("embedding.weight", rows: numTypes, cols: hiddenDim, path: datapath)

// 3. Timestep MLP (2-stage)
loadAndVerify("timestep_mlp.0.weight", rows: hiddenDim, cols: hiddenDim, path: datapath)
loadAndVerify("timestep_mlp.0.bias",   rows: 1,         cols: hiddenDim, path: datapath)
loadAndVerify("timestep_mlp.2.weight", rows: hiddenDim, cols: hiddenDim, path: datapath)
loadAndVerify("timestep_mlp.2.bias",   rows: 1,         cols: hiddenDim, path: datapath)

// 4. Recursive Layers (0-3)
for i in 0..<numLayers {
    // Message MLP
    loadAndVerify("layers.\(i).message_mlp.0.weight", rows: hiddenDim, cols: 2 * hiddenDim + 1, path: datapath)
    loadAndVerify("layers.\(i).message_mlp.0.bias",   rows: 1,         cols: hiddenDim, path: datapath)
    loadAndVerify("layers.\(i).message_mlp.2.weight", rows: hiddenDim, cols: hiddenDim, path: datapath)
    loadAndVerify("layers.\(i).message_mlp.2.bias",   rows: 1,         cols: hiddenDim, path: datapath)
    
    // Coordination MLP
    loadAndVerify("layers.\(i).coord_mlp.0.weight", rows: hiddenDim, cols: hiddenDim, path: datapath)
    loadAndVerify("layers.\(i).coord_mlp.0.bias",   rows: 1,         cols: hiddenDim, path: datapath)
    loadAndVerify("layers.\(i).coord_mlp.2.weight", rows: 1,         cols: hiddenDim, path: datapath)
    loadAndVerify("layers.\(i).coord_mlp.2.bias",   rows: 1,         cols: 1,         path: datapath)
    
    // Node MLP
    loadAndVerify("layers.\(i).node_mlp.0.weight", rows: hiddenDim, cols: 2 * hiddenDim, path: datapath)
    loadAndVerify("layers.\(i).node_mlp.0.bias",   rows: 1,         cols: hiddenDim, path: datapath)
    loadAndVerify("layers.\(i).node_mlp.2.weight", rows: hiddenDim, cols: hiddenDim, path: datapath)
    loadAndVerify("layers.\(i).node_mlp.2.bias",   rows: 1,         cols: hiddenDim, path: datapath)
}

print("Chunk Verification: Loaded \(weights.count) total weight buffers.")
print(sectionBreak)

// --- DATA PREPARATION ---
var numNodes = 5 // for Methane only
var numEdges = 8 // for methane, 4 bonds x2 (bidirectional)
let atomTypes: [Int32] = [6, 1, 1, 1, 1] // Example: Methane (C, H, H, H, H)

var nNodes = UInt32(numNodes)
// Output dimension for the Coordinate MLP final stage
var outDim1 = UInt32(1)

// Atom types and initial noise (The starting point)
let nodeBuf = device.makeBuffer(length: numNodes * MemoryLayout<Node>.stride, options: .storageModeShared)!
let typeBuf = device.makeBuffer(length: numNodes * MemoryLayout<Int32>.stride, options: .storageModeShared)!
let edgeBuf = device.makeBuffer(length: numEdges * MemoryLayout<SIMD2<Int32>>.stride, options: .storageModeShared)!
let hBuf = device.makeBuffer(length: numNodes * hiddenDim * 4, options: .storageModeShared)!

// Methane (CH4) positions in angstroms (approx tetrahedral geometry)
let a: Float = 1.09
let carbonPos = SIMD3<Float>(0.0, 0.0, 0.0)
// Four hydrogens at the corners of a tetrahedron around the origin
let h1 = a * normalize(SIMD3<Float>( 1,  1,  1))
let h2 = a * normalize(SIMD3<Float>(-1, -1,  1))
let h3 = a * normalize(SIMD3<Float>(-1,  1, -1))
let h4 = a * normalize(SIMD3<Float>( 1, -1, -1))
let positions: [SIMD3<Float>] = [carbonPos, h1, h2, h3, h4]

// Atom types in CHHHH order
let atomTypesCH4: [Int32] = [6, 1, 1, 1, 1]
// Build Node array matching device layout
var hostNodes = [Node](repeating: Node(pos: SIMD3<Float>(0,0,0), atomType: 0), count: numNodes)
for i in 0..<numNodes {
    hostNodes[i] = Node(pos: positions[i], atomType: Float(atomTypesCH4[i]))
}
// Copy to device buffers
nodeBuf.contents().copyMemory(from: hostNodes, byteCount: hostNodes.count * MemoryLayout<Node>.stride)
typeBuf.contents().copyMemory(from: atomTypesCH4, byteCount: atomTypesCH4.count * MemoryLayout<Int32>.stride)

// Bidirectional edges: C<->H for each H
let edges: [SIMD2<Int32>] = [
    SIMD2(0,1), SIMD2(1,0),
    SIMD2(0,2), SIMD2(2,0),
    SIMD2(0,3), SIMD2(3,0),
    SIMD2(0,4), SIMD2(4,0)
]
precondition(edges.count == numEdges, "edges.count must equal numEdges")
edgeBuf.contents().copyMemory(from: edges, byteCount: edges.count * MemoryLayout<SIMD2<Int32>>.stride)

let currentStep: Float = 501.0 // Example timestep index
var sinData = [Float](repeating: 0, count: hiddenDim)
let halfDim = hiddenDim / 2
let exponentBase = log(10000.0) / Float(halfDim - 1)

for i in 0..<halfDim {
    let freq = exp(Float(i) * -exponentBase)
    let arg = currentStep * freq
    sinData[i] = sin(arg)
    sinData[i + halfDim] = cos(arg)
}

// Create the buffer from the CPU data
let tSinBuf = device.makeBuffer(bytes: sinData, length: hiddenDim * 4, options: .storageModeShared)!
let tProcessedBuf = device.makeBuffer(length: hiddenDim * 4, options: .storageModeShared)! // Final t_emb after MLP

// --- DISPATCH ---
guard let library = device.makeDefaultLibrary() else { fatalError("Couldn't load default compute library.") }
var pipeline: [String: MTLComputePipelineState] = [:]

do {
    // Pipeline for atom embedding lookup
    let embedFunction = library.makeFunction(name: "embed_atoms")!
    pipeline["embed_atoms"] = try device.makeComputePipelineState(function: embedFunction)
    // Pipeline for timestep injection conditioning
    let injectFunction = library.makeFunction(name: "inject_timestamp")!
    pipeline["inject_timestamp"] = try device.makeComputePipelineState(function: injectFunction)
    print("Embedding Pipelines: \(pipeline.keys.joined(separator: ", "))")
} catch {
    fatalError("Failed to create compute pipeline states: \(error)")
}

let commandQueue = device.makeCommandQueue()!
let cb = commandQueue.makeCommandBuffer()!

// Allocate the Buffers for the Main MLP stuff HERE
// Temporary buffer for MLP stages (used to hold intermediate activations)
let tempH = device.makeBuffer(length: max(numNodes, numEdges) * hiddenDim * 4, options: .storageModeShared)!

// Message buffers
let msgBuf = device.makeBuffer(length: numEdges * hiddenDim * 4, options: .storageModeShared)!
let msgAggBuf = device.makeBuffer(length: numNodes * hiddenDim * 4, options: .storageModeShared)!

// Coordinate & Translation buffers
let coordScalarBuf = device.makeBuffer(length: numEdges * 1 * 4, options: .storageModeShared)!
let transBuf = device.makeBuffer(length: numEdges * 3 * MemoryLayout<Float>.stride, options: .storageModeShared)!
let posAggBuf = device.makeBuffer(length: numNodes * 3 * MemoryLayout<Float>.stride, options: .storageModeShared)!

// Node update residual
let hUpdateBuf = device.makeBuffer(length: numNodes * hiddenDim * 4, options: .storageModeShared)!

print("EGNNLayer buffers allocated for \(numNodes) nodes and \(numEdges) edges.")

let blit = cb.makeBlitCommandEncoder()!
blit.fill(buffer: msgAggBuf, range: 0..<msgAggBuf.length, value: 0)
blit.fill(buffer: posAggBuf, range: 0..<posAggBuf.length, value: 0)
blit.endEncoding()

let enc = cb.makeComputeCommandEncoder()!

// A. Embed raw atom types
enc.setComputePipelineState(pipeline["embed_atoms"]!)
enc.setBuffer(typeBuf, offset: 0, index: 0)
enc.setBuffer(weights["embedding.weight"], offset: 0, index: 1)
enc.setBuffer(hBuf, offset: 0, index: 2)
enc.setBytes(&hDim, length: 4, index: 3)
enc.setBytes(&nNodes, length: 4, index: 4)
enc.dispatchThreads(MTLSize(width: numNodes, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

// B. Inject Timestep into Hidden States
enc.setComputePipelineState(pipeline["inject_timestamp"]!)
enc.setBuffer(hBuf, offset: 0, index: 0)
enc.setBuffer(tProcessedBuf, offset: 0, index: 1)
enc.setBytes(&hDim, length: 4, index: 2)
enc.setBytes(&nNodes, length: 4, index: 3)
enc.dispatchThreads(MTLSize(width: numNodes, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

print("Features embedded and conditioned on timestep.")
print(sectionBreak)

do {
    let linearFunc = library.makeFunction(name: "linear_layer")!
    pipeline["linear_layer"] = try device.makeComputePipelineState(function: linearFunc)
} catch {
    fatalError("Failed to create compute pipeline states: \(error)")
}

var applySiLU: Bool = true
var noSiLU: Bool = false

print("Timstep MLP Activated")
// Linear (128, 128) + SiLU
enc.setComputePipelineState(pipeline["linear_layer"]!)
enc.setBuffer(tSinBuf, offset: 0, index: 0) // Raw Sinusoidal input
enc.setBuffer(weights["timestep_mlp.0.weight"], offset: 0, index: 1)
enc.setBuffer(weights["timestep_mlp.0.bias"], offset: 0, index: 2)
enc.setBuffer(tProcessedBuf, offset: 0, index: 3)
enc.setBytes(&hDim, length: 4, index: 4) // inDim
enc.setBytes(&hDim, length: 4, index: 5) // outDim
enc.setBytes(&applySiLU, length: 1, index: 6)
enc.dispatchThreads(
    MTLSize(width: hiddenDim, height: 1, depth: 1),
    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
)
// Linear (128, 128) - No Activation
enc.setBuffer(tProcessedBuf, offset: 0, index: 0)
enc.setBuffer(weights["timestep_mlp.2.weight"], offset: 0, index: 1)
enc.setBuffer(weights["timestep_mlp.2.bias"], offset: 0, index: 2)
enc.setBuffer(tProcessedBuf, offset: 0, index: 3) // Overwrite with final result
enc.setBytes(&noSiLU, length: 1, index: 6) // apply_silu = false
enc.dispatchThreads(
    MTLSize(width: hiddenDim, height: 1, depth: 1),
    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
)
print("Timstep MLP Complete")
print(sectionBreak)
print("Three-Layer EGNNLayer Loop Activated")

let layerKernels = [
    "compute_message",
    "compute_displacement",
    "compute_node",
    "aggregate",
    "apply_update"
]

for name in layerKernels{
    pipeline[name] = try! device.makeComputePipelineState(function: library.makeFunction(name: name)!)
}

let geometryKernels = [
    "cog_normalization",
    "compute_cog"
]

for name in geometryKernels{
    pipeline[name] = try! device.makeComputePipelineState(function: library.makeFunction(name: name)!)
}

for i in 0..<3 {
    print("\tCurrently in Layer \(i)...")
    // 1. MESSAGE MLP
    enc.setComputePipelineState(pipeline["compute_message"]!)
    enc.setBuffer(hBuf, offset: 0, index: 0);
    enc.setBuffer(nodeBuf, offset: 0, index: 1);
    enc.setBuffer(edgeBuf, offset: 0, index: 2)
    enc.setBuffer(weights["layers.\(i).message_mlp.0.weight"], offset: 0, index: 3);
    enc.setBuffer(weights["layers.\(i).message_mlp.0.bias"], offset: 0, index: 4)
    enc.setBuffer(msgBuf, offset: 0, index: 5);
    enc.setBytes(&hDim, length: 4, index: 6)
    enc.dispatchThreads(
        MTLSize(width: numEdges, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
    )
    // Linear -> SiLU -> Linear
    enc.setComputePipelineState(pipeline["linear_layer"]!) // Message Stage 2
    enc.setBuffer(msgBuf, offset: 0, index: 0);
    enc.setBuffer(weights["layers.\(i).message_mlp.2.weight"], offset: 0, index: 1);
    enc.setBuffer(weights["layers.\(i).message_mlp.2.bias"], offset: 0, index: 2)
    enc.setBuffer(msgBuf, offset: 0, index: 3);
    enc.setBytes(&hDim, length: 4, index: 4);
    enc.setBytes(&hDim, length: 4, index: 5);
    enc.setBytes(&noSiLU, length: 1, index: 6)
    enc.dispatchThreads(
        MTLSize(width: numEdges, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
    )
    // 2. COORDINATE MLP & DISPLACEMENT
    enc.setComputePipelineState(pipeline["linear_layer"]!) // Coord Stage 1
    enc.setBuffer(msgBuf, offset: 0, index: 0);
    enc.setBuffer(weights["layers.\(i).coord_mlp.0.weight"], offset: 0, index: 1);
    enc.setBuffer(weights["layers.\(i).coord_mlp.0.bias"], offset: 0, index: 2)
    enc.setBuffer(tempH, offset: 0, index: 3);
    enc.setBytes(&hDim, length: 4, index: 4);
    enc.setBytes(&hDim, length: 4, index: 5);
    enc.setBytes(&applySiLU, length: 1, index: 6)
    enc.dispatchThreads(
        MTLSize(width: numEdges, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
    )
    // Linear -> SiLU -> Linear
    enc.setBuffer(tempH, offset: 0, index: 0);
    enc.setBuffer(weights["layers.\(i).coord_mlp.2.weight"], offset: 0, index: 1);
    enc.setBuffer(weights["layers.\(i).coord_mlp.2.bias"], offset: 0, index: 2)
    enc.setBuffer(coordScalarBuf, offset: 0, index: 3);
    enc.setBytes(&hDim, length: 4, index: 4);
    enc.setBytes(&outDim1, length: 4, index: 5);
    enc.setBytes(&noSiLU, length: 1, index: 6)
    enc.dispatchThreads(
        MTLSize(width: numEdges, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
    )
    // Compute Displacement
    enc.setComputePipelineState(pipeline["compute_displacement"]!)
    enc.setBuffer(coordScalarBuf, offset: 0, index: 0);
    enc.setBuffer(nodeBuf, offset: 0, index: 1);
    enc.setBuffer(edgeBuf, offset: 0, index: 2);
    enc.setBuffer(transBuf, offset: 0, index: 3)
    enc.setBytes(&numEdges, length: 4, index: 4)
    enc.dispatchThreads(
        MTLSize(width: numEdges, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
    )
    // 3. AGGREGATE
    enc.setComputePipelineState(pipeline["aggregate"]!)
    enc.setBuffer(msgBuf, offset: 0, index: 0);
    enc.setBuffer(transBuf, offset: 0, index: 1);
    enc.setBuffer(edgeBuf, offset: 0, index: 2)
    enc.setBuffer(msgAggBuf, offset: 0, index: 3);
    enc.setBuffer(posAggBuf, offset: 0, index: 4);
    enc.setBytes(&hDim, length: 4, index: 5)
    enc.setBytes(&numEdges, length: 4, index: 6)
    enc.dispatchThreads(
        MTLSize(width: numEdges, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
    )
    // 4. NODE MLP
    enc.setComputePipelineState(pipeline["compute_node"]!) // SiLU is built in
    enc.setBuffer(hBuf, offset: 0, index: 0);
    enc.setBuffer(msgAggBuf, offset: 0, index: 1);
    enc.setBuffer(weights["layers.\(i).node_mlp.0.weight"], offset: 0, index: 2);
    enc.setBuffer(weights["layers.\(i).node_mlp.0.bias"], offset: 0, index: 3)
    enc.setBuffer(tempH, offset: 0, index: 4);
    enc.setBytes(&hDim, length: 4, index: 5)
    enc.dispatchThreads(
        MTLSize(width: numNodes, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
    )
    // Linear -> SiLU -> Linear
    enc.setComputePipelineState(pipeline["linear_layer"]!)
    enc.setBuffer(tempH, offset: 0, index: 0);
    enc.setBuffer(weights["layers.\(i).node_mlp.2.weight"], offset: 0, index: 1);
    enc.setBuffer(weights["layers.\(i).node_mlp.2.bias"], offset: 0, index: 2)
    enc.setBuffer(hUpdateBuf, offset: 0, index: 3);
    enc.setBytes(&hDim, length: 4, index: 4); enc.setBytes(&hDim, length: 4, index: 5);
    enc.setBytes(&noSiLU, length: 1, index: 6)
    enc.dispatchThreads(
        MTLSize(width: numNodes, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
    )
    // 5. RESIDUAL CONNECTIONS
    enc.setComputePipelineState(pipeline["apply_update"]!)
    enc.setBuffer(nodeBuf, offset: 0, index: 0);
    enc.setBuffer(hBuf, offset: 0, index: 1);
    enc.setBuffer(hUpdateBuf, offset: 0, index: 2);
    enc.setBuffer(posAggBuf, offset: 0, index: 3);
    enc.setBytes(&hDim, length: 4, index: 4)
    enc.dispatchThreads(
        MTLSize(width: numNodes, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
    )
}

enc.endEncoding()
cb.commit()
cb.waitUntilCompleted()

print("Three Layer EGNNLayer Stack Complete")
print(sectionBreak)

// Apply Center of Gravity Normalization
let cogBuf = device.makeBuffer(length: 3 * MemoryLayout<Float>.stride, options: .storageModeShared)!
let cogCB = commandQueue.makeCommandBuffer()!

// Reset the CoG sum buffer to zero
let cogBlit = cogCB.makeBlitCommandEncoder()!
cogBlit.fill(buffer: cogBuf, range: 0..<cogBuf.length, value: 0)
cogBlit.endEncoding()

let cogEnc = cogCB.makeComputeCommandEncoder()!

// Compute the sum of all positions
cogEnc.setComputePipelineState(pipeline["compute_cog"]!)
cogEnc.setBuffer(nodeBuf, offset: 0, index: 0)
cogEnc.setBuffer(cogBuf, offset: 0, index: 1)
cogEnc.setBytes(&nNodes, length: 4, index: 2)
cogEnc.dispatchThreads(
    MTLSize(width: numNodes, height: 1, depth: 1),
    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
)

// Subtract the centroid from every node
cogEnc.setComputePipelineState(pipeline["cog_normalization"]!)
cogEnc.setBuffer(nodeBuf, offset: 0, index: 0)
cogEnc.setBuffer(cogBuf, offset: 0, index: 1)
cogEnc.setBytes(&nNodes, length: 4, index: 2)
cogEnc.dispatchThreads(
    MTLSize(width: numNodes, height: 1, depth: 1),
    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
)

cogEnc.endEncoding()
cogCB.commit()
cogCB.waitUntilCompleted()

print("Molecule centered at origin.")
print(sectionBreak)
print("Final Molecular Coordinates (Angstroms)")

// Access the raw pointer from the GPU buffer
let pointer = nodeBuf.contents().bindMemory(to: Node.self, capacity: numNodes)

// Iterate and print each node's position
for i in 0..<numNodes {
    let node = pointer[i]
    let p = node.pos
    let type = Int(node.atomType)
    // Format for easy reading: "Atom [Type]: (x, y, z)"
    let name = (type == 6) ? "Carbon" : "Hydrogen"
    print(String(format: "Node %d [%@]: (%7.4f, %7.4f, %7.4f)", i, name, p.x, p.y, p.z))
}

// Sanity Check: Center of Gravity calculation on CPU
var sum = SIMD3<Float>(0, 0, 0)
for i in 0..<numNodes { sum += pointer[i].pos }
let avg = sum / Float(numNodes)
print(String(format: "\nCalculated Centroid: (%7.4f, %7.4f, %7.4f)", avg.x, avg.y, avg.z))
print(sectionBreak)
