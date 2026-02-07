//
//  main.swift
//  MSLGraphDiffusion
//
//  Created by Raiden Makoto on 2026-02-04.
//

import Metal
import Foundation

// --- UTILITIES ---

func KaimingInit(_ buffer: MTLBuffer, count: Int, fanIn: Int) {
    let ptr = buffer.contents().bindMemory(to: Float.self, capacity: count)
    let std = sqrt(2.0 / Double(fanIn))
    for i in 0..<count {
        let u1 = max(Float.leastNonzeroMagnitude, Float.random(in: 0..<1))
        let u2 = Float.random(in: 0..<1)
        let z = Float(sqrt(-2.0 * log(Double(u1))) * cos(2.0 * Double.pi * Double(u2)))
        ptr[i] = z * Float(std)
    }
}

func ZeroInit(_ buffer: MTLBuffer) {
    memset(buffer.contents(), 0, buffer.length)
}

func SaveModelWeights(buffers: [(String, MTLBuffer)], path: String) {
    for (name, buffer) in buffers {
        let fileURL = URL(fileURLWithPath: path).appendingPathComponent("\(name).bin")
        let data = Data(bytes: buffer.contents(), count: buffer.length)
        do { try data.write(to: fileURL); print("Saved \(name)") } catch { print("Save Fail: \(error)") }
    }
}

// --- INITIALIZATION ---

guard let device = MTLCreateSystemDefaultDevice() else { fatalError() }
let commandQueue = device.makeCommandQueue()!
let loader = QM9Loader(device: device)
let datapath = URL(fileURLWithPath: #filePath).deletingLastPathComponent().path
try! loader.load(from: datapath)

// --- DATASET SIZING & PHYSICAL ALIGNMENT ---

let nodeStride = MemoryLayout<Node>.stride
let edgeStride = MemoryLayout<SIMD2<Int32>>.stride

let totalNodesDataset = loader.nodeBuffer!.length / nodeStride
let totalEdgesDataset = loader.edgeBuffer!.length / edgeStride

var trainNodes = Int(Double(totalNodesDataset) * 0.8)
while (trainNodes * nodeStride) % 32 != 0 { trainNodes -= 1 }
let alignedByteOffset = trainNodes * nodeStride

var trainEdges = Int(Double(totalEdgesDataset) * 0.8)
while (trainEdges * edgeStride) % 16 != 0 { trainEdges -= 1 }
let valEdges = totalEdgesDataset - trainEdges
let alignedEdgeByteOffset = trainEdges * edgeStride

let hiddenDim = 64

// --- BUFFER ALLOCATIONS ---

let nodeBufT = device.makeBuffer(length: trainNodes * nodeStride, options: .storageModeShared)!
let edgeBufT = device.makeBuffer(length: trainEdges * edgeStride, options: .storageModeShared)!

let hT = device.makeBuffer(length: trainNodes * hiddenDim * 4, options: .storageModeShared)!
let msgT = device.makeBuffer(length: trainEdges * hiddenDim * 4, options: .storageModeShared)!
let aggT = device.makeBuffer(length: trainNodes * hiddenDim * 4, options: .storageModeShared)!
let posT = device.makeBuffer(length: trainNodes * 3 * 4, options: .storageModeShared)!
let noiseT = device.makeBuffer(length: trainNodes * 3 * 4, options: .storageModeShared)!

let msgInputT = device.makeBuffer(length: trainEdges * (2 * hiddenDim + 1) * 4, options: .storageModeShared)!
let preActivT = device.makeBuffer(length: trainEdges * hiddenDim * 4, options: .storageModeShared)!
let nodeActivT = device.makeBuffer(length: trainNodes * 2 * hiddenDim * 4, options: .storageModeShared)!
let preActivNodeT = device.makeBuffer(length: trainNodes * hiddenDim * 4, options: .storageModeShared)!

let weights = device.makeBuffer(length: hiddenDim * (2 * hiddenDim + 1) * 4, options: .storageModeShared)!
let nodeW = device.makeBuffer(length: hiddenDim * (2 * hiddenDim) * 4, options: .storageModeShared)!
let bias = device.makeBuffer(length: hiddenDim * 4, options: .storageModeShared)!
let coordW = device.makeBuffer(length: hiddenDim * 4, options: .storageModeShared)!
let coordB = device.makeBuffer(length: 4, options: .storageModeShared)!
let embedTable = device.makeBuffer(length: 10 * hiddenDim * 4, options: .storageModeShared)!

let tEmbBuf = device.makeBuffer(length: hiddenDim * 4, options: .storageModeShared)!
let cogT = device.makeBuffer(length: 12, options: .storageModeShared)!

// --- DATA MIGRATION ---

let setupCB = commandQueue.makeCommandBuffer()!
let blit = setupCB.makeBlitCommandEncoder()!
blit.copy(from: loader.nodeBuffer!, sourceOffset: 0, to: nodeBufT, destinationOffset: 0, size: nodeBufT.length)
blit.copy(from: loader.edgeBuffer!, sourceOffset: 0, to: edgeBufT, destinationOffset: 0, size: edgeBufT.length)
blit.endEncoding(); setupCB.commit(); setupCB.waitUntilCompleted()

// --- WEIGHT INITIALIZATION ---

KaimingInit(weights, count: hiddenDim * (2 * hiddenDim + 1), fanIn: (2 * hiddenDim + 1))
KaimingInit(nodeW, count: hiddenDim * 2 * hiddenDim, fanIn: 2 * hiddenDim)
KaimingInit(coordW, count: hiddenDim, fanIn: hiddenDim)
KaimingInit(embedTable, count: 10 * hiddenDim, fanIn: 10)
ZeroInit(bias); ZeroInit(coordB)

let nPtrT = noiseT.contents().bindMemory(to: Float.self, capacity: trainNodes * 3)
for i in 0..<(trainNodes * 3) { nPtrT[i] = Float.random(in: -1.0...1.0) }

let lib = device.makeDefaultLibrary()!
let names = ["embed_atoms", "inject_timestep", "compute_message", "aggregate_message", "update_coords", "apply_updates", "compute_cog", "apply_cog_normalization"]
var p = [String: MTLComputePipelineState]()
for n in names { p[n] = try! device.makeComputePipelineState(function: lib.makeFunction(name: n)!) }

// --- DIFFUSION GENERATION PASS ---

let numSteps = 500 // Loop over timesteps
var hDim = UInt32(hiddenDim)
var trainNodeCount = UInt32(trainNodes)

// Prepare generation seed (Start from pure Noise)
let preCB = commandQueue.makeCommandBuffer()!
let preBlit = preCB.makeBlitCommandEncoder()!
preBlit.copy(from: noiseT, sourceOffset: 0, to: posT, destinationOffset: 0, size: posT.length)
preBlit.endEncoding(); preCB.commit(); preCB.waitUntilCompleted()

// Loop T -> 0 to generate the molecule structure
for t in (1...numSteps).reversed() {
    let cb = commandQueue.makeCommandBuffer()!
    
    // Reset temporary aggregators
    let bEnc = cb.makeBlitCommandEncoder()!
    [msgT, aggT].forEach { bEnc.fill(buffer: $0, range: 0..<$0.length, value: 0) }
    bEnc.fill(buffer: cogT, range: 0..<12, value: 0)
    
    // Generate Sinusoidal Embedding for current step t
    let tPtr = tEmbBuf.contents().bindMemory(to: Float.self, capacity: hiddenDim)
    for i in 0..<hiddenDim/2 {
        let freq = exp(Float(i) * -log(10000.0) / Float(hiddenDim/2 - 1))
        tPtr[i] = sin(Float(t) * freq)
        tPtr[i + hiddenDim/2] = cos(Float(t) * freq)
    }
    bEnc.endEncoding()

    let enc = cb.makeComputeCommandEncoder()!
    
    // 1. Initial Embedding
    enc.setComputePipelineState(p["embed_atoms"]!)
    enc.setBuffer(nodeBufT, offset: 0, index: 0); enc.setBuffer(embedTable, offset: 0, index: 1); enc.setBuffer(hT, offset: 0, index: 2); enc.setBytes(&hDim, length: 4, index: 3)
    var numTypes: UInt32 = 10; enc.setBytes(&trainNodeCount, length: 4, index: 4); enc.setBytes(&numTypes, length: 4, index: 5)
    enc.dispatchThreads(MTLSize(width: trainNodes, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

    // 2. Inject Timestep Information into Node Features
    enc.setComputePipelineState(p["inject_timestep"]!)
    enc.setBuffer(hT, offset: 0, index: 0); enc.setBuffer(tEmbBuf, offset: 0, index: 1); enc.setBytes(&hDim, length: 4, index: 2); enc.setBytes(&trainNodeCount, length: 4, index: 3)
    enc.dispatchThreads(MTLSize(width: trainNodes, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

    // 3. EGNN refinement layers
    for _ in 0..<4 {
        enc.setComputePipelineState(p["compute_message"]!); enc.setBuffer(nodeBufT, offset: 0, index: 0); enc.setBuffer(hT, offset: 0, index: 1); enc.setBuffer(edgeBufT, offset: 0, index: 2); enc.setBuffer(weights, offset: 0, index: 3); enc.setBuffer(bias, offset: 0, index: 4); enc.setBuffer(msgT, offset: 0, index: 5); enc.setBytes(&hDim, length: 4, index: 6); enc.setBuffer(msgInputT, offset: 0, index: 7); enc.setBuffer(preActivT, offset: 0, index: 8)
        enc.dispatchThreads(MTLSize(width: trainEdges, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        
        enc.setComputePipelineState(p["aggregate_message"]!); enc.setBuffer(msgT, offset: 0, index: 0); enc.setBuffer(edgeBufT, offset: 0, index: 1); enc.setBuffer(aggT, offset: 0, index: 2); enc.setBytes(&hDim, length: 4, index: 3)
        enc.dispatchThreads(MTLSize(width: trainEdges, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        
        enc.setComputePipelineState(p["update_coords"]!); enc.setBuffer(nodeBufT, offset: 0, index: 0); enc.setBuffer(msgT, offset: 0, index: 1); enc.setBuffer(edgeBufT, offset: 0, index: 2); enc.setBuffer(coordW, offset: 0, index: 3); enc.setBuffer(coordB, offset: 0, index: 4); enc.setBuffer(posT, offset: 0, index: 5); enc.setBytes(&hDim, length: 4, index: 6)
        enc.dispatchThreads(MTLSize(width: trainEdges, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        
        enc.setComputePipelineState(p["apply_updates"]!); enc.setBuffer(nodeBufT, offset: 0, index: 0); enc.setBuffer(hT, offset: 0, index: 1); enc.setBuffer(posT, offset: 0, index: 2); enc.setBuffer(aggT, offset: 0, index: 3); enc.setBuffer(nodeW, offset: 0, index: 4); enc.setBuffer(bias, offset: 0, index: 5); enc.setBytes(&hDim, length: 4, index: 6); enc.setBuffer(nodeActivT, offset: 0, index: 7); enc.setBuffer(preActivNodeT, offset: 0, index: 8)
        enc.dispatchThreads(MTLSize(width: trainNodes, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

        // Center Molecule per layer
        enc.setComputePipelineState(p["compute_cog"]!); enc.setBuffer(nodeBufT, offset: 0, index: 0); enc.setBuffer(cogT, offset: 0, index: 1)
        enc.dispatchThreads(MTLSize(width: trainNodes, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.setComputePipelineState(p["apply_cog_normalization"]!); enc.setBuffer(nodeBufT, offset: 0, index: 0); enc.setBuffer(cogT, offset: 0, index: 1); enc.setBytes(&trainNodeCount, length: 4, index: 2)
        enc.dispatchThreads(MTLSize(width: trainNodes, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

    enc.endEncoding()
    cb.commit(); cb.waitUntilCompleted()
    if t % 50 == 0 { print("Diffusion Step: \(t)") }
}

print("Molecule generation complete. Final coordinates in posT.")
SaveModelWeights(buffers: [("generated_molecule", posT)], path: datapath)
