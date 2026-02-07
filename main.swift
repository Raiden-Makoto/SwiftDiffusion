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

// --- DATASET SIZING & BOUNDS-SAFE SPLIT ---

let nodeStride = MemoryLayout<Node>.stride
let edgeStride = MemoryLayout<SIMD2<Int32>>.stride

let totalNodesDataset = loader.nodeBuffer!.length / nodeStride
let totalEdgesDataset = loader.edgeBuffer!.length / edgeStride

let splitNodes = Int(Double(totalNodesDataset) * 0.8)
let splitEdges = Int(Double(totalEdgesDataset) * 0.8)

let rawByteOffset = splitNodes * nodeStride
let alignedByteOffset = (rawByteOffset + 15) & ~15
let trainNodes = alignedByteOffset / nodeStride
let valNodes = totalNodesDataset - trainNodes

let rawEdgeByteOffset = splitEdges * edgeStride
let alignedEdgeByteOffset = (rawEdgeByteOffset + 15) & ~15
let trainEdges = alignedEdgeByteOffset / edgeStride
let valEdges = totalEdgesDataset - trainEdges

let hiddenDim = 64

// --- BUFFER ALLOCATIONS ---

let nodeBufT = device.makeBuffer(length: trainNodes * nodeStride, options: .storageModeShared)!
let nodeBufV = device.makeBuffer(length: valNodes * nodeStride, options: .storageModeShared)!
let edgeBufT = device.makeBuffer(length: trainEdges * edgeStride, options: .storageModeShared)!
let edgeBufV = device.makeBuffer(length: valEdges * edgeStride, options: .storageModeShared)!

let hT = device.makeBuffer(length: trainNodes * hiddenDim * 4, options: .storageModeShared)!
let hV = device.makeBuffer(length: valNodes * hiddenDim * 4, options: .storageModeShared)!
let msgT = device.makeBuffer(length: trainEdges * hiddenDim * 4, options: .storageModeShared)!
let msgV = device.makeBuffer(length: valEdges * hiddenDim * 4, options: .storageModeShared)!
let aggT = device.makeBuffer(length: trainNodes * hiddenDim * 4, options: .storageModeShared)!
let aggV = device.makeBuffer(length: valNodes * hiddenDim * 4, options: .storageModeShared)!
let posT = device.makeBuffer(length: trainNodes * 3 * 4, options: .storageModeShared)!
let posV = device.makeBuffer(length: valNodes * 3 * 4, options: .storageModeShared)!
let noiseT = device.makeBuffer(length: trainNodes * 3 * 4, options: .storageModeShared)!
let noiseV = device.makeBuffer(length: valNodes * 3 * 4, options: .storageModeShared)!

let msgInputT = device.makeBuffer(length: trainEdges * (2 * hiddenDim + 1) * 4, options: .storageModeShared)!
let msgInputV = device.makeBuffer(length: valEdges * (2 * hiddenDim + 1) * 4, options: .storageModeShared)!
let preActivT = device.makeBuffer(length: trainEdges * hiddenDim * 4, options: .storageModeShared)!
let preActivV = device.makeBuffer(length: valEdges * hiddenDim * 4, options: .storageModeShared)!
let nodeActivT = device.makeBuffer(length: trainNodes * 2 * hiddenDim * 4, options: .storageModeShared)!
let nodeActivV = device.makeBuffer(length: valNodes * 2 * hiddenDim * 4, options: .storageModeShared)!
let preActivNodeT = device.makeBuffer(length: trainNodes * hiddenDim * 4, options: .storageModeShared)!
let preActivNodeV = device.makeBuffer(length: valNodes * hiddenDim * 4, options: .storageModeShared)!

let weights = device.makeBuffer(length: hiddenDim * (2 * hiddenDim + 1) * 4, options: .storageModeShared)!
let nodeW = device.makeBuffer(length: hiddenDim * (2 * hiddenDim) * 4, options: .storageModeShared)!
let bias = device.makeBuffer(length: hiddenDim * 4, options: .storageModeShared)!
let coordW = device.makeBuffer(length: hiddenDim * 4, options: .storageModeShared)!
let coordB = device.makeBuffer(length: 4, options: .storageModeShared)!

let gradW = device.makeBuffer(length: weights.length, options: .storageModeShared)!
let gradNodeW = device.makeBuffer(length: nodeW.length, options: .storageModeShared)!
let gradNodeB = device.makeBuffer(length: hiddenDim * 4, options: .storageModeShared)!
let gradCoordW = device.makeBuffer(length: coordW.length, options: .storageModeShared)!
let gradH = device.makeBuffer(length: hT.length, options: .storageModeShared)!
let gradPos = device.makeBuffer(length: posT.length, options: .storageModeShared)!
let gradMsg = device.makeBuffer(length: msgT.length, options: .storageModeShared)!

let weightsM = device.makeBuffer(length: weights.length, options: .storageModeShared)!
let weightsV = device.makeBuffer(length: weights.length, options: .storageModeShared)!
let lossT = device.makeBuffer(length: 4, options: .storageModeShared)!
let lossV = device.makeBuffer(length: 4, options: .storageModeShared)!
let normSq = device.makeBuffer(length: 4, options: .storageModeShared)!

// --- DATA MIGRATION ---

let setupCB = commandQueue.makeCommandBuffer()!
let blit = setupCB.makeBlitCommandEncoder()!
blit.copy(from: loader.nodeBuffer!, sourceOffset: 0, to: nodeBufT, destinationOffset: 0, size: nodeBufT.length)
blit.copy(from: loader.edgeBuffer!, sourceOffset: 0, to: edgeBufT, destinationOffset: 0, size: edgeBufT.length)
blit.copy(from: loader.nodeBuffer!, sourceOffset: alignedByteOffset, to: nodeBufV, destinationOffset: 0, size: nodeBufV.length)
blit.copy(from: loader.edgeBuffer!, sourceOffset: alignedEdgeByteOffset, to: edgeBufV, destinationOffset: 0, size: edgeBufV.length)
blit.endEncoding(); setupCB.commit(); setupCB.waitUntilCompleted()

// --- INITIALIZATION ---

[gradW, gradNodeW, gradNodeB, gradCoordW, gradH, gradPos, gradMsg, weightsM, weightsV, normSq, lossT, lossV, hT, hV, posT, posV, aggT, aggV, msgT, msgV].forEach { ZeroInit($0) }
KaimingInit(weights, count: hiddenDim * (2 * hiddenDim + 1), fanIn: (2 * hiddenDim + 1))
KaimingInit(nodeW, count: hiddenDim * 2 * hiddenDim, fanIn: 2 * hiddenDim)
KaimingInit(coordW, count: hiddenDim, fanIn: hiddenDim)
KaimingInit(loader.nodeBuffer!, count: 10 * hiddenDim, fanIn: 10) // Embed Table Init
ZeroInit(bias); ZeroInit(coordB)

let nPtrT = noiseT.contents().bindMemory(to: Float.self, capacity: trainNodes * 3)
for i in 0..<(trainNodes * 3) { nPtrT[i] = Float.random(in: -0.1...0.1) }
let nPtrV = noiseV.contents().bindMemory(to: Float.self, capacity: valNodes * 3)
for i in 0..<(valNodes * 3) { nPtrV[i] = Float.random(in: -0.1...0.1) }

let lib = device.makeDefaultLibrary()!
let names = ["embed_atoms", "compute_message", "aggregate_message", "update_coords", "apply_updates", "compute_mse_loss", "backward_node", "backward_coordinate", "backward_message", "compute_grad_norm_sq", "apply_clipping", "apply_adam_update"]
var p = [String: MTLComputePipelineState]()
for n in names { p[n] = try! device.makeComputePipelineState(function: lib.makeFunction(name: n)!) }

// --- TRAINING LOOP ---

var timestep: UInt32 = 1; var lr: Float = 1e-4

for epoch in 1...150 {
    let cb = commandQueue.makeCommandBuffer()!
    var hDim = UInt32(hiddenDim)
    if epoch % 30 == 0 { lr *= 0.5 }

    // 1. TRAINING PHASE
    let reset = cb.makeBlitCommandEncoder()!
    [gradW, gradNodeW, gradNodeB, gradCoordW, gradH, gradPos, gradMsg, hT, aggT, posT].forEach { reset.fill(buffer: $0, range: 0..<$0.length, value: 0) }
    reset.endEncoding()

    let enc = cb.makeComputeCommandEncoder()!
    
    enc.setComputePipelineState(p["embed_atoms"]!)
    enc.setBuffer(nodeBufT, offset: 0, index: 0)
    enc.setBuffer(loader.nodeBuffer!, offset: 0, index: 1)
    enc.setBuffer(hT, offset: 0, index: 2)
    enc.setBytes(&hDim, length: 4, index: 3)
    enc.dispatchThreads(MTLSize(width: trainNodes, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    
    for _ in 0..<4 {
        enc.setComputePipelineState(p["compute_message"]!)
        enc.setBuffer(nodeBufT, offset: 0, index: 0)
        enc.setBuffer(hT, offset: 0, index: 1)
        enc.setBuffer(edgeBufT, offset: 0, index: 2)
        enc.setBuffer(weights, offset: 0, index: 3)
        enc.setBuffer(bias, offset: 0, index: 4)
        enc.setBuffer(msgT, offset: 0, index: 5)
        enc.setBytes(&hDim, length: 4, index: 6)
        enc.setBuffer(msgInputT, offset: 0, index: 7)
        enc.setBuffer(preActivT, offset: 0, index: 8)
        enc.dispatchThreads(MTLSize(width: trainEdges, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        
        enc.setComputePipelineState(p["aggregate_message"]!)
        enc.setBuffer(msgT, offset: 0, index: 0)
        enc.setBuffer(edgeBufT, offset: 0, index: 1)
        enc.setBuffer(aggT, offset: 0, index: 2)
        enc.setBytes(&hDim, length: 4, index: 3)
        enc.dispatchThreads(MTLSize(width: trainEdges, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        
        enc.setComputePipelineState(p["update_coords"]!)
        enc.setBuffer(nodeBufT, offset: 0, index: 0)
        enc.setBuffer(msgT, offset: 0, index: 1)
        enc.setBuffer(edgeBufT, offset: 0, index: 2)
        enc.setBuffer(coordW, offset: 0, index: 3)
        enc.setBuffer(coordB, offset: 0, index: 4)
        enc.setBuffer(posT, offset: 0, index: 5)
        enc.setBytes(&hDim, length: 4, index: 6)
        enc.dispatchThreads(MTLSize(width: trainEdges, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        
        enc.setComputePipelineState(p["apply_updates"]!)
        enc.setBuffer(nodeBufT, offset: 0, index: 0)
        enc.setBuffer(hT, offset: 0, index: 1)
        enc.setBuffer(posT, offset: 0, index: 2)
        enc.setBuffer(aggT, offset: 0, index: 3)
        enc.setBuffer(nodeW, offset: 0, index: 4)
        enc.setBuffer(bias, offset: 0, index: 5)
        enc.setBytes(&hDim, length: 4, index: 6)
        enc.setBuffer(nodeActivT, offset: 0, index: 7)
        enc.setBuffer(preActivNodeT, offset: 0, index: 8)
        enc.dispatchThreads(MTLSize(width: trainNodes, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

    enc.setComputePipelineState(p["compute_mse_loss"]!)
    enc.setBuffer(posT, offset: 0, index: 0)
    enc.setBuffer(noiseT, offset: 0, index: 1)
    enc.setBuffer(lossT, offset: 0, index: 2)
    var trN = UInt32(trainNodes*3)
    enc.setBytes(&trN, length: 4, index: 3)
    enc.dispatchThreads(MTLSize(width: Int(trN), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    
    // Backward Logic
    enc.setComputePipelineState(p["backward_node"]!)
    enc.setBuffer(gradH, offset: 0, index: 0)
    enc.setBuffer(nodeW, offset: 0, index: 1)
    enc.setBuffer(nodeActivT, offset: 0, index: 2)
    enc.setBuffer(preActivNodeT, offset: 0, index: 3)
    enc.setBuffer(gradNodeW, offset: 0, index: 4)
    enc.setBuffer(gradNodeB, offset: 0, index: 5)
    enc.setBuffer(gradH, offset: 0, index: 6)
    enc.setBytes(&hDim, length: 4, index: 7)
    enc.dispatchThreads(MTLSize(width: trainNodes, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

    enc.setComputePipelineState(p["apply_adam_update"]!)
    enc.setBuffer(weights, offset: 0, index: 0)
    enc.setBuffer(weightsM, offset: 0, index: 1)
    enc.setBuffer(weightsV, offset: 0, index: 2)
    enc.setBuffer(gradW, offset: 0, index: 3)
    enc.setBytes(&lr, length: 4, index: 4)
    var b1: Float = 0.9
    var b2: Float = 0.999
    var eps: Float = 1e-8
    var t = timestep
    enc.setBytes(&b1, length: 4, index: 5)
    enc.setBytes(&b2, length: 4, index: 6)
    enc.setBytes(&eps, length: 4, index: 7)
    enc.setBytes(&t, length: 4, index: 8)
    var tW = UInt32(weights.length / 4)
    enc.dispatchThreads(MTLSize(width: Int(tW), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

    // 2. VALIDATION PHASE
    enc.setComputePipelineState(p["embed_atoms"]!)
    enc.setBuffer(nodeBufV, offset: 0, index: 0)
    enc.setBuffer(loader.nodeBuffer!, offset: 0, index: 1)
    enc.setBuffer(hV, offset: 0, index: 2)
    enc.setBytes(&hDim, length: 4, index: 3)
    enc.dispatchThreads(MTLSize(width: valNodes, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    
    for _ in 0..<4 {
        // Re-binding index 7/8 to mutable buffers kills the "Read-only bytes" assertion
        enc.setComputePipelineState(p["compute_message"]!)
        enc.setBuffer(nodeBufV, offset: 0, index: 0)
        enc.setBuffer(hV, offset: 0, index: 1)
        enc.setBuffer(edgeBufV, offset: 0, index: 2)
        enc.setBuffer(weights, offset: 0, index: 3)
        enc.setBuffer(bias, offset: 0, index: 4)
        enc.setBuffer(msgV, offset: 0, index: 5)
        enc.setBytes(&hDim, length: 4, index: 6)
        enc.setBuffer(msgInputV, offset: 0, index: 7)
        enc.setBuffer(preActivV, offset: 0, index: 8)
        enc.dispatchThreads(MTLSize(width: valEdges, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        
        enc.setComputePipelineState(p["aggregate_message"]!)
        enc.setBuffer(msgV, offset: 0, index: 0)
        enc.setBuffer(edgeBufV, offset: 0, index: 1)
        enc.setBuffer(aggV, offset: 0, index: 2)
        enc.setBytes(&hDim, length: 4, index: 3)
        enc.dispatchThreads(MTLSize(width: valEdges, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        
        enc.setComputePipelineState(p["update_coords"]!)
        enc.setBuffer(nodeBufV, offset: 0, index: 0)
        enc.setBuffer(msgV, offset: 0, index: 1)
        enc.setBuffer(edgeBufV, offset: 0, index: 2)
        enc.setBuffer(coordW, offset: 0, index: 3)
        enc.setBuffer(coordB, offset: 0, index: 4)
        enc.setBuffer(posV, offset: 0, index: 5)
        enc.setBytes(&hDim, length: 4, index: 6)
        enc.dispatchThreads(MTLSize(width: valEdges, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        
        enc.setComputePipelineState(p["apply_updates"]!)
        enc.setBuffer(nodeBufV, offset: 0, index: 0)
        enc.setBuffer(hV, offset: 0, index: 1)
        enc.setBuffer(posV, offset: 0, index: 2)
        enc.setBuffer(aggV, offset: 0, index: 3)
        enc.setBuffer(nodeW, offset: 0, index: 4)
        enc.setBuffer(bias, offset: 0, index: 5)
        enc.setBytes(&hDim, length: 4, index: 6)
        enc.setBuffer(nodeActivV, offset: 0, index: 7)
        enc.setBuffer(preActivNodeV, offset: 0, index: 8)
        enc.dispatchThreads(MTLSize(width: valNodes, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

    enc.setComputePipelineState(p["compute_mse_loss"]!)
    enc.setBuffer(posV, offset: 0, index: 0)
    enc.setBuffer(noiseV, offset: 0, index: 1)
    enc.setBuffer(lossV, offset: 0, index: 2)
    var vN = UInt32(valNodes*3)
    enc.setBytes(&vN, length: 4, index: 3)
    enc.dispatchThreads(MTLSize(width: Int(vN), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

    enc.endEncoding()
    cb.addCompletedHandler { _ in
        let tMse = lossT.contents().bindMemory(to: Float.self, capacity: 1).pointee / Float(trainNodes*3)
        let vMse = lossV.contents().bindMemory(to: Float.self, capacity: 1).pointee / Float(valNodes*3)
        print("Epoch \(epoch) | Train: \(tMse) | Val: \(vMse) | LR: \(lr)")
    }
    cb.commit(); cb.waitUntilCompleted(); timestep += 1
}

SaveModelWeights(buffers: [("weights", weights), ("nodeW", nodeW)], path: datapath)
