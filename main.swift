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
    precondition(count > 0 && fanIn > 0, "count and fanIn must be positive")
    let ptr = buffer.contents().bindMemory(to: Float.self, capacity: count)
    let std = sqrt(2.0 / Double(fanIn))
    for i in 0..<count {
        // Box-Muller transform for normal(0,1)
        let u1 = max(Float.leastNonzeroMagnitude, Float.random(in: 0..<1))
        let u2 = Float.random(in: 0..<1)
        let z = Float(sqrt(-2.0 * log(Double(u1))) * cos(2.0 * Double.pi * Double(u2)))
        ptr[i] = z * Float(std)
    }
}

func ZeroInit(_ buffer: MTLBuffer) {
    memset(buffer.contents(), 0, buffer.length)
}

func TimestepEmbedding(t: Float, dim: Int) -> [Float]{
    var embedding = [Float](repeating: 0, count: dim)
    let halfDim = dim / 2
    let exponent = log(10000.0) / Double(halfDim - 1)
    for i in 0..<halfDim {
        let freq = exp(-exponent * Double(i))
        let arg = Double(t) * freq
        embedding[i] = Float(sin(arg))
        embedding[i + halfDim] = Float(cos(arg))
    }
    return embedding
}

// --- INITIALIZATION ---

guard let device = MTLCreateSystemDefaultDevice() else {
    fatalError("Metal is not supported on this device!")
}

print("Engine initialized on: \(device.name)")
let commandQueue = device.makeCommandQueue()!
let loader = QM9Loader(device: device)
let sourceDirURL = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
let datapath = sourceDirURL.path

do {
    print("Attempting to load data from \(datapath)...")
    try loader.load(from: datapath)
} catch {
    fatalError("Data Load Failed: \(error.localizedDescription)")
}

guard let nodeBuffer = loader.nodeBuffer else {
    fatalError("loader.nodeBuffer is nil after loading data")
}
let firstGraphMetadata = loader.graphdataBuffer!.contents().bindMemory(to: GraphData.self, capacity: 1).pointee
let activeEdgeCount = Int(firstGraphMetadata.edgeCount)
let activeNodeCount = Int(firstGraphMetadata.nodeCount)
let hiddenDim = 64

// --- BUFFER ALLOCATIONS ---

// 1. Feature & Time Buffers
let embedTableBuffer = device.makeBuffer(length: 10 * hiddenDim * MemoryLayout<Float>.size, options: .storageModeShared)!
let hBuffer = device.makeBuffer(length: activeNodeCount * hiddenDim * MemoryLayout<Float>.size, options: .storageModeShared)!
let t_vector = TimestepEmbedding(t: 0.5, dim: hiddenDim)
let tEmbBuffer = device.makeBuffer(bytes: t_vector, length: hiddenDim * MemoryLayout<Float>.size, options: .storageModeShared)!

// 2. Message Buffers
let weightsBuffer = device.makeBuffer(length: hiddenDim * (2 * hiddenDim + 1) * MemoryLayout<Float>.size, options: .storageModeShared)!
let biasBuffer = device.makeBuffer(length: hiddenDim * MemoryLayout<Float>.size, options: .storageModeShared)!
let msgBuffer = device.makeBuffer(length: activeEdgeCount * hiddenDim * MemoryLayout<Float>.size, options: .storageModeShared)!

// 3. Aggregation & Coordinate Buffers
let aggBuffer = device.makeBuffer(length: activeNodeCount * hiddenDim * MemoryLayout<Float>.size, options: .storageModeShared)!
let coordWeightBuffer = device.makeBuffer(length: hiddenDim * MemoryLayout<Float>.size, options: .storageModeShared)!
let coordBiasBuffer = device.makeBuffer(length: MemoryLayout<Float>.size, options: .storageModeShared)!
let posUpdateBuffer = device.makeBuffer(length: activeNodeCount * 3 * MemoryLayout<Float>.size, options: .storageModeShared)!

// 4. Node MLP Buffers
let nodeWBuffer = device.makeBuffer(length: hiddenDim * (2 * hiddenDim) * MemoryLayout<Float>.size, options: .storageModeShared)!
let nodeBBuffer = device.makeBuffer(length: hiddenDim * MemoryLayout<Float>.size, options: .storageModeShared)!

// 5. Normalization Buffers
let cogSumBuffer = device.makeBuffer(length: 3 * MemoryLayout<Float>.size, options: .storageModeShared)!
memset(cogSumBuffer.contents(), 0, cogSumBuffer.length)

// --- WEIGHT INITIALIZATION ---

KaimingInit(embedTableBuffer, count: 10 * hiddenDim, fanIn: 10)
KaimingInit(weightsBuffer, count: hiddenDim * (2 * hiddenDim + 1), fanIn: (2 * hiddenDim + 1))
ZeroInit(biasBuffer)
KaimingInit(coordWeightBuffer, count: hiddenDim, fanIn: hiddenDim)
KaimingInit(nodeWBuffer, count: hiddenDim * 2 * hiddenDim, fanIn: 2 * hiddenDim)
ZeroInit(nodeBBuffer)

// --- PIPELINES ---

let lib = device.makeDefaultLibrary()!
let embedPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "embed_atoms")!)
let timePipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "inject_timestep")!)
let msgPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "compute_message")!)
let aggPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "aggregate_message")!)
let coordPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "update_coords")!)
let applyPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "apply_updates")!)
let cogPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "compute_cog")!)
let normPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "apply_cog_normalization")!)

// --- TRAINING TARGET SETUP ---
let lossPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "compute_mse_loss")!)
let targetNoiseBuffer = device.makeBuffer(length: activeNodeCount * 3 * MemoryLayout<Float>.size, options: .storageModeShared)!
let targetPtr = targetNoiseBuffer.contents().bindMemory(to: Float.self, capacity: activeNodeCount * 3)

// Generate Gaussian Noise on CPU
for i in 0..<(activeNodeCount * 3) {
    let u1 = Float.random(in: 0...1), u2 = Float.random(in: 0...1)
    let mag = sqrt(-2.0 * log(u1))
    targetPtr[i] = mag * cos(2.0 * Float.pi * u2)
}

// Add noise to the starting positions so the model has something to "denoise"
let nodePtrStart = nodeBuffer.contents().bindMemory(to: Node.self, capacity: activeNodeCount)
for i in 0..<activeNodeCount {
    nodePtrStart[i].pos.x += targetPtr[i*3]
    nodePtrStart[i].pos.y += targetPtr[i*3+1]
    nodePtrStart[i].pos.z += targetPtr[i*3+2]
}

// --- EXECUTION LOOP ---

let commandBuffer = commandQueue.makeCommandBuffer()!

// STAGE 1: INITIALIZATION (Embed + Time Injection)
let embedEncoder = commandBuffer.makeComputeCommandEncoder()!
embedEncoder.setComputePipelineState(embedPipeline)
embedEncoder.setBuffer(nodeBuffer, offset: 0, index: 0)
embedEncoder.setBuffer(embedTableBuffer, offset: 0, index: 1)
embedEncoder.setBuffer(hBuffer, offset: 0, index: 2)
var hDimVal = UInt32(hiddenDim)
embedEncoder.setBytes(&hDimVal, length: MemoryLayout<UInt32>.size, index: 3)
embedEncoder.dispatchThreads(MTLSize(width: activeNodeCount, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
embedEncoder.endEncoding()

let timeEncoder = commandBuffer.makeComputeCommandEncoder()!
timeEncoder.setComputePipelineState(timePipeline)
timeEncoder.setBuffer(hBuffer, offset: 0, index: 0)
timeEncoder.setBuffer(tEmbBuffer, offset: 0, index: 1)
timeEncoder.setBytes(&hDimVal, length: MemoryLayout<UInt32>.size, index: 2)
timeEncoder.dispatchThreads(MTLSize(width: activeNodeCount, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
timeEncoder.endEncoding()

// STAGE 2: 4-LAYER EGNN LOOP
for _ in 0..<4 {
    // A. Compute Messages
    let msgEncoder = commandBuffer.makeComputeCommandEncoder()!
    msgEncoder.setComputePipelineState(msgPipeline)
    msgEncoder.setBuffer(nodeBuffer, offset: 0, index: 0)
    msgEncoder.setBuffer(hBuffer, offset: 0, index: 1)
    msgEncoder.setBuffer(loader.edgeBuffer, offset: 0, index: 2)
    msgEncoder.setBuffer(weightsBuffer, offset: 0, index: 3)
    msgEncoder.setBuffer(biasBuffer, offset: 0, index: 4)
    msgEncoder.setBuffer(msgBuffer, offset: 0, index: 5)
    msgEncoder.setBytes(&hDimVal, length: MemoryLayout<UInt32>.size, index: 6)
    msgEncoder.dispatchThreads(MTLSize(width: activeEdgeCount, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    msgEncoder.endEncoding()

    // B. Clear & Aggregate
    let blit = commandBuffer.makeBlitCommandEncoder()!
    blit.fill(buffer: aggBuffer, range: 0..<aggBuffer.length, value: 0)
    blit.endEncoding()

    let aggEncoder = commandBuffer.makeComputeCommandEncoder()!
    aggEncoder.setComputePipelineState(aggPipeline)
    aggEncoder.setBuffer(msgBuffer, offset: 0, index: 0)
    aggEncoder.setBuffer(loader.edgeBuffer, offset: 0, index: 1)
    aggEncoder.setBuffer(aggBuffer, offset: 0, index: 2)
    aggEncoder.setBytes(&hDimVal, length: MemoryLayout<UInt32>.size, index: 3)
    aggEncoder.dispatchThreads(MTLSize(width: activeEdgeCount, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    aggEncoder.endEncoding()

    // C. Coordinate Update Calculation
    let blit2 = commandBuffer.makeBlitCommandEncoder()!
    blit2.fill(buffer: posUpdateBuffer, range: 0..<posUpdateBuffer.length, value: 0)
    blit2.endEncoding()

    let coordEncoder = commandBuffer.makeComputeCommandEncoder()!
    coordEncoder.setComputePipelineState(coordPipeline)
    coordEncoder.setBuffer(nodeBuffer, offset: 0, index: 0)
    coordEncoder.setBuffer(msgBuffer, offset: 0, index: 1)
    coordEncoder.setBuffer(loader.edgeBuffer, offset: 0, index: 2)
    coordEncoder.setBuffer(coordWeightBuffer, offset: 0, index: 3)
    coordEncoder.setBuffer(coordBiasBuffer, offset: 0, index: 4)
    coordEncoder.setBuffer(posUpdateBuffer, offset: 0, index: 5)
    coordEncoder.setBytes(&hDimVal, length: MemoryLayout<UInt32>.size, index: 6)
    coordEncoder.dispatchThreads(MTLSize(width: activeEdgeCount, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    coordEncoder.endEncoding()

    // D. Final Apply (Pos Displacement + Node MLP)
    let applyEncoder = commandBuffer.makeComputeCommandEncoder()!
    applyEncoder.setComputePipelineState(applyPipeline)
    applyEncoder.setBuffer(nodeBuffer, offset: 0, index: 0)
    applyEncoder.setBuffer(hBuffer, offset: 0, index: 1)
    applyEncoder.setBuffer(posUpdateBuffer, offset: 0, index: 2)
    applyEncoder.setBuffer(aggBuffer, offset: 0, index: 3)
    applyEncoder.setBuffer(nodeWBuffer, offset: 0, index: 4)
    applyEncoder.setBuffer(nodeBBuffer, offset: 0, index: 5)
    applyEncoder.setBytes(&hDimVal, length: MemoryLayout<UInt32>.size, index: 6)
    applyEncoder.dispatchThreads(MTLSize(width: activeNodeCount, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    applyEncoder.endEncoding()
    
    // E: CENTER OF GRAVITY NORMALIZATION
    // 1. Clear the Cog Sum
    let blitCog = commandBuffer.makeBlitCommandEncoder()!
    blitCog.fill(buffer: cogSumBuffer, range: 0..<cogSumBuffer.length, value: 0)
    blitCog.endEncoding()

    // 2. Compute the center (Atomic sum into cogSumBuffer)
    let cogEncoder = commandBuffer.makeComputeCommandEncoder()!
    cogEncoder.setComputePipelineState(cogPipeline)
    cogEncoder.setBuffer(loader.nodeBuffer, offset: 0, index: 0)
    cogEncoder.setBuffer(cogSumBuffer, offset: 0, index: 1)
    cogEncoder.dispatchThreads(MTLSize(width: activeNodeCount, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    cogEncoder.endEncoding()

    // 3. Subtract the Center (GPU-Side)
    // Note: To keep this 100% GPU-sync, we pass the atom count as a constant
    // and the kernel handles the division.
    let normEncoder = commandBuffer.makeComputeCommandEncoder()!
    normEncoder.setComputePipelineState(normPipeline)
    normEncoder.setBuffer(loader.nodeBuffer, offset: 0, index: 0)
    normEncoder.setBuffer(cogSumBuffer, offset: 0, index: 1)
    var nodeCountU32 = UInt32(activeNodeCount)
    normEncoder.setBytes(&nodeCountU32, length: MemoryLayout<UInt32>.size, index: 2)
    normEncoder.dispatchThreads(MTLSize(width: activeNodeCount, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    normEncoder.endEncoding()
}

// --- COMPUTE LOSS (MSE) ---
let lossBuffer = device.makeBuffer(length: MemoryLayout<Float>.size, options: .storageModeShared)!
memset(lossBuffer.contents(), 0, lossBuffer.length)

let lossEncoder = commandBuffer.makeComputeCommandEncoder()!
lossEncoder.setComputePipelineState(lossPipeline)
lossEncoder.setBuffer(posUpdateBuffer, offset: 0, index: 0) // Final prediction
lossEncoder.setBuffer(targetNoiseBuffer, offset: 0, index: 1) // Original noise added
lossEncoder.setBuffer(lossBuffer, offset: 0, index: 2)
var totalElements = UInt32(activeNodeCount * 3)
lossEncoder.setBytes(&totalElements, length: MemoryLayout<UInt32>.size, index: 3)
lossEncoder.dispatchThreads(MTLSize(width: Int(totalElements), height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
lossEncoder.endEncoding()

// Print result once GPU finishes
commandBuffer.addCompletedHandler { _ in
    let totalLoss = lossBuffer.contents().bindMemory(to: Float.self, capacity: 1).pointee
    let mse = totalLoss / Float(activeNodeCount * 3)
    print("Training Step Complete - MSE Loss: \(mse)")
}

commandBuffer.commit()
commandBuffer.waitUntilCompleted()

// Safely unwrap node buffer before accessing contents
guard let nodeBufferFinal = loader.nodeBuffer else {
    fatalError("loader.nodeBuffer is nil after GPU execution")
}
let finalNodes = nodeBufferFinal.contents().bindMemory(to: Node.self, capacity: activeNodeCount)
print("Final Position Node 0: \(finalNodes[0].pos)")

