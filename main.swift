//
//  main.swift
//  MSLGraphDiffusion
//
//  Created by Raiden Makoto on 2026-02-04.
//

import Metal
import Foundation

func XavierInit(_ buffer: MTLBuffer, count: Int, fanIn: Int, fanOut: Int) {
    precondition(count > 0 && fanIn > 0 && fanOut > 0, "count, fanIn, and fanOut must be positive")
    let ptr = buffer.contents().bindMemory(to: Float.self, capacity: count)
    // Glorot/Xavier uniform: U(-limit, limit) where limit = sqrt(6 / (fanIn + fanOut))
    let limit = Float(sqrt(6.0 / Double(fanIn + fanOut)))
    for i in 0..<count {
        ptr[i] = Float.random(in: -limit...limit)
    }
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

guard let device = MTLCreateSystemDefaultDevice() else {
    fatalError("Metal is not supported on this device!")
}

print("Engine initialized on: \(device.name)")
let commandQueue = device.makeCommandQueue()!
let loader = QM9Loader(device: device)
let sourceDirURL = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
let datapath = sourceDirURL.path

do {
    print("Attempting to load the data from \(datapath)...")
    try loader.load(from: datapath)
    if let metaBuffer = loader.graphdataBuffer {
        let metaPtr = metaBuffer.contents().bindMemory(to: GraphData.self, capacity: loader.graphCount)
        let sample = metaPtr.pointee
        print("Data Load Success:")
        print("Total Molecules Loaded: \(loader.graphCount)")
        print("Sample Molecule Stats")
        print(" - Node Data: \(sample.nodeStart)")
        print(" - Atom Count: \(sample.nodeCount)")
        print(" - Edge Count: \(sample.edgeCount)")
        print("----------------------------------")
    }
}
catch {
    print("Data Load Failed:")
    print("Error \(error.localizedDescription)")
    print("Check if the datapath exists and try again")
}

//  Get real counts from your loaded data
let firstGraphMetadata = loader.graphdataBuffer!.contents().bindMemory(to: GraphData.self, capacity: 1).pointee
let activeEdgeCount = Int(firstGraphMetadata.edgeCount)
let activeNodeCount = Int(firstGraphMetadata.nodeCount)
let hiddenDim = 64

// Embedding Table setup
let numAtomTypes = 10
let embedTableSize = numAtomTypes * hiddenDim * MemoryLayout<Float>.size
let embedTableBuffer = device.makeBuffer(length: embedTableSize, options: .storageModeShared)!
// Xavier Init the table buffer
XavierInit(embedTableBuffer, count: numAtomTypes * hiddenDim, fanIn: numAtomTypes, fanOut: hiddenDim)

// Create the real Node features (h) buffer here (Must be done before execution)
let hBuffer = device.makeBuffer(length: activeNodeCount * hiddenDim * MemoryLayout<Float>.size, options: .storageModeShared)!

let numEdges = activeEdgeCount // don't hardcode 26
let inputDim = 2 * hiddenDim + 1
let weightCount = hiddenDim * inputDim
let biasCount = hiddenDim

let weightSize = Int(hiddenDim) * (2 * Int(hiddenDim) + 1) * MemoryLayout<Float>.size
let weightsBuffer = device.makeBuffer(length: weightSize, options: .storageModeShared)!
let biasBuffer = device.makeBuffer(length: Int(hiddenDim) * MemoryLayout<Float>.size, options: .storageModeShared)!

XavierInit(weightsBuffer, count: weightCount, fanIn: inputDim, fanOut: hiddenDim)
XavierInit(biasBuffer, count: biasCount, fanIn: hiddenDim, fanOut: 1)

let msgBuffer = device.makeBuffer(length: numEdges * Int(hiddenDim) * MemoryLayout<Float>.size, options: .storageModeShared)!

// Setup all pipelines upfront
let lib = device.makeDefaultLibrary()!

let embedFunction = lib.makeFunction(name: "embed_atoms")!
let embedPipeline = try device.makeComputePipelineState(function: embedFunction)

let msgFunction = lib.makeFunction(name: "compute_message")! // Renamed from 'function' to 'msgFunction' for clarity
let msgPipeline = try device.makeComputePipelineState(function: msgFunction)

let aggFunction = lib.makeFunction(name: "aggregate_message")!
let aggPipeline = try! device.makeComputePipelineState(function: aggFunction)

let coordFunction = lib.makeFunction(name: "update_coords")!
let coordPipeline = try device.makeComputePipelineState(function: coordFunction)

// --- EXECUTION START ---
let commandBuffer = commandQueue.makeCommandBuffer()!

// 1. EMBEDDING ENCODER
let embedEncoder = commandBuffer.makeComputeCommandEncoder()!
embedEncoder.setComputePipelineState(embedPipeline)

embedEncoder.setBuffer(loader.nodeBuffer, offset: 0, index: 0)
embedEncoder.setBuffer(embedTableBuffer, offset: 0, index: 1)
embedEncoder.setBuffer(hBuffer, offset: 0, index: 2)
var hDimEmbed = UInt32(hiddenDim)
embedEncoder.setBytes(&hDimEmbed, length: MemoryLayout<UInt32>.size, index: 3)

embedEncoder.dispatchThreads(
    MTLSize(width: activeNodeCount, height: 1, depth: 1),
    threadsPerThreadgroup: MTLSize(width: min(activeNodeCount, 32), height: 1, depth: 1)
)
embedEncoder.endEncoding()

// 2. MESSAGE ENCODER
let encoder = commandBuffer.makeComputeCommandEncoder()!
encoder.setComputePipelineState(msgPipeline)

encoder.setBuffer(loader.nodeBuffer, offset: 0, index: 0)
encoder.setBuffer(hBuffer, offset: 0, index: 1) // Using the filled hBuffer
encoder.setBuffer(loader.edgeBuffer, offset: 0, index: 2)
encoder.setBuffer(weightsBuffer, offset: 0, index: 3)
encoder.setBuffer(biasBuffer, offset: 0, index: 4)
encoder.setBuffer(msgBuffer, offset: 0, index: 5)

var hDim = UInt32(hiddenDim)
encoder.setBytes(&hDim, length: MemoryLayout<UInt32>.size, index: 6)

let gridSize = MTLSize(width: activeEdgeCount, height: 1, depth: 1)
let threadgroupSize = MTLSize(width: min(activeEdgeCount, msgPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)

encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
encoder.endEncoding() // MUST END BEFORE STARTING NEXT

// 3. AGGREGATION ENCODER
let aggBuffer = device.makeBuffer(length: activeNodeCount * hiddenDim * MemoryLayout<Float>.size, options: .storageModeShared)!
memset(aggBuffer.contents(), 0, aggBuffer.length); // ensure it starts a zero

let aggEncoder = commandBuffer.makeComputeCommandEncoder()!
aggEncoder.setComputePipelineState(aggPipeline)

aggEncoder.setBuffer(msgBuffer, offset: 0, index: 0)
aggEncoder.setBuffer(loader.edgeBuffer, offset: 0, index: 1)
aggEncoder.setBuffer(aggBuffer, offset: 0, index: 2)
var hDimAgg = UInt32(hiddenDim)
aggEncoder.setBytes(&hDimAgg, length: MemoryLayout<UInt32>.size, index: 3)

let aggGridSize = MTLSize(width: activeEdgeCount, height: 1, depth: 1)
aggEncoder.dispatchThreads(aggGridSize, threadsPerThreadgroup: threadgroupSize)
aggEncoder.endEncoding() // MUST END BEFORE STARTING NEXT

// 4. COORDINATE UPDATE ENCODER
let coordWeightBuffer = device.makeBuffer(length: hiddenDim * MemoryLayout<Float>.size, options: .storageModeShared)!
let coordBiasBuffer = device.makeBuffer(length: MemoryLayout<Float>.size, options: .storageModeShared)!
let posUpdateBuffer = device.makeBuffer(length: activeNodeCount * 3 * MemoryLayout<Float>.size, options: .storageModeShared)!
// Xavier Init the weights
XavierInit(coordWeightBuffer, count: hiddenDim, fanIn: hiddenDim, fanOut: 1)
memset(posUpdateBuffer.contents(), 0, posUpdateBuffer.length)

let coordEncoder = commandBuffer.makeComputeCommandEncoder()!
coordEncoder.setComputePipelineState(coordPipeline)

coordEncoder.setBuffer(loader.nodeBuffer, offset: 0, index: 0) // Pos
coordEncoder.setBuffer(msgBuffer, offset: 0, index: 1)         // Messages
coordEncoder.setBuffer(loader.edgeBuffer, offset: 0, index: 2) // Edge Index
coordEncoder.setBuffer(coordWeightBuffer, offset: 0, index: 3) // Weights
coordEncoder.setBuffer(coordBiasBuffer, offset: 0, index: 4)   // Bias
coordEncoder.setBuffer(posUpdateBuffer, offset: 0, index: 5)   // Output
coordEncoder.setBytes(&hDim, length: MemoryLayout<UInt32>.size, index: 6)

coordEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
coordEncoder.endEncoding()

// Finalize and Submit
commandBuffer.commit()
commandBuffer.waitUntilCompleted() // Wait for GPU to finish

// Check Output (V3)
let posPtr = posUpdateBuffer.contents().bindMemory(to: Float.self, capacity: 3)
print("Position Update for Node 0 (dx, dy, dz): (\(posPtr[0]), \(posPtr[1]), \(posPtr[2]))")
