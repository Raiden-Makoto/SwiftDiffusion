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

let numEdges = 26
let hiddenDim = 64
let inputDim = 2 * hiddenDim + 1
let weightCount = hiddenDim * inputDim
let biasCount = hiddenDim

let weightSize = Int(hiddenDim) * (2 * Int(hiddenDim) + 1) * MemoryLayout<Float>.size
let weightsBuffer = device.makeBuffer(length: weightSize, options: .storageModeShared)!
let biasBuffer = device.makeBuffer(length: Int(hiddenDim) * MemoryLayout<Float>.size, options: .storageModeShared)!

XavierInit(weightsBuffer, count: weightCount, fanIn: inputDim, fanOut: hiddenDim)
XavierInit(biasBuffer, count: biasCount, fanIn: hiddenDim, fanOut: 1)

let msgBuffer = device.makeBuffer(length: numEdges * Int(hiddenDim) * MemoryLayout<Float>.size, options: .storageModeShared)!

let lib = device.makeDefaultLibrary()!
let function = lib.makeFunction(name: "compute_message")!
let pipeline = try device.makeComputePipelineState(function: function)

let commandBuffer = commandQueue.makeCommandBuffer()!
let encoder = commandBuffer.makeComputeCommandEncoder()!
encoder.setComputePipelineState(pipeline)

//  Get real counts from your loaded data instead of placeholders
let firstGraphMetadata = loader.graphdataBuffer!.contents().bindMemory(to: GraphData.self, capacity: 1).pointee
let activeEdgeCount = Int(firstGraphMetadata.edgeCount)
let activeNodeCount = Int(firstGraphMetadata.nodeCount)

// Index 0: Nodes [x, y, z, type]
encoder.setBuffer(loader.nodeBuffer, offset: 0, index: 0)

// Index 1: Node Features (h) - Initially using a zeros or random buffer
// For testing, let's create an 'h' buffer for the nodes of the first molecule
let hBuffer = device.makeBuffer(length: activeNodeCount * hiddenDim * MemoryLayout<Float>.size, options: .storageModeShared)!
// We don't use Xavier for activation/feature tensors, use small normal instead
let hPtr = hBuffer.contents().bindMemory(to: Float.self, capacity: activeNodeCount * hiddenDim)
for i in 0..<(activeNodeCount * hiddenDim) {
  hPtr[i] = Float.random(in: -0.01...0.01)
}
encoder.setBuffer(hBuffer, offset: 0, index: 1)

// Index 2: Edges [row, col]
encoder.setBuffer(loader.edgeBuffer, offset: 0, index: 2)

// Indices 3 & 4: Weights and Biases (already initialized by your XavierInit)
encoder.setBuffer(weightsBuffer, offset: 0, index: 3)
encoder.setBuffer(biasBuffer, offset: 0, index: 4)

// Index 5: The output buffer for messages
encoder.setBuffer(msgBuffer, offset: 0, index: 5)

// Index 6: Pass hiddenDim as a constant
var hDim = UInt32(hiddenDim)
encoder.setBytes(&hDim, length: MemoryLayout<UInt32>.size, index: 6)

// Dispatch: Launch one thread per edge
let gridSize = MTLSize(width: activeEdgeCount, height: 1, depth: 1)
let threadgroupSize = MTLSize(width: min(activeEdgeCount, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)

encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
encoder.endEncoding()

// perform aggregation
let aggBuffer = device.makeBuffer(length: activeNodeCount * hiddenDim * MemoryLayout<Float>.size, options: .storageModeShared)!
memset(aggBuffer.contents(), 0, aggBuffer.length); // ensure it starts a zero
let aggFunction = lib.makeFunction(name: "aggregate_message")!
let aggPipeline = try! device.makeComputePipelineState(function: aggFunction)

let aggEncoder = commandBuffer.makeComputeCommandEncoder()!
aggEncoder.setComputePipelineState(aggPipeline)

aggEncoder.setBuffer(msgBuffer, offset: 0, index: 0)
aggEncoder.setBuffer(loader.edgeBuffer, offset: 0, index: 1)
aggEncoder.setBuffer(aggBuffer, offset: 0, index: 2)
var hDimAgg = UInt32(hiddenDim)
aggEncoder.setBytes(&hDimAgg, length: MemoryLayout<UInt32>.size, index: 3)

let aggGridSize = MTLSize(width: activeEdgeCount, height: 1, depth: 1)
aggEncoder.dispatchThreads(aggGridSize, threadsPerThreadgroup: threadgroupSize)
aggEncoder.endEncoding()

// Now we perform the coordinate update
let coordWeightBuffer = device.makeBuffer(length: hiddenDim * MemoryLayout<Float>.size, options: .storageModeShared)!
let coordBiasBuffer = device.makeBuffer(length: MemoryLayout<Float>.size, options: .storageModeShared)!
let posUpdateBuffer = device.makeBuffer(length: activeNodeCount * 3 * MemoryLayout<Float>.size, options: .storageModeShared)!
// Xavier Init the weights
XavierInit(coordWeightBuffer, count: hiddenDim, fanIn: hiddenDim, fanOut: 1)
memset(posUpdateBuffer.contents(), 0, posUpdateBuffer.length)

let coordFunction = lib.makeFunction(name: "update_coords")!
let coordPipeline = try device.makeComputePipelineState(function: coordFunction)
let coordEncoder = commandBuffer.makeComputeCommandEncoder()!
coordEncoder.setComputePipelineState(coordPipeline)

coordEncoder.setBuffer(aggBuffer, offset: 0, index: 0)
coordEncoder.setBuffer(coordWeightBuffer, offset: 0, index: 1)
coordEncoder.setBuffer(loader.nodeBuffer, offset: 0, index: 0)
coordEncoder.setBuffer(msgBuffer, offset: 0, index: 1)
coordEncoder.setBuffer(loader.edgeBuffer, offset: 0, index: 2)
coordEncoder.setBuffer(coordWeightBuffer, offset: 0, index: 3)
coordEncoder.setBuffer(coordBiasBuffer, offset: 0, index: 4)
coordEncoder.setBuffer(posUpdateBuffer, offset: 0, index: 5)
coordEncoder.setBytes(&hDim, length: MemoryLayout<UInt32>.size, index: 6)

coordEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
coordEncoder.endEncoding()

// Finalize and Submit
commandBuffer.commit()
commandBuffer.waitUntilCompleted() // Wait for GPU to finish

// Check Output (V3)
let posPtr = posUpdateBuffer.contents().bindMemory(to: Float.self, capacity: 3)
print("Position Update for Node 0 (dx, dy, dz): (\(posPtr[0]), \(posPtr[1]), \(posPtr[2]))")
