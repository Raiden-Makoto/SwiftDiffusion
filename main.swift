//
//  main.swift
//  MSLGraphDiffusion
//
//  Created by Raiden Makoto on 2026-02-07.
//

import Metal
import Foundation

let sectionBreak = String(repeating: "=", count: 50)

// --- CONFIGURATION ---
let hiddenDim = 128
let numLayers = 3
let numTypes = 10

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
            print("VERIFIED: \(name).bin (\(data.count) bytes)")
        } else if data.count < expectedBytes {
            print("PARTIAL LOAD: \(data.count)/\(expectedBytes) loaded. Padding with zeros")
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
    print("Layer \(i):")
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
let numNodes = 5 // for Methane only
let atomTypes: [Int32] = [6, 1, 1, 1, 1] // Example: Methane (C, H, H, H, H)

let typeBuf = device.makeBuffer(bytes: atomTypes, length: numNodes * MemoryLayout<Int32>.size, options: .storageModeShared)!
let hBuf = device.makeBuffer(length: numNodes * hiddenDim * 4, options: .storageModeShared)!
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
    pipeline["inject_timestep"] = try device.makeComputePipelineState(function: injectFunction)
    
    print("Metal Pipelines initialized for: \(pipeline.keys.joined(separator: ", "))")
} catch {
    fatalError("Failed to create compute pipeline states: \(error)")
}

let commandQueue = device.makeCommandQueue()!
let cb = commandQueue.makeCommandBuffer()!
let enc = cb.makeComputeCommandEncoder()!

var hDim = UInt32(hiddenDim)
var nNodes = UInt32(numNodes)

// A. Embed raw atom types
enc.setComputePipelineState(pipeline["embed_atoms"]!)
enc.setBuffer(typeBuf, offset: 0, index: 0)
enc.setBuffer(weights["embedding.weight"], offset: 0, index: 1)
enc.setBuffer(hBuf, offset: 0, index: 2)
enc.setBytes(&hDim, length: 4, index: 3)
enc.setBytes(&nNodes, length: 4, index: 4)
enc.dispatchThreads(MTLSize(width: numNodes, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

// B. Inject Timestep into Hidden States
enc.setComputePipelineState(pipeline["inject_timestep"]!)
enc.setBuffer(hBuf, offset: 0, index: 0)
enc.setBuffer(tProcessedBuf, offset: 0, index: 1)
enc.setBytes(&hDim, length: 4, index: 2)
enc.setBytes(&nNodes, length: 4, index: 3)
enc.dispatchThreads(MTLSize(width: numNodes, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

enc.endEncoding()
cb.commit()
cb.waitUntilCompleted()

print("Features embedded and conditioned on timestep.")
print(sectionBreak)
