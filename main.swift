//
//  main.swift
//  MSLGraphDiffusion
//
//  Created by Raiden Makoto on 2026-02-07.
//

import Metal
import Foundation
import simd

let clock = ContinuousClock()
let timeElapsed = clock.measure {
    let sectionBreak = String(repeating: "=", count: 50)
    
    // --- CONFIGURATION ---
    let hiddenDim = 128
    let numLayers = 4
    let numTypes = 10
    
    // Hidden dimension from your PyTorch checkpoint
    var hDim = UInt32(hiddenDim)
    
    // --- BUFFER STORAGE ---
    var weights: [String: MTLBuffer] = [:]
    
    guard let device = MTLCreateSystemDefaultDevice() else { fatalError("Metal not supported") }
    let commandQueue = device.makeCommandQueue()!
    
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
                //print("PARTIAL LOAD: \(name).bin \(data.count)/\(expectedBytes) loaded. Padding with zeros")
                weights[name] = buffer
            } else {
               // print("SIZE MISMATCH: \(name) - File: \(data.count), Buffer Needs: \(expectedBytes)")
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
    var numNodes = 5 // Methane
    var numEdges = 8 // 4 bonds * 2
    let atomTypesCH4: [Int32] = [6, 1, 1, 1, 1]
    
    // DIFFUSION BUFFERS
    let alphasBuf = device.makeBuffer(length: 2500 * 4, options: .storageModeShared)!
    let alphasCumprodBuf = device.makeBuffer(length: 2500 * 4, options: .storageModeShared)!
    
    // Random Noise buffers
    let rngSize = numNodes * MemoryLayout<UInt64>.size * 2 // state + inc
    let rngBuf = device.makeBuffer(length: rngSize, options: .storageModeShared)!
    let rngPtr = rngBuf.contents().bindMemory(to: UInt64.self, capacity: numNodes * 2)

    for i in 0..<numNodes {
        rngPtr[i * 2] = UInt64.random(in: 0...UInt64.max)     // seed state
        rngPtr[i * 2 + 1] = UInt64(i * 2 + 1) | 1            // unique stream ID
    }
    
    // --- Pre-compute the Schedule
    let timesteps = 2500
    let betaStart: Float = 1e-4
    let betaEnd: Float = 0.02
    
    var alphas = [Float](repeating: 1.0, count: 2500)
    var alphasCumprod = [Float](repeating: 1.0, count: 2500)
    
    var currentCumprod: Float = 1.0
    for i in 0..<timesteps {
        let beta = betaStart + (betaEnd - betaStart) * (Float(i) / Float(timesteps - 1))
        let alpha = 1.0 - beta
        currentCumprod *= alpha
        
        alphas[i] = alpha
        alphasCumprod[i] = currentCumprod
    }
    
    alphasBuf.contents().copyMemory(from: alphas, byteCount: 2500 * 4)
    alphasCumprodBuf.contents().copyMemory(from: alphasCumprod, byteCount: 2500 * 4)
    
    // GRAPH BUFFERS
    let nodeBuf = device.makeBuffer(length: numNodes * MemoryLayout<Node>.stride, options: .storageModeShared)!
    let typeBuf = device.makeBuffer(length: numNodes * 4, options: .storageModeShared)!
    let edgeBuf = device.makeBuffer(length: numEdges * MemoryLayout<SIMD2<Int32>>.stride, options: .storageModeShared)!
    let coordTempBuf = device.makeBuffer(length: numEdges * hiddenDim * 4, options: .storageModeShared)!
    
    // WORKSPACE BUFFERS
    // 1. Features & Messages
    let hBuf = device.makeBuffer(length: numNodes * hiddenDim * 4, options: .storageModeShared)!
    let tempH = device.makeBuffer(length: max(numNodes, numEdges) * hiddenDim * 4, options: .storageModeShared)!
    let msgBuf = device.makeBuffer(length: numEdges * hiddenDim * 4, options: .storageModeShared)!
    let msgAggBuf = device.makeBuffer(length: numNodes * hiddenDim * 4, options: .storageModeShared)!
    
    // 2. Coordinates
    let coordScalarBuf = device.makeBuffer(length: numEdges * 4, options: .storageModeShared)! // 1 float per edge
    let transBuf = device.makeBuffer(length: numEdges * 3 * 4, options: .storageModeShared)! // 3 floats per edge
    let posAggBuf = device.makeBuffer(length: numNodes * 3 * 4, options: .storageModeShared)! // 3 floats per node
    let posInputBuf = device.makeBuffer(length: numNodes * 3 * 4, options: .storageModePrivate)!
    let cogBuf = device.makeBuffer(length: 3 * 4, options: .storageModeShared)!
    
    // 3. Timestep specific buffers (CRITICAL for the loop)
    let tSinBuf = device.makeBuffer(length: hiddenDim * 4, options: .storageModeShared)!
    let tTempBuf = device.makeBuffer(length: hiddenDim * 4, options: .storageModeShared)! // Missing in previous version!
    let tProcessedBuf = device.makeBuffer(length: hiddenDim * 4, options: .storageModeShared)!
    
    // 4. Update Residual
    let hUpdateBuf = device.makeBuffer(length: numNodes * hiddenDim * 4, options: .storageModeShared)!
    
    // --- INITIALIZATION ---
    // Initialize Nodes with Noise
    var noiseNodes = [Node](repeating: Node(pos: .init(0,0,0), atomType: 0), count: numNodes)
    let initialSigma = Float(1.67)
    for i in 0..<numNodes {
        let randomPos = SIMD3<Float>(
            Float.random(in: -1...1) * initialSigma,
            Float.random(in: -1...1) * initialSigma,
            Float.random(in: -1...1) * initialSigma
        )
        noiseNodes[i] = Node(pos: randomPos, atomType: Float(atomTypesCH4[i]))
    }
    nodeBuf.contents().copyMemory(from: noiseNodes, byteCount: numNodes * MemoryLayout<Node>.stride)
    typeBuf.contents().copyMemory(from: atomTypesCH4, byteCount: numNodes * 4)
    
    // Initialize Edges
    let edges: [SIMD2<Int32>] = [SIMD2(0,1), SIMD2(1,0), SIMD2(0,2), SIMD2(2,0), SIMD2(0,3), SIMD2(3,0), SIMD2(0,4), SIMD2(4,0)]
    edgeBuf.contents().copyMemory(from: edges, byteCount: numEdges * MemoryLayout<SIMD2<Int32>>.stride)
    
    // --- PIPELINE SETUP ---
    guard let library = device.makeDefaultLibrary() else { fatalError() }
    var pipeline: [String: MTLComputePipelineState] = [:]
    
    // Register all required kernels
    let allKernels = [
        "embed_atoms", "inject_timestamp",
        "compute_message", "compute_displacement", "compute_node",
        "aggregate", "apply_diffusion",
        "force_zero_center", "add_residual", "clear_buffer_float",
        "linear_128x128", "linear_128x1" // The new robust kernels
    ]
    
    for name in allKernels {
        guard let fn = library.makeFunction(name: name) else { fatalError("Kernel \(name) not found in Library!") }
        pipeline[name] = try! device.makeComputePipelineState(function: fn)
    }
    
    print("Buffers & Pipelines Ready.")
    print(sectionBreak)
    
    // --- DIFFUSION LOOP --
    let halfDim = hiddenDim / 2
    let exponentBase = log(10000.0) / Float(halfDim - 1)
    
    // SAFETY FLAGS (UInt32 for 4-byte alignment)
    var doSiLU: UInt32 = 1
    var skipSiLU: UInt32 = 0
    var oneItem: UInt32 = 1
    var nEdgesU = UInt32(numEdges)
    var nNodesU = UInt32(numNodes)
    
    print("STARTING DIFFUSION LOOP (2500 -> 1)")
    
    let initCB = commandQueue.makeCommandBuffer()!
    let initEnc = initCB.makeComputeCommandEncoder()!
    initEnc.setComputePipelineState(pipeline["embed_atoms"]!)
    initEnc.setBuffer(typeBuf, offset: 0, index: 0); initEnc.setBuffer(weights["embedding.weight"]!, offset: 0, index: 1)
    initEnc.setBuffer(hBuf, offset: 0, index: 2);
    initEnc.setBytes(&hDim, length: 4, index: 3);
    initEnc.setBytes(&nNodesU, length: 4, index: 4)
    initEnc.dispatchThreads(MTLSize(width: numNodes, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    initEnc.endEncoding()
    initCB.commit()
    initCB.waitUntilCompleted()
    
    let baseHBuf = device.makeBuffer(length: hBuf.length, options: .storageModePrivate)!
    
    let copyCB = commandQueue.makeCommandBuffer()!
    let blitInit = copyCB.makeBlitCommandEncoder()!
    blitInit.copy(from: hBuf, sourceOffset: 0, to: baseHBuf, destinationOffset: 0, size: hBuf.length)
    blitInit.endEncoding()
    copyCB.commit()
    copyCB.waitUntilCompleted()
    
    for t in (0...2499).reversed() {
        let currentT = Float(t)
        let resetCB = commandQueue.makeCommandBuffer()!
        let blit = resetCB.makeBlitCommandEncoder()!
        blit.copy(from: baseHBuf, sourceOffset: 0, to: hBuf, destinationOffset: 0, size: hBuf.length)
        for i in 0..<numNodes {
            blit.copy(
                from: nodeBuf, sourceOffset: i * MemoryLayout<Node>.stride,
                to: posInputBuf, destinationOffset: i * 12, size: 12)
        }
        blit.endEncoding()
        resetCB.commit()
        
        // 1. UPDATE FREQUENCIES (CPU)
        var sinData = [Float](repeating: 0, count: hiddenDim)
        for i in 0..<halfDim {
            let freq = exp(Float(i) * -exponentBase)
            let arg = currentT * freq
            sinData[i] = sin(arg); sinData[i + halfDim] = cos(arg)
        }
        tSinBuf.contents().copyMemory(from: sinData, byteCount: hiddenDim * 4)
        
        let stepCB = commandQueue.makeCommandBuffer()!
        
        // Clear Aggregators
        let blit2 = stepCB.makeBlitCommandEncoder()!
        blit2.fill(buffer: msgAggBuf, range: 0..<msgAggBuf.length, value: 0)
        blit2.fill(buffer: posAggBuf, range: 0..<posAggBuf.length, value: 0)
        blit2.fill(buffer: cogBuf, range: 0..<cogBuf.length, value: 0)
        blit2.endEncoding()
        
        let enc = stepCB.makeComputeCommandEncoder()!
        
        // --- TIMESTEP MLP ---
        // Stage 1: Sin -> Temp
        enc.setComputePipelineState(pipeline["linear_128x128"]!)
        enc.setBuffer(tSinBuf, offset: 0, index: 0)
        enc.setBuffer(weights["timestep_mlp.0.weight"]!, offset: 0, index: 1)
        enc.setBuffer(weights["timestep_mlp.0.bias"]!, offset: 0, index: 2)
        enc.setBuffer(tTempBuf, offset: 0, index: 3) // Output to Temp
        enc.setBytes(&oneItem, length: 4, index: 4) // Count = 1
        enc.setBytes(&doSiLU, length: 4, index: 5)
        enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        
        // Stage 2: Temp -> Processed
        enc.setBuffer(tTempBuf, offset: 0, index: 0)
        enc.setBuffer(weights["timestep_mlp.2.weight"]!, offset: 0, index: 1)
        enc.setBuffer(weights["timestep_mlp.2.bias"]!, offset: 0, index: 2)
        enc.setBuffer(tProcessedBuf, offset: 0, index: 3)
        enc.setBytes(&skipSiLU, length: 4, index: 5)
        enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        
        enc.setComputePipelineState(pipeline["inject_timestamp"]!)
        enc.setBuffer(hBuf, offset: 0, index: 0); enc.setBuffer(tProcessedBuf, offset: 0, index: 1)
        enc.setBytes(&hDim, length: 4, index: 2); enc.setBytes(&nNodesU, length: 4, index: 3)
        enc.dispatchThreads(MTLSize(width: numNodes, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

        // --- EGNN LAYERS ---
        for i in 0..<numLayers {
            // A. CLEAR MESSAGE AGGREGATOR (This IS per layer)
            enc.setComputePipelineState(pipeline["clear_buffer_float"]!)
            enc.setBuffer(msgAggBuf, offset: 0, index: 0)
            var mCount = nNodesU * hDim
            enc.setBytes(&mCount, length: 4, index: 1)
            enc.dispatchThreads(MTLSize(width: Int(mCount), height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            
            enc.memoryBarrier(scope: .buffers)
            
            // B. MESSAGE MLP (Standard)
            enc.setComputePipelineState(pipeline["compute_message"]!)
            // ... (keep your existing buffer setting code) ...
            enc.setBuffer(hBuf, offset: 0, index: 0); enc.setBuffer(nodeBuf, offset: 0, index: 1); enc.setBuffer(edgeBuf, offset: 0, index: 2)
            enc.setBuffer(weights["layers.\(i).message_mlp.0.weight"]!, offset: 0, index: 3)
            enc.setBuffer(weights["layers.\(i).message_mlp.0.bias"]!, offset: 0, index: 4)
            enc.setBuffer(tempH, offset: 0, index: 5)
            enc.setBytes(&hDim, length: 4, index: 6)
            enc.dispatchThreads(MTLSize(width: numEdges, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            
            // ... (keep linear_128x128 stage 2) ...
            enc.setComputePipelineState(pipeline["linear_128x128"]!)
            enc.setBuffer(tempH, offset: 0, index: 0)
            enc.setBuffer(weights["layers.\(i).message_mlp.2.weight"]!, offset: 0, index: 1)
            enc.setBuffer(weights["layers.\(i).message_mlp.2.bias"]!, offset: 0, index: 2)
            enc.setBuffer(msgBuf, offset: 0, index: 3)
            enc.setBytes(&nEdgesU, length: 4, index: 4); enc.setBytes(&doSiLU, length: 4, index: 5)
            enc.dispatchThreads(MTLSize(width: numEdges, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

            enc.memoryBarrier(scope: .buffers)

            // C. COORD MLP (Standard)
            // ... (keep your existing coord mlp code) ...
            enc.setComputePipelineState(pipeline["linear_128x128"]!)
            enc.setBuffer(msgBuf, offset: 0, index: 0)
            enc.setBuffer(weights["layers.\(i).coord_mlp.0.weight"]!, offset: 0, index: 1)
            enc.setBuffer(weights["layers.\(i).coord_mlp.0.bias"]!, offset: 0, index: 2)
            enc.setBuffer(coordTempBuf, offset: 0, index: 3)
            enc.setBytes(&nEdgesU, length: 4, index: 4); enc.setBytes(&doSiLU, length: 4, index: 5)
            enc.dispatchThreads(MTLSize(width: numEdges, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            
            enc.setComputePipelineState(pipeline["linear_128x1"]!)
            enc.setBuffer(coordTempBuf, offset: 0, index: 0)
            enc.setBuffer(weights["layers.\(i).coord_mlp.2.weight"]!, offset: 0, index: 1)
            enc.setBuffer(weights["layers.\(i).coord_mlp.2.bias"]!, offset: 0, index: 2)
            enc.setBuffer(coordScalarBuf, offset: 0, index: 3)
            enc.setBytes(&nEdgesU, length: 4, index: 4); enc.setBytes(&skipSiLU, length: 4, index: 5)
            enc.dispatchThreads(MTLSize(width: numEdges, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            
            // D. DISPLACEMENT & AGGREGATE
            enc.setComputePipelineState(pipeline["compute_displacement"]!)
            enc.setBuffer(coordScalarBuf, offset: 0, index: 0); enc.setBuffer(nodeBuf, offset: 0, index: 1)
            enc.setBuffer(edgeBuf, offset: 0, index: 2); enc.setBuffer(transBuf, offset: 0, index: 3)
            enc.setBytes(&nEdgesU, length: 4, index: 4)
            enc.dispatchThreads(MTLSize(width: numEdges, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            
            // C. AGGREGATE (Updated for Serial Safety)
            enc.setComputePipelineState(pipeline["aggregate"]!)
            enc.setBuffer(msgBuf, offset: 0, index: 0)
            enc.setBuffer(transBuf, offset: 0, index: 1)
            enc.setBuffer(edgeBuf, offset: 0, index: 2)
            enc.setBuffer(msgAggBuf, offset: 0, index: 3)
            enc.setBuffer(posAggBuf, offset: 0, index: 4)
            enc.setBytes(&hDim, length: 4, index: 5)
            enc.setBytes(&nEdgesU, length: 4, index: 6)
            
            // CRITICAL CHANGE: Dispatch numNodes, not numEdges
            enc.dispatchThreads(MTLSize(width: numNodes, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            
            // E. NODE MLP (Standard)
            // ... (keep your existing node mlp code) ...
            enc.setComputePipelineState(pipeline["compute_node"]!)
            enc.setBuffer(hBuf, offset: 0, index: 0); enc.setBuffer(msgAggBuf, offset: 0, index: 1)
            enc.setBuffer(weights["layers.\(i).node_mlp.0.weight"]!, offset: 0, index: 2)
            enc.setBuffer(weights["layers.\(i).node_mlp.0.bias"]!, offset: 0, index: 3)
            enc.setBuffer(tempH, offset: 0, index: 4)
            enc.setBytes(&hDim, length: 4, index: 5)
            enc.dispatchThreads(MTLSize(width: numNodes, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            
            enc.setComputePipelineState(pipeline["linear_128x128"]!)
            enc.setBuffer(tempH, offset: 0, index: 0)
            enc.setBuffer(weights["layers.\(i).node_mlp.2.weight"]!, offset: 0, index: 1)
            enc.setBuffer(weights["layers.\(i).node_mlp.2.bias"]!, offset: 0, index: 2)
            enc.setBuffer(hUpdateBuf, offset: 0, index: 3)
            enc.setBytes(&nNodesU, length: 4, index: 4); enc.setBytes(&skipSiLU, length: 4, index: 5)
            enc.dispatchThreads(MTLSize(width: numNodes, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

            enc.memoryBarrier(scope: .buffers)

            // F. UPDATE FEATURES (CRITICAL MISSING STEP)
            // hBuf = hBuf + hUpdateBuf
            enc.setComputePipelineState(pipeline["add_residual"]!)
            enc.setBuffer(hBuf, offset: 0, index: 0)
            enc.setBuffer(hUpdateBuf, offset: 0, index: 1)
            var hCount = nNodesU * hDim
            enc.setBytes(&hCount, length: 4, index: 2)
            enc.dispatchThreads(MTLSize(width: Int(hCount), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        }
        
        enc.memoryBarrier(scope: .buffers)
        
        // E. UPDATE ONCE OUTSIDE THE LOOP
        enc.setComputePipelineState(pipeline["apply_diffusion"]!)
        enc.setBuffer(nodeBuf, offset: 0, index: 0)
        enc.setBuffer(alphasBuf, offset: 0, index: 1)
        enc.setBuffer(alphasCumprodBuf, offset: 0, index: 2)
        enc.setBuffer(posAggBuf, offset: 0, index: 3) // Original x_t
        var tUint = UInt32(t)
        enc.setBytes(&tUint, length: 4, index: 4)
        enc.setBytes(&nNodesU, length: 4, index: 5)
        enc.setBuffer(rngBuf, offset: 0, index: 6)
        enc.dispatchThreads(MTLSize(width: numNodes, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        
        enc.memoryBarrier(scope: .buffers)
        
        // --- CoG NORMALIZATION ---
        enc.setComputePipelineState(pipeline["force_zero_center"]!)
        enc.setBuffer(nodeBuf, offset: 0, index: 0)
        enc.setBytes(&nNodesU, length: 4, index: 1)
        // Dispatch exactly ONE thread to handle the whole molecule
        enc.dispatchThreads(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1)
        )
        
        enc.memoryBarrier(scope: .buffers)
        
        enc.endEncoding()
        stepCB.commit()
        stepCB.waitUntilCompleted()
        if (t % 100 == 0){
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
        }
    }
    
    print(sectionBreak)
    print("Generation Complete.")
}
print("Total runtime: \((timeElapsed).formatted(.time(pattern: .minuteSecond)))")
