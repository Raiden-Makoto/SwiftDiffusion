//
//  QM9Loader.swift
//  MSLGraphDiffusion
//
//  Created by Raiden Makoto on 2026-02-05.
//

import Metal
import Foundation

class QM9Loader {
    let device: MTLDevice
    // GPU-accessible buffers
    var nodeBuffer: MTLBuffer?
    var edgeBuffer: MTLBuffer?
    var graphdataBuffer: MTLBuffer?
    
    var graphCount: Int = 0
    
    init(device: MTLDevice){
        self.device = device
    }
    
    func load(from directoryPath: String) throws {
        let baseUrl = URL(fileURLWithPath: directoryPath)
        // load nodes
        let nodesData = try Data(contentsOf: baseUrl.appendingPathComponent("qm9_nodes.bin"), options: .mappedIfSafe)
        nodeBuffer = device.makeBuffer(bytes: (nodesData as NSData).bytes, length: nodesData.count, options: .storageModeShared)
        // load edges
        let edgesData = try Data(contentsOf: baseUrl.appendingPathComponent("qm9_edges.bin"), options: .mappedIfSafe)
        edgeBuffer = device.makeBuffer(bytes: (edgesData as NSData).bytes, length: edgesData.count, options: .storageModeShared)
        // load graphdata
        let graphData = try Data(contentsOf: baseUrl.appendingPathComponent("qm9_metadata.bin"), options: .mappedIfSafe)
        graphdataBuffer = device.makeBuffer(bytes: (graphData as NSData).bytes, length: graphData.count, options: .storageModeShared)
        // how many molecules
        self.graphCount = graphData.count / MemoryLayout<GraphData>.stride
        print("Loaded QM9 dataset with \(self.graphCount) molecules to Unified Memory.")
    }
}

