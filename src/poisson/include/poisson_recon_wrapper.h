#pragma once
#include <vector>
#include <string>

struct PipelinePoint {
    double x, y, z;
    double nx, ny, nz;
};

struct PipelineMesh {
    struct Vertex {
        double x, y, z;
        double value; // density value
    };
    std::vector<Vertex> vertices;
    std::vector<std::vector<int>> faces;
};

struct PoissonParams {
    int depth = 10;
    int solveDepth = -1;
    int kernelDepth = -1;
    double samplesPerNode = 1.5;
    double pointWeight = 2.0;
    double scale = 1.1;
    int degree = 1;
    int bType = 2; // Neumann
    bool verbose = false;
};

struct TrimParams {
    double trimThreshold = 7.0;
    double islandAreaRatio = 0.001;
    bool removeIslands = true;
    bool verbose = false;
};

// Runs Screened Poisson Reconstruction in memory and returns untrimmed watertight mesh (Stage 3)
PipelineMesh run_poisson_reconstruction(
    const std::vector<PipelinePoint>& points,
    const PoissonParams& poisson_params
);

// Trims a watertight mesh in memory based on vertex densities (Stage 4)
PipelineMesh trim_mesh(
    const PipelineMesh& input_mesh,
    const TrimParams& trim_params
);
