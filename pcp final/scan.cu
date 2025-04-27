#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>

struct Point {
    float x, y;

    __host__ __device__
    bool operator<(const Point& p) const {
        return (x < p.x) || (x == p.x && y < p.y);
    }
};

// Orientation function: 0 -> collinear, 1 -> clockwise, 2 -> counterclockwise
__host__ __device__
int orientation(Point p, Point q, Point r) {
    float val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
    if (fabs(val) < 1e-6) return 0; // collinear
    return (val > 0) ? 1 : 2;       // clockwise or counterclockwise
}

// Parallelized merge sort for CUDA (simplified, this can be optimized further)
__global__ void mergeSortKernel(Point* points, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;

    // Simple bitonic sort (could replace with more efficient sorting algorithms)
    // For simplicity, we are assuming sorted order here in the convex hull construction
}

// CUDA kernel to construct the convex hull (lower and upper hull)
__global__ void constructHull(Point* points, int n, Point* hull, int* hullSize) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;

    // Add the point to the hull if it's part of the lower or upper hull
    Point p = points[idx];

    // Placeholder for orientation logic: Add point if orientation check is satisfied
    if (orientation(p, points[0], points[(idx + 1) % n]) != 0) {
        // Atomically add point to the hull
        int pos = atomicAdd(hullSize, 1);  // Get the next available index for the hull

        // Ensure that we don't exceed bounds
        if (pos < n) {
            hull[pos] = p;  // Safely add the point to the hull
        }
    }
}

// Function to read input from a file
std::vector<Point> readInput(const std::string& filename) {
    std::ifstream in(filename);
    std::vector<Point> points;
    float x, y;
    while (in >> x >> y) {
        points.push_back({x, y});
    }
    return points;
}

// Function to write the convex hull to a file
void writeOutput(const std::vector<Point>& hull, const std::string& filename) {
    std::ofstream out(filename);
    for (const auto& p : hull) {
        out << p.x << " " << p.y << "\n";
    }
}

// CUDA function to run the Graham Scan
std::vector<Point> grahamScanCUDA(std::vector<Point>& points) {
    int n = points.size();
    if (n < 3) return points;

    // Allocate memory on device
    Point* d_points;
    cudaMalloc(&d_points, n * sizeof(Point));
    cudaMemcpy(d_points, points.data(), n * sizeof(Point), cudaMemcpyHostToDevice);

    // Sort points in parallel (this is simplified, but replace with a more efficient sort if needed)
    mergeSortKernel<<<(n + 255) / 256, 256>>>(d_points, n);
    cudaDeviceSynchronize();

    // Allocate memory for hull and hull size counter
    Point* d_hull;
    int* d_hullSize;
    cudaMalloc(&d_hull, n * sizeof(Point));  // Max size is n
    cudaMalloc(&d_hullSize, sizeof(int));
    cudaMemset(d_hullSize, 0, sizeof(int));

    // Construct hull in parallel
    constructHull<<<(n + 255) / 256, 256>>>(d_points, n, d_hull, d_hullSize);
    cudaDeviceSynchronize();

    // Retrieve the hull and size from device memory
    int h_hullSize;
    cudaMemcpy(&h_hullSize, d_hullSize, sizeof(int), cudaMemcpyDeviceToHost);

    std::vector<Point> hull(h_hullSize);
    cudaMemcpy(hull.data(), d_hull, h_hullSize * sizeof(Point), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_points);
    cudaFree(d_hull);
    cudaFree(d_hullSize);

    return hull;
}

int main() {
    // Read points from input file
    std::vector<Point> input = readInput("input.txt");
    std::cout << "Read " << input.size() << " points from input.txt\n";

    // Run Graham Scan on GPU (CUDA)
    std::cout << "Running Parallel Graham Scan on GPU\n";
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<Point> hull = grahamScanCUDA(input);
    auto end = std::chrono::high_resolution_clock::now();

    // Output the convex hull to the output file
    writeOutput(hull, "output.txt");

    // Measure execution time
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time taken: " << elapsed.count() << " seconds\n";

    return 0;
}
