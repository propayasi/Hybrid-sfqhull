#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include <algorithm>

struct Point {
    float x, y;
    bool operator==(const Point& other) const {
        return fabs(x - other.x) < 1e-6 && fabs(y - other.y) < 1e-6;
    }
};

__host__ __device__
float pointLineDistance(Point A, Point B, Point C) {
    return fabs((C.y - A.y) * (B.x - A.x) - (B.y - A.y) * (C.x - A.x));
}

__device__
float getDistanceFromLine(Point p1, Point p2, Point p) {
    return pointLineDistance(p1, p2, p);
}

__device__ float atomicMaxFloat(float* addr, float value) {
    int* address_as_int = (int*)addr;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                       __float_as_int(fmaxf(value, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__
void findFarthestPointKernel(Point* d_points, int n, Point p1, Point p2, int* d_index, float* d_maxDist) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;

    float dist = getDistanceFromLine(p1, p2, d_points[idx]);
    float oldMax = atomicMaxFloat(d_maxDist, dist);
    
    if (dist > oldMax) {
        atomicExch(d_index, idx);
    }
}

std::vector<Point> readInput(const std::string& filename) {
    std::ifstream in(filename);
    std::vector<Point> points;
    float x, y;
    while (in >> x >> y) {
        points.push_back({x, y});
    }
    return points;
}

void writeOutput(const std::vector<Point>& hull, const std::string& filename) {
    std::ofstream out(filename);
    for (const auto& p : hull) {
        out << p.x << " " << p.y << "\n";
    }
}

int orientation(Point p, Point q, Point r) {
    float val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
    if (fabs(val) < 1e-6) return 0;
    return (val > 0) ? 1 : 2;
}

void quickHull(std::vector<Point>& points, Point p1, Point p2, int side, std::vector<Point>& hull, int threadsPerBlock) {
    int n = points.size();
    if (n == 0) {
        if (hull.empty() || !(hull.back().x == p2.x && hull.back().y == p2.y)) {
            hull.push_back(p2);
        }
        return;
    }

    Point* d_points;
    int* d_index;
    float* d_maxDist;

    cudaMalloc(&d_points, n * sizeof(Point));
    cudaMemcpy(d_points, points.data(), n * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMalloc(&d_index, sizeof(int));
    cudaMalloc(&d_maxDist, sizeof(float));

    float initialMaxDist = -INFINITY;
    cudaMemcpy(d_maxDist, &initialMaxDist, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_index, -1, sizeof(int));

    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    findFarthestPointKernel<<<blocks, threadsPerBlock>>>(d_points, n, p1, p2, d_index, d_maxDist);
    cudaDeviceSynchronize();

    int idx;
    float maxDist;
    cudaMemcpy(&idx, d_index, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&maxDist, d_maxDist, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_points);
    cudaFree(d_index);
    cudaFree(d_maxDist);

    if (idx == -1) {
        if (std::find(hull.begin(), hull.end(), p2) == hull.end()) {
            hull.push_back(p2);
        }
        return;
    }

    Point farthest = points[idx];
    std::vector<Point> leftSet1, leftSet2;
    for (const auto& pt : points) {
        if (orientation(p1, farthest, pt) == side)
            leftSet1.push_back(pt);
        if (orientation(farthest, p2, pt) == side)
            leftSet2.push_back(pt);
    }

    quickHull(leftSet1, p1, farthest, side, hull, threadsPerBlock);
    quickHull(leftSet2, farthest, p2, side, hull, threadsPerBlock);
}

std::vector<Point> computeQuickHull(std::vector<Point>& points, int threadsPerBlock) {
    std::vector<Point> hull;
    if (points.size() < 3) return points;

    // Find min and max x points
    int minIdx = 0, maxIdx = 0;
    for (size_t i = 1; i < points.size(); ++i) {
        if (points[i].x < points[minIdx].x) minIdx = i;
        if (points[i].x > points[maxIdx].x) maxIdx = i;
    }

    Point minPoint = points[minIdx];
    Point maxPoint = points[maxIdx];

    // Split points into two sets
    std::vector<Point> leftSet, rightSet;
    for (const auto& pt : points) {
        int o = orientation(minPoint, maxPoint, pt);
        if (o == 2) leftSet.push_back(pt);
        else if (o == 1) rightSet.push_back(pt);
    }

    // Compute hull recursively
    hull.push_back(minPoint);
    quickHull(leftSet, minPoint, maxPoint, 2, hull, threadsPerBlock);
    quickHull(rightSet, maxPoint, minPoint, 1, hull, threadsPerBlock);

    // Remove duplicates while preserving order
    std::vector<Point> uniqueHull;
    for (const auto& p : hull) {
        if (std::find(uniqueHull.begin(), uniqueHull.end(), p) == uniqueHull.end()) {
            uniqueHull.push_back(p);
        }
    }

    // Sort by x then y
    std::sort(uniqueHull.begin(), uniqueHull.end(), [](const Point& a, const Point& b) {
        return (a.x < b.x) || (a.x == b.x && a.y < b.y);
    });

    return uniqueHull;
}

int main() {
    std::cout << "========= RUNNING quickhull =========\n";
    std::vector<Point> points = readInput("input.txt");
    std::cout << "Read " << points.size() << " points from input.txt\n";

    for (int threads : {2, 4, 8, 16, 32, 64}) {
        std::cout << "========== THREADS: " << threads << " ==========\n";
        
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<Point> hull = computeQuickHull(points, threads);
        auto end = std::chrono::high_resolution_clock::now();
        
        if (threads == 64) {
            writeOutput(hull, "output.txt");
        }
        
        std::chrono::duration<double> duration = end - start;
        std::cout << "Time taken: " << duration.count() << " seconds\n";
    }

    std::cout << "Done.\n";
    return 0;
}