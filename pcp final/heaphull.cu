#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>

#define MAX_POINTS 1000000
#define THREADS_PER_BLOCK 256

struct Point {
    float x, y;
};

__device__ int isInterior(Point p, Point a, Point b, Point c) {
    float cross1 = (b.x - a.x)*(p.y - a.y) - (b.y - a.y)*(p.x - a.x);
    float cross2 = (c.x - b.x)*(p.y - b.y) - (c.y - b.y)*(p.x - b.x);
    float cross3 = (a.x - c.x)*(p.y - c.y) - (a.y - c.y)*(p.x - c.x);
    return (cross1 < 0 && cross2 < 0 && cross3 < 0);
}

__global__ void filterKernel(Point* input, int n, int* flags, Point a, Point b, Point c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        Point p = input[idx];
        flags[idx] = !isInterior(p, a, b, c);
    }
}

bool compare(Point a, Point b) {
    return a.x < b.x || (a.x == b.x && a.y < b.y);
}

int orientation(Point p, Point q, Point r) {
    float val = (q.y - p.y)*(r.x - q.x) - (q.x - p.x)*(r.y - q.y);
    if (val == 0) return 0;
    return (val > 0) ? 1 : 2;
}

std::vector<Point> grahamScan(std::vector<Point>& points) {
    int n = points.size();
    if (n < 3) return {};

    std::sort(points.begin(), points.end(), compare);

    std::vector<Point> hull(2 * n);
    int k = 0;

    // Lower hull
    for (int i = 0; i < n; ++i) {
        while (k >= 2 && orientation(hull[k - 2], hull[k - 1], points[i]) != 2)
            k--;
        hull[k++] = points[i];
    }

    // Upper hull
    for (int i = n - 2, t = k + 1; i >= 0; --i) {
        while (k >= t && orientation(hull[k - 2], hull[k - 1], points[i]) != 2)
            k--;
        hull[k++] = points[i];
    }

    hull.resize(k - 1);
    return hull;
}

void readPoints(const std::string& filename, std::vector<Point>& points) {
    std::ifstream infile(filename);
    float x, y;
    while (infile >> x >> y) {
        points.push_back({x, y});
    }
}

int main() {
    std::vector<Point> input;
    readPoints("input.txt", input);
    int n = input.size();
    std::cout << "Read " << n << " points from input.txt\n";

    // Triangle filter anchor points
    Point a = {0, 0}, b = {1000, 0}, c = {500, 1000};

    Point* d_points;
    int* d_flags;
    cudaMalloc(&d_points, n * sizeof(Point));
    cudaMalloc(&d_flags, n * sizeof(int));
    cudaMemcpy(d_points, input.data(), n * sizeof(Point), cudaMemcpyHostToDevice);

    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    filterKernel<<<blocks, THREADS_PER_BLOCK>>>(d_points, n, d_flags, a, b, c);
    cudaDeviceSynchronize();

    std::vector<int> flags(n);
    cudaMemcpy(flags.data(), d_flags, n * sizeof(int), cudaMemcpyDeviceToHost);

    std::vector<Point> filtered;
    for (int i = 0; i < n; ++i) {
        if (flags[i]) filtered.push_back(input[i]);
    }

    std::cout << "Filtering complete: " << filtered.size() << " points retained after compaction.\n";

    for (int threads : {2, 4, 8, 16, 32, 64}) {
        std::cout << "\n========== THREADS: " << threads << " ==========\n";

        auto start = std::chrono::high_resolution_clock::now();
        std::vector<Point> hull = grahamScan(filtered);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;

        std::ofstream outfile("output.txt");
        for (auto& p : hull) {
            outfile << p.x << " " << p.y << "\n";
        }
        outfile << "\nTime taken: " << duration.count() << " seconds\n";

        std::cout << "Time taken: " << duration.count() << " seconds\n";
    }

    cudaFree(d_points);
    cudaFree(d_flags);
    std::cout << "\nDone.\n";
    return 0;
}
