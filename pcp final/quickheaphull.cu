#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iomanip>
#include <cstdio>
#include <stdint.h>

using namespace std;

struct Point {
    float x, y;
};

__host__ __device__ int orientation(Point a, Point b, Point c) {
    float val = (b.y - a.y) * (c.x - b.x) -
                (b.x - a.x) * (c.y - b.y);
    if (fabs(val) < 1e-9) return 0;
    return (val > 0) ? 1 : 2;
}

__host__ __device__ float distance(Point a, Point b, Point c) {
    return fabs((b.y - a.y) * c.x - (b.x - a.x) * c.y + b.x * a.y - b.y * a.x) /
           sqrt((b.y - a.y)*(b.y - a.y) + (b.x - a.x)*(b.x - a.x));
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

__global__ void findFarthest(Point* d_points, Point a, Point b, int n, int* maxIdx, float* maxDist) {
    __shared__ float dist[256];
    __shared__ int idx[256];

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    float d = -1.0f;
    int myIdx = -1;

    if (i < n) {
        d = distance(a, b, d_points[i]);
        myIdx = i;
    }

    dist[tid] = d;
    idx[tid] = myIdx;

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && dist[tid] < dist[tid + s]) {
            dist[tid] = dist[tid + s];
            idx[tid] = idx[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMaxFloat(maxDist, dist[0]);
        *maxIdx = idx[0];
    }
}

__global__ void spatialFilter(Point* points, int* mask, int n, float minX, float maxX, float minY, float maxY) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= n) return;

    float marginX = 0.03f * (maxX - minX);
    float marginY = 0.03f * (maxY - minY);

    if (points[i].x <= minX + marginX || points[i].x >= maxX - marginX ||
        points[i].y <= minY + marginY || points[i].y >= maxY - marginY)
        mask[i] = 1;
    else
        mask[i] = 0;
}

void quickHullRec(vector<Point>& points, Point a, Point b, vector<Point>& hull) {
    if (points.empty()) return;

    int n = points.size();
    Point* d_points;
    int* d_maxIdx;
    float* d_maxDist;
    int h_maxIdx;
    float h_maxDist = -1.0f;

    cudaMalloc(&d_points, n * sizeof(Point));
    cudaMemcpy(d_points, points.data(), n * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMalloc(&d_maxIdx, sizeof(int));
    cudaMalloc(&d_maxDist, sizeof(float));
    cudaMemcpy(d_maxDist, &h_maxDist, sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    findFarthest<<<blocks, threadsPerBlock>>>(d_points, a, b, n, d_maxIdx, d_maxDist);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_maxIdx, d_maxIdx, sizeof(int), cudaMemcpyDeviceToHost);

    Point p = points[h_maxIdx];
    hull.push_back(p);

    vector<Point> leftSet1, leftSet2;
    for (int i = 0; i < n; i++) {
        if (i == h_maxIdx) continue;
        int o1 = orientation(a, p, points[i]);
        int o2 = orientation(p, b, points[i]);
        if (o1 == 2) leftSet1.push_back(points[i]);
        else if (o2 == 2) leftSet2.push_back(points[i]);
    }

    quickHullRec(leftSet1, a, p, hull);
    quickHullRec(leftSet2, p, b, hull);

    cudaFree(d_points);
    cudaFree(d_maxIdx);
    cudaFree(d_maxDist);
}

void quickHull(vector<Point>& points, vector<Point>& hull) {
    int n = points.size();
    if (n < 3) return;

    int minIdx = 0, maxIdx = 0;
    for (int i = 1; i < n; i++) {
        if (points[i].x < points[minIdx].x)
            minIdx = i;
        if (points[i].x > points[maxIdx].x)
            maxIdx = i;
    }

    Point a = points[minIdx];
    Point b = points[maxIdx];
    hull.push_back(a);
    hull.push_back(b);

    vector<Point> leftSet, rightSet;
    for (int i = 0; i < n; i++) {
        int o = orientation(a, b, points[i]);
        if (o == 2)
            leftSet.push_back(points[i]);
        else if (o == 1)
            rightSet.push_back(points[i]);
    }

    quickHullRec(leftSet, a, b, hull);
    quickHullRec(rightSet, b, a, hull);
}

int main() {
    cout << "\n========= RUNNING quickheaphull =========\n" << endl;

    vector<Point> points;
    FILE* fp = fopen("input.txt", "r");
    if (!fp) {
        cerr << "Error opening input.txt" << endl;
        return 1;
    }

    int n;
    fscanf(fp, "%d", &n);
    points.resize(n);
    for (int i = 0; i < n; i++) {
        fscanf(fp, "%f %f", &points[i].x, &points[i].y);
    }
    fclose(fp);

    cout << "Read " << n << " points from input.txt" << endl;

    for (int threadsPerBlock : {2, 4, 8, 16, 32, 64}) {
        cout << "\n========== THREADS: " << threadsPerBlock << " ==========" << endl;

        float minX = points[0].x, maxX = points[0].x;
        float minY = points[0].y, maxY = points[0].y;
        for (const auto& p : points) {
            minX = min(minX, p.x);
            maxX = max(maxX, p.x);
            minY = min(minY, p.y);
            maxY = max(maxY, p.y);
        }

        Point* d_points;
        int* d_mask;
        cudaMalloc(&d_points, n * sizeof(Point));
        cudaMalloc(&d_mask, n * sizeof(int));
        cudaMemcpy(d_points, points.data(), n * sizeof(Point), cudaMemcpyHostToDevice);

        int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
        spatialFilter<<<blocks, threadsPerBlock>>>(d_points, d_mask, n, minX, maxX, minY, maxY);
        cudaDeviceSynchronize();

        vector<int> h_mask(n);
        cudaMemcpy(h_mask.data(), d_mask, n * sizeof(int), cudaMemcpyDeviceToHost);

        vector<Point> filtered;
        for (int i = 0; i < n; i++) {
            if (h_mask[i] == 1) filtered.push_back(points[i]);
        }

        vector<Point> hull;
        auto t0 = chrono::high_resolution_clock::now();
        quickHull(filtered, hull);
        auto t1 = chrono::high_resolution_clock::now();
        chrono::duration<double> duration = t1 - t0;

        sort(hull.begin(), hull.end(), [](const Point& a, const Point& b) {
            return (a.x < b.x) || (a.x == b.x && a.y < b.y);
        });
        hull.erase(unique(hull.begin(), hull.end(), [](const Point& a, const Point& b) {
            return fabs(a.x - b.x) < 1e-6 && fabs(a.y - b.y) < 1e-6;
        }), hull.end());

        cout << "Time taken: " << fixed << setprecision(6) << duration.count() << " seconds" << endl;

        cudaFree(d_points);
        cudaFree(d_mask);
    }

    cout << "Done.\n" << endl;
    return 0;
}
