#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <algorithm>
using namespace std;

struct Point {
    float x, y;
    bool operator<(const Point& p) const {
        return x < p.x || (x == p.x && y < p.y);
    }
};

int orientation(Point p, Point q, Point r) {
    float val = (q.y - p.y) * (r.x - q.x) -
                (q.x - p.x) * (r.y - q.y);
    if (val == 0) return 0;
    return (val > 0) ? 1 : 2;
}

vector<Point> incrementalConvexHull(vector<Point>& points) {
    int n = points.size();
    if (n < 3) return {};

    sort(points.begin(), points.end());
    vector<Point> lower, upper;

    for (int i = 0; i < n; ++i) {
        while (lower.size() >= 2 &&
               orientation(lower[lower.size() - 2], lower[lower.size() - 1], points[i]) != 2)
            lower.pop_back();
        lower.push_back(points[i]);
    }

    for (int i = n - 1; i >= 0; --i) {
        while (upper.size() >= 2 &&
               orientation(upper[upper.size() - 2], upper[upper.size() - 1], points[i]) != 2)
            upper.pop_back();
        upper.push_back(points[i]);
    }

    lower.pop_back();
    upper.pop_back();

    lower.insert(lower.end(), upper.begin(), upper.end());
    return lower;
}

void runIncremental(int threads) {
    cout << "\n========== THREADS: " << threads << " ==========" << endl;

    ifstream input("input.txt");
    int n;
    input >> n;
    vector<Point> points(n);
    for (int i = 0; i < n; ++i) {
        input >> points[i].x >> points[i].y;
    }

    auto start = chrono::high_resolution_clock::now();
    vector<Point> hull = incrementalConvexHull(points);
    auto end = chrono::high_resolution_clock::now();

    double time_taken = chrono::duration<double>(end - start).count();
    cout << "Time taken: " << time_taken << " seconds" << endl;
}

int main() {
    cout << "Read points from input.txt" << endl;
    for (int threads : {2, 4, 8, 16, 32, 64}) {
        runIncremental(threads);
    }
    cout << "Done.\n";
    return 0;
}
