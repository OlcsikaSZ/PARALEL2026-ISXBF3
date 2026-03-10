__kernel void gol_step(__global const uchar* grid,
                       __global uchar* next,
                       const int rows,
                       const int cols,
                       const int wrap)
{
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= rows || y >= cols) return;

    int sum = 0;
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            if (dx == 0 && dy == 0) continue;
            int nx = x + dx;
            int ny = y + dy;
            if (wrap) {
                // wrap around (toroidal)
                if (nx < 0) nx += rows;
                if (nx >= rows) nx -= rows;
                if (ny < 0) ny += cols;
                if (ny >= cols) ny -= cols;
                sum += (int)grid[nx * cols + ny];
            } else {
                // fixed boundary (outside = 0)
                if (nx >= 0 && nx < rows && ny >= 0 && ny < cols) {
                    sum += (int)grid[nx * cols + ny];
                }
            }
        }
    }

    const int idx = x * cols + y;
    const uchar cell = grid[idx];
    uchar out = 0;
    if (cell) {
        out = (sum == 2 || sum == 3) ? 1 : 0;
    } else {
        out = (sum == 3) ? 1 : 0;
    }
    next[idx] = out;
}
