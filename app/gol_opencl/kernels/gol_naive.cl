// Compute one Game of Life generation directly from global memory.
__kernel void gol_step(__global const uchar* grid,
                       __global uchar* next,
                       const int rows,
                       const int cols,
                       const int wrap)
{
    // Map this work-item to one output cell.
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= rows || y >= cols) return;

    int sum = 0;
    // Visit the 3x3 neighborhood and skip the center cell itself.
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            if (dx == 0 && dy == 0) continue;
            int nx = x + dx;
            int ny = y + dy;
            if (wrap) {
                // Wrap neighbors around the grid edges like a torus.
                if (nx < 0) nx += rows;
                if (nx >= rows) nx -= rows;
                if (ny < 0) ny += cols;
                if (ny >= cols) ny -= cols;
                sum += (int)grid[nx * cols + ny];
            } else {
                // Treat out-of-range neighbors as dead cells.
                if (nx >= 0 && nx < rows && ny >= 0 && ny < cols) {
                    sum += (int)grid[nx * cols + ny];
                }
            }
        }
    }

    // Convert the 2D coordinate into a linear buffer index.
    const int idx = x * cols + y;
    const uchar cell = grid[idx];
    uchar out = 0;
    if (cell) {
        // Live cells survive only with two or three neighbors.
        out = (sum == 2 || sum == 3) ? 1 : 0;
    } else {
        // Dead cells are born only when exactly three neighbors are alive.
        out = (sum == 3) ? 1 : 0;
    }
    // Store the next-generation state for this cell.
    next[idx] = out;
}
