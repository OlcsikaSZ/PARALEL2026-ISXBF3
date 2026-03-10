// Game of Life - tiled (local memory) version
// Works with arbitrary 2D local sizes (lx, ly) chosen on the host.
// Uses dynamic local memory via an explicit __local kernel argument.

static inline int wrap_coord(int v, int maxv) {
    int r = v % maxv;
    return (r < 0) ? (r + maxv) : r;
}

// Read cell with either wrap-around or fixed-zero boundary.
static inline uchar read_cell(__global const uchar* grid, int x, int y, int rows, int cols, int wrap) {
    if (wrap) {
        int xx = wrap_coord(x, rows);
        int yy = wrap_coord(y, cols);
        return grid[xx * cols + yy];
    }
    if (x < 0 || x >= rows || y < 0 || y >= cols) return (uchar)0;
    return grid[x * cols + y];
}

__kernel void gol_step_tiled(__global const uchar* grid,
                            __global uchar* next,
                            const int rows,
                            const int cols,
                            const int wrap,
                            __local uchar* tile)
{
    const int gx = (int)get_global_id(0); // row
    const int gy = (int)get_global_id(1); // col

    const int lx = (int)get_local_id(0);
    const int ly = (int)get_local_id(1);

    const int LX = (int)get_local_size(0);
    const int LY = (int)get_local_size(1);

    // tile has 1-cell halo on each side
    const int pitch = (LY + 2);

    // center
    tile[(lx + 1) * pitch + (ly + 1)] = read_cell(grid, gx, gy, rows, cols, wrap);

    // left/right halos
    if (ly == 0)
        tile[(lx + 1) * pitch + 0] = read_cell(grid, gx, gy - 1, rows, cols, wrap);
    if (ly == (LY - 1))
        tile[(lx + 1) * pitch + (LY + 1)] = read_cell(grid, gx, gy + 1, rows, cols, wrap);

    // top/bottom halos
    if (lx == 0)
        tile[0 * pitch + (ly + 1)] = read_cell(grid, gx - 1, gy, rows, cols, wrap);
    if (lx == (LX - 1))
        tile[(LX + 1) * pitch + (ly + 1)] = read_cell(grid, gx + 1, gy, rows, cols, wrap);

    // corners
    if (lx == 0 && ly == 0)
        tile[0 * pitch + 0] = read_cell(grid, gx - 1, gy - 1, rows, cols, wrap);
    if (lx == 0 && ly == (LY - 1))
        tile[0 * pitch + (LY + 1)] = read_cell(grid, gx - 1, gy + 1, rows, cols, wrap);
    if (lx == (LX - 1) && ly == 0)
        tile[(LX + 1) * pitch + 0] = read_cell(grid, gx + 1, gy - 1, rows, cols, wrap);
    if (lx == (LX - 1) && ly == (LY - 1))
        tile[(LX + 1) * pitch + (LY + 1)] = read_cell(grid, gx + 1, gy + 1, rows, cols, wrap);

    barrier(CLK_LOCAL_MEM_FENCE);

    if (gx >= rows || gy >= cols) return;

    int sum = 0;
    sum += tile[(lx + 0) * pitch + (ly + 0)];
    sum += tile[(lx + 0) * pitch + (ly + 1)];
    sum += tile[(lx + 0) * pitch + (ly + 2)];
    sum += tile[(lx + 1) * pitch + (ly + 0)];
    sum += tile[(lx + 1) * pitch + (ly + 2)];
    sum += tile[(lx + 2) * pitch + (ly + 0)];
    sum += tile[(lx + 2) * pitch + (ly + 1)];
    sum += tile[(lx + 2) * pitch + (ly + 2)];

    const uchar alive = tile[(lx + 1) * pitch + (ly + 1)];
    uchar out;
    if (alive) out = (sum == 2 || sum == 3) ? (uchar)1 : (uchar)0;
    else       out = (sum == 3) ? (uchar)1 : (uchar)0;

    next[gx * cols + gy] = out;
}
