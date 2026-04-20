// Normalize coordinates into the valid range for toroidal wrap-around.
static inline int wrap_coord(int v, int maxv) {
    int r = v % maxv;
    return (r < 0) ? (r + maxv) : r;
}

// Read one cell using either wrap-around or zero-padded borders.
static inline uchar read_cell(__global const uchar* grid, int x, int y, int rows, int cols, int wrap) {
    if (wrap) {
        int xx = wrap_coord(x, rows);
        int yy = wrap_coord(y, cols);
        return grid[xx * cols + yy];
    }
    if (x < 0 || x >= rows || y < 0 || y >= cols) return (uchar)0;
    return grid[x * cols + y];
}

// Compute one generation using local-memory tiles with a 1-cell halo.
__kernel void gol_step_tiled(__global const uchar* grid,
                            __global uchar* next,
                            const int rows,
                            const int cols,
                            const int wrap,
                            __local uchar* tile)
{
    // Global IDs locate the output cell in the full board.
    const int gx = (int)get_global_id(0); // row
    const int gy = (int)get_global_id(1); // col

    // Local IDs locate the work-item inside the current tile.
    const int lx = (int)get_local_id(0);
    const int ly = (int)get_local_id(1);

    // Local sizes define the tile dimensions chosen by the host.
    const int LX = (int)get_local_size(0);
    const int LY = (int)get_local_size(1);

    // The halo makes the local tile two cells wider in both dimensions.
    const int pitch = (LY + 2);

    // Load the center cell handled by this work-item.
    tile[(lx + 1) * pitch + (ly + 1)] = read_cell(grid, gx, gy, rows, cols, wrap);

    // Border threads cooperatively load the left and right halo columns.
    if (ly == 0)
        tile[(lx + 1) * pitch + 0] = read_cell(grid, gx, gy - 1, rows, cols, wrap);
    if (ly == (LY - 1))
        tile[(lx + 1) * pitch + (LY + 1)] = read_cell(grid, gx, gy + 1, rows, cols, wrap);

    // Border threads cooperatively load the top and bottom halo rows.
    if (lx == 0)
        tile[0 * pitch + (ly + 1)] = read_cell(grid, gx - 1, gy, rows, cols, wrap);
    if (lx == (LX - 1))
        tile[(LX + 1) * pitch + (ly + 1)] = read_cell(grid, gx + 1, gy, rows, cols, wrap);

    // Corner threads load the four halo corner cells.
    if (lx == 0 && ly == 0)
        tile[0 * pitch + 0] = read_cell(grid, gx - 1, gy - 1, rows, cols, wrap);
    if (lx == 0 && ly == (LY - 1))
        tile[0 * pitch + (LY + 1)] = read_cell(grid, gx - 1, gy + 1, rows, cols, wrap);
    if (lx == (LX - 1) && ly == 0)
        tile[(LX + 1) * pitch + 0] = read_cell(grid, gx + 1, gy - 1, rows, cols, wrap);
    if (lx == (LX - 1) && ly == (LY - 1))
        tile[(LX + 1) * pitch + (LY + 1)] = read_cell(grid, gx + 1, gy + 1, rows, cols, wrap);

    // Wait until the full tile and halo are available in local memory.
    barrier(CLK_LOCAL_MEM_FENCE);

    // Ignore padded work-items that fall outside the real grid.
    if (gx >= rows || gy >= cols) return;

    int sum = 0;
    // Sum the eight neighbors from fast local memory instead of global memory.
    sum += tile[(lx + 0) * pitch + (ly + 0)];
    sum += tile[(lx + 0) * pitch + (ly + 1)];
    sum += tile[(lx + 0) * pitch + (ly + 2)];
    sum += tile[(lx + 1) * pitch + (ly + 0)];
    sum += tile[(lx + 1) * pitch + (ly + 2)];
    sum += tile[(lx + 2) * pitch + (ly + 0)];
    sum += tile[(lx + 2) * pitch + (ly + 1)];
    sum += tile[(lx + 2) * pitch + (ly + 2)];

    // Read the current cell state from the tile center.
    const uchar alive = tile[(lx + 1) * pitch + (ly + 1)];
    uchar out;
    if (alive) out = (sum == 2 || sum == 3) ? (uchar)1 : (uchar)0;
    else       out = (sum == 3) ? (uchar)1 : (uchar)0;

    // Write the computed next-generation value back to global memory.
    next[gx * cols + gy] = out;
}
