__kernel void quicksort_ranges(__global int* data,
                               __global const int2* ranges,
                               const int num_ranges) {
    int gid = get_global_id(0);
    if (gid >= num_ranges) return;

    int left = ranges[gid].x;
    int right = ranges[gid].y;

    if (left >= right) return;

    // Iterativ quicksort sajat veremmel
    // Max blokkmerethez elegendo stack kell.
    // 2048-es blokkhoz boven eleg egy 64-es stack.
    int lstack[64];
    int rstack[64];
    int top = 0;

    lstack[top] = left;
    rstack[top] = right;
    top++;

    while (top > 0) {
        top--;
        int l = lstack[top];
        int r = rstack[top];

        while (l < r) {
            int i = l;
            int j = r;
            int pivot = data[l + (r - l) / 2];

            while (i <= j) {
                while (data[i] < pivot) i++;
                while (data[j] > pivot) j--;

                if (i <= j) {
                    int tmp = data[i];
                    data[i] = data[j];
                    data[j] = tmp;
                    i++;
                    j--;
                }
            }

            // Kisebb reszt dolgozzuk fel azonnal,
            // nagyobbat stackre rakjuk -> kisebb stackigeny
            if ((j - l) < (r - i)) {
                if (i < r) {
                    lstack[top] = i;
                    rstack[top] = r;
                    top++;
                }
                r = j;
            } else {
                if (l < j) {
                    lstack[top] = l;
                    rstack[top] = j;
                    top++;
                }
                l = i;
            }
        }
    }
}