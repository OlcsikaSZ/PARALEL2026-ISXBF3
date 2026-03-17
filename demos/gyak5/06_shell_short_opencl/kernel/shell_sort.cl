__kernel void shell_pass(__global int* data, const int n, const int gap) {
    int gid = get_global_id(0);

    if (gid >= gap) return;

    // A gid-edik gap-os részsorozat rendezése
    // Példa gap=4 esetén:
    // gid=0 -> 0,4,8,12,...
    // gid=1 -> 1,5,9,13,...
    // stb.
    for (int i = gid + gap; i < n; i += gap) {
        int temp = data[i];
        int j = i;

        while (j >= gap && data[j - gap] > temp) {
            data[j] = data[j - gap];
            j -= gap;
        }

        data[j] = temp;
    }
}