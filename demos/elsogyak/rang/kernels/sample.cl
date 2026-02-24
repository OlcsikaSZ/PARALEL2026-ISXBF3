// sample.cl
__kernel void compute_rank(__global int* arr, __global int* rank, const unsigned int n)
{
    int i = get_global_id(0);
    if (i >= n) return;   // ne lépjünk túl a tömbön
    int r = 0;
    for (int j=0; j<n; j++)
        if(arr[j]<arr[i]) r++;
    rank[i] = r;
}