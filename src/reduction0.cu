__global__ void cuda_reduction_0(I *z, I *x, L n){
    I s = 0; DOI(n, s+=x[i]) z[0]=s;
}

static I run_gpu_v0(I *x, L n){
    I *dx, *dz; I z[0];
    CUDA_NEW(dx, I, n);
    CUDA_NEW(dz, I, 1);
    // copy
    CUDA_COPY(dx, x, I, n);
    //DOI(15, P("x[%lld] = %f\n", i,x[i]))
    struct timeval t1, t2;
    gettimeofday(&t1, 0);
    cuda_reduction_0<<<1, 1>>>(dz,dx,n);
    cudaDeviceSynchronize(); /* Wait for compute device to finish */
    gettimeofday(&t2, 0);
    CUDA_SAVE(&z, dz, I, 1);
    P("[GPU] The elapsed time (ms): %g (v0)\n", calcTime(t1,t2));
    CUDA_FREE(dx);
    CUDA_FREE(dz);
    R z[0];
}


