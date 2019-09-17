// blockDim.x --> NUM_THREADS
__global__ void cuda_reduction_1(I *z, I *x){
    extern __shared__ I sharedData[];
    UI tid = threadIdx.x;
    UI i = blockIdx.x * blockDim.x + threadIdx.x;
    sharedData[tid] = x[i];
    __syncthreads(); // load to shared memory
    //printf("tid.x = %d, i = %d, dim = %d\n", threadIdx.x,(I)i,blockDim.x);
    // reduction
    for(UI k=1; k<blockDim.x; k*=2){
        if(tid % (2*k) == 0){
            sharedData[tid] += sharedData[tid + k];
        }
        __syncthreads(); // need sync for one stride
    }
    // write result to global mem
    if(tid == 0) z[blockIdx.x] = sharedData[0];
}

static I run_gpu_v1(I *x, L n){
    I *dx, *dz, *z;
    I numThread = NUM_THREAD;
    I numBlock  = n/numThread;
    I memSize   = numThread * sizeof(I);
    NEW(z, I, numBlock);
    CUDA_NEW(dx, I, n);
    CUDA_NEW(dz, I, numBlock);
    // copy
    CUDA_COPY(dx, x, I, n);
    //CUDA_COPY(dz, x, F, numBlock);
    //DOI(15, P("x[%lld] = %f\n", i,x[i]))
    struct timeval t1, t2;
    P("# of Block: %d\n# of Thread: %d\n", numBlock, numThread);
    gettimeofday(&t1, 0);
    // Dynamic Shared Memory
    //   https://devblogs.nvidia.com/using-shared-memory-cuda-cc/
    cuda_reduction_1<<<numBlock, numThread, memSize>>>(dz, dx);
    cudaDeviceSynchronize(); /* Wait for compute device to finish */
    CUDA_SAVE(z, dz, F, numBlock);
    I rtn = calcSum(z, numBlock);
    gettimeofday(&t2, 0);
    P("[GPU] The elapsed average time (ms): %g (v1)\n", calcTime(t1,t2));
    DOI(10, P("z[%lld] = %d\n", i,z[i]))
    CUDA_FREE(dx);
    CUDA_FREE(dz);
    R rtn;
}

