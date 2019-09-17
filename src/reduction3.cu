__global__ void cuda_reduction_3(I *z, I *x){
    extern __shared__ I sharedData[];
    UI tid = threadIdx.x;
    UI i = blockIdx.x * blockDim.x + threadIdx.x;
    sharedData[tid] = x[i];
    __syncthreads(); // load to shared memory
    //printf("tid.x = %d, i = %d, dim = %d\n", threadIdx.x,(I)i,blockDim.x);
    // reduction
    for(UI k=blockDim.x>>1; k>0; k>>=1){
        if(tid < k){
            sharedData[tid] += sharedData[tid + k];
        }
        __syncthreads(); 
    }
    // write result to global mem
    if(tid == 0) z[blockIdx.x] = sharedData[0];
}

static I run_gpu_v3(I *x, L n){
    I *dx1, *dz1, *z1;
    I *dx2, *dz2, *z2;
    I numThread = NUM_THREAD;
    I numBlock  = n/numThread/2;
    I memSize   = numThread * sizeof(I);
    I half1     = n / 2;
    I half2     = n - half1;
    NEW(z1, I, numBlock);
    NEW(z2, I, numBlock);
    CUDA_NEW(dx1, I, half1);
    CUDA_NEW(dx2, I, half2);
    CUDA_NEW(dz1, I, numBlock);
    CUDA_NEW(dz2, I, numBlock);
    // copy
    CUDA_COPY(dx1, x      , I, half1);
    CUDA_COPY(dx2, x+half1, I, half2);
    //CUDA_COPY(dz, x, F, numBlock);
    //DOI(15, P("x[%lld] = %f\n", i,x[i]))
    struct timeval t1, t2;
    P("# of Block: %d\n# of Thread: %d\n", numBlock, numThread);
    gettimeofday(&t1, 0);
    // Dynamic Shared Memory
    //   https://devblogs.nvidia.com/using-shared-memory-cuda-cc/
    cuda_reduction_3<<<numBlock, numThread, memSize>>>(dz1, dx1);
    cudaDeviceSynchronize(); /* Wait for compute device to finish */
    CUDA_SAVE(z1, dz1, F, numBlock);
    cuda_reduction_3<<<numBlock, numThread, memSize>>>(dz2, dx2);
    CUDA_SAVE(z2, dz2, F, numBlock);
    I rtn = calcSum(z1, numBlock) + calcSum(z2, numBlock);
    gettimeofday(&t2, 0);
    P("[GPU] The elapsed average time (ms): %g (v3)\n", calcTime(t1,t2));
    //DOI(10, P("z[%lld] = %d\n", i,z[i]))
    CUDA_FREE(dx1); CUDA_FREE(dx2);
    CUDA_FREE(dz1); CUDA_FREE(dz2);
    R rtn;
}


