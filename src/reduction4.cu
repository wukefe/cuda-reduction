__global__ void cuda_reduction_4(I *z, I *x, I bound){
    extern __shared__ I sharedData[];
    UI tid = threadIdx.x;
    UI i = blockIdx.x * blockDim.x + threadIdx.x;
    sharedData[tid] = (i<bound)?x[i]:0;
    __syncthreads(); // load to shared memory
    //printf("tid.x = %d, i = %d, dim = %d\n", threadIdx.x,(I)i,blockDim.x);
    // reduction
    for(UI k=blockDim.x>>1; k>0; k>>=1){
        if(tid < k){
            sharedData[tid] += sharedData[tid + k];
        }
        __syncthreads(); 
    }
    //P("blockIdx = %d, tid = %d, i = %d, sharedData[%d] = %d\n", \
            blockIdx.x, tid, (I)i, tid, sharedData[tid]);
    // write result to global mem
    if(tid == 0) z[blockIdx.x] = sharedData[0];
}

static I run_gpu_v4(I *x, L n){
    I *dx, *dz, *z;
    I numThread = NUM_THREAD;
    I numBlock  = (n/numThread) + (n%numThread!=0);
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
    cuda_reduction_4<<<numBlock, numThread, memSize>>>(dz, dx, n);
    cudaDeviceSynchronize(); /* Wait for compute device to finish */
    CUDA_SAVE(z, dz, F, numBlock);
    I rtn = calcSum(z, numBlock);
    gettimeofday(&t2, 0);
    P("[GPU] The elapsed average time (ms): %g (v4)\n", calcTime(t1,t2));
    DOI(5, P("z[%lld] = %d\n", i,z[i]))
    CUDA_FREE(dx);
    CUDA_FREE(dz);
    R rtn;
}


