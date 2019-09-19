## Evaluation of Parallel Reduction in CUDA

### Overview

Given a one-dimension integer array (i.e. vector), implement a reduction operation on it in CUDA.

### Implementation

I have implemented one serial version in C code (CPU) and a couple of versions in CUDA code (GPU).
Their source code can be found in the folder [./src](./src).

Based on different strategies, the overview of these versions is shown as follows.

| Version         | Is Parallel | Step Complexity | Require Input 2^n | Description                           |
| :-------------- | :---------: | :-------------: | :---------------: | :------------------------------------ | 
| [cpu][cpu0]     |     No      |  O(n)           | No                | Serial code for validation            |
| [cuda 0][cuda0] |     No      |  O(n)           | No                | Single thread `<<<1, 1>>>`            |
| [cuda 1][cuda1] |     Yes     |  O(log(n))      | Yes               | Reduction on 2^n, left to right       |
| [cuda 2][cuda2] |     Yes     |  O(log(n))      | Yes               | Reduction on 2^n, right to left       |
| [cuda 3][cuda3] |     Yes     |  O(log(n))      | Yes               | Similar to cuda 2, but with two loads |
| [cuda 4][cuda4] |     Yes     |  O(log(n))      | No                | A general version of cuda 2           |
| [cuda 5][cuda5] |     Yes     |  O(log(n))      | No                | An improved version of cuda 4         |
| [cuda 6][cuda6] |     Yes     |  O(log(n))      | No                | A further improved version of cuda 5  |

[cpu0]: ./src/main.cu
[cuda0]: ./src/reduction0.cu
[cuda1]: ./src/reduction1.cu
[cuda2]: ./src/reduction2.cu
[cuda3]: ./src/reduction3.cu
[cuda4]: ./src/reduction4.cu
[cuda5]: ./src/reduction5.cu
[cuda6]: ./src/reduction6.cu

**Note** that `cuda 1` and `cuda 2` are from
[the slides (Optimizing Parallel Redduction in CUDA, by Mark Harris)](doc/cuda-reduction.pdf)
found online from the NVIDIA company.  Other CUDA versions are inspired by the slides.

#### cuda 0

- Serial code with single GPU thread

```cuda
__global__ void cuda_reduction_0(I *z, I *x, L n){
    I s = 0; DOI(n, s+=x[i]) z[0]=s;
}
...
cuda_reduction_0<<<1, 1>>>(dz,dx,n);
```

#### cuda 1

- Interleaved addressing (page 8 on [Mark Harris's slides](doc/cuda-reduction.pdf))

```cuda
__global__ void cuda_reduction_1(I *z, I *x){
    ...
    for(UI k=1; k<blockDim.x; k*=2){
        if(tid % (2*k) == 0){
            sharedData[tid] += sharedData[tid + k];
        }
        __syncthreads(); // need sync for one stride
    }
    // write result to global mem
    if(tid == 0) z[blockIdx.x] = sharedData[0];
}
...
cuda_reduction_1<<<numBlock, numThread, memSize>>>(dz, dx);
```

#### cuda 2

- Sequential addressing (page 14 on [Mark Harris's slides](doc/cuda-reduction.pdf))

```cuda
__global__ void cuda_reduction_2(I *z, I *x){
    ...
    for(UI k=blockDim.x>>1; k>0; k>>=1){
        if(tid < k){
            sharedData[tid] += sharedData[tid + k];
        }
        __syncthreads();
    }
    // write result to global mem
    if(tid == 0) z[blockIdx.x] = sharedData[0];
}
...
cuda_reduction_2<<<numBlock, numThread, memSize>>>(dz, dx);
```

#### cuda 3

- Reuse the kernel function `cuda_reduction_2`
- Split input data into two parts and load them in order

```cuda
...
cuda_reduction_3<<<numBlock, numThread, memSize>>>(dz1, dx1);
cudaDeviceSynchronize();
...
cuda_reduction_3<<<numBlock, numThread, memSize>>>(dz2, dx2);
cudaDeviceSynchronize();
...
```

#### cuda 4

- Consider uneven cases

```cuda
__global__ void cuda_reduction_4(I *z, I *x, I bound){
    ...
    sharedData[tid] = (i<bound)?x[i]:0;
    __syncthreads(); // load to shared memory
    ...
}
...
I numBlock  = (n/numThread) + (n%numThread!=0);
...
```

#### cuda 5

- Unrolling to reduce the number of loops so that the number of synchronization is reduced

```cuda
__global__ void cuda_reduction_5(I *z, I *x, I bound){
    ...
    for(UI k=blockDim.x>>2; k>0; k>>=2){ // require 4^x = blockDim.x
        if(tid < k){
            sharedData[tid] += sharedData[tid + k] + sharedData[tid + (k<<1)] + sharedData[tid + (k<<1) + k];
        }
        __syncthreads();
    }
    ...
}
```

#### cuda 6

- Inspired by Brent’s theorem: let one thread do more things!

```cuda
__global__ void cuda_reduction_6(I *z, I *x, I total, I bound){
    ...
    sharedData[tid] = 0;
    while(i < total) { sharedData[tid] += (i<bound)?(x[i]+x[i+blockSize]):0; i+=gridSize; }
    ...
}
```

### Evaluation

Setup

- CPU
    - i7-8700K @ 3.70GHz (GFLOPS 320GF)
    - 12 (1x12) threads
    - 32 GB RAM
- GPU (GeForce GTX 1080Ti)
    - 3584 cores
    - Driver Version: 430.14
- OS
    - Ubuntu 16.04.6 LTS

Configurations

- Input: 4M integers (4194304 = 2^22)
- Threads per block: 256

Run versions

    ./run.sh <id>   # id can be: 0/1/2/3/4/5/6

Results:

| Versions | Time (ms)  |
| :------: | :--------: |
| cuda 0   | 541.723    |
| cuda 1   | 3.978      |
| cuda 2   | 2.037      |
| cuda 3   | 2.08       |
| cuda 4   | 2.0595     |
| cuda 5   | 1.5145     |
| cuda 6   | 0.969      |

- See the log file: [click](./src/log1.txt)

Discussions

- Single GPU thread is very slow (`cuda 0`)
- Loop unrolling (`cuda 5`) and one-thread-do-more-things (`cuda 6`) are effective techniques
- Feeding data to GPU in chunks (in `cuda 3`) seems not a good idea


