#include <stdio.h>
#include <sys/time.h>

typedef float        F;
typedef int          I;
typedef long long    L;
typedef unsigned int UI;

#define NUM_THREAD 128 //4

#define P printf
#define R return
#define printBanner(x) P("=== " x " ===\n")
#define DOI(n, x) for(L i=0,i2=n; i<i2; i++){x;}

#define NEW(x,t,n) x=(t*)malloc(sizeof(t)*(n))
#define FREE(x)    free(x)

#define CUDA_NEW(x,t,n)      cudaMalloc((void**)&x, sizeof(t) * (n))
#define CUDA_FREE(x)         cudaFree(x)
#define CUDA_COPY(d_x,x,t,n) cudaMemcpy(d_x, x, sizeof(t)*(n), cudaMemcpyHostToDevice)
#define CUDA_SAVE(x,d_x,t,n) cudaMemcpy(x, d_x, sizeof(t)*(n), cudaMemcpyDeviceToHost)

static F calcTime(struct timeval t1, struct timeval t2){
    return (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
}

static I calcSum(I *x, I n){
    I s=0; DOI(n, s+=x[i]) R s;
}

static I* createVector(L n){
    I *x = NEW(x, I, n); DOI(n, x[i]=i); R x;
}

static I run_cpu_version(I *x, L n){
    struct timeval t1, t2;
    gettimeofday(&t1, 0);
    I s=0; DOI(n, s+=x[i]);
    gettimeofday(&t2, 0);
    P("[CPU] The elapsed time (ms): %g\n", calcTime(t1, t2));
    R s;
}

#include "reduction0.cu"
#include "reduction1.cu"
#include "reduction2.cu"
#include "reduction3.cu"
#include "reduction4.cu"
#include "reduction5.cu"
#include "reduction6.cu"

static I run_gpu_version(I *x, L n, I op){
    switch(op){
        case 0: R run_gpu_v0(x, n); /* single thread reduction    */
        case 1: R run_gpu_v1(x, n); /* efficient log(n) reduction */
        case 2: R run_gpu_v2(x, n); /* improved log(n) reduction  */
        case 3: R run_gpu_v3(x, n); /* two loads reduction        */
        case 4: R run_gpu_v4(x, n); /* general reduction          */
        case 5: R run_gpu_v5(x, n); /* improved general reduction */
        case 6: R run_gpu_v6(x, n); /*  */
    }
    R 0;
}

int main(int argc, char *argv[]) {
    if(argc != 3){
        fprintf(stderr, "Usage: %s <size> <op>\n", argv[0]);
        exit(1);
    }
    printBanner("Starting CUDA");
    L n  = strtoll(argv[1], NULL, 10);
    I op = atoi(argv[2]);
    I *x = createVector(n);
    // cpu: check result
    I result1 = run_cpu_version(x, n);
    I result2 = run_gpu_version(x, n, op);
    P("\nOutput (n = %lld):", n);
    P("\n\tResult [cpu]: %d\n\tResult [gpu]: %d\n", result1, result2);
    return 0;
}

/*
reference:
- https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
*/


