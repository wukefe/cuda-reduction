## Evaluation of Parallel Reduction in CUDA

### Overview

Given a one-dimension vector which has **N** integers, implement a reduction
operation in CUDA.

### Implementation

I have implemented one serial version in C code (CPU) and a couple of versions in CUDA code (GPU).
Their source code can be found in the folder [./src](./src).

Based on different strategies, the overview of these versions is shown as follows.

| Version    | Is Parallel | Step Complexity | Require input 2^n | Description                           |
| :--------- | :---------: | :-------------: | :---------------: | :------------------------------------ | 
| cpu        |     No      |  O(n)           | No                | Serial code for validation            |
| cuda 0     |     No      |  O(n)           | No                | Single thread `<<<1, 1>>>`            |
| cuda 1     |     Yes     |  O(log(n))      | Yes               | Reduction on 2^n, left to right       |
| cuda 2     |     Yes     |  O(log(n))      | Yes               | Reduction on 2^n, right to left       |
| cuda 3     |     Yes     |  O(log(n))      | Yes               | Similar to cuda 2, but with two loads |
| cuda 4     |     Yes     |  O(log(n))      | No                | A general version of cuda 2           |
| cuda 5     |     Yes     |  O(log(n))      | No                | An improved version of cuda 4         |
| cuda 6     |     Yes     |  O(log(n))      | No                | A further improved version of cuda 5  |

**Note** that `cuda 1` and `cuda 2` are from [the slides](doc/cuda-reduction.pdf) found online from the NVIDIA company.
Other CUDA versions are inspired by the slides.


### Evaluation

### Discussion


