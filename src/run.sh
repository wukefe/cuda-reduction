#!/bin/bash
echo "nvcc compile & run"
flag="-Xptxas -O3,-v"
(set -x && nvcc main.cu -o main ${flag} && ./main 4194304 $1)
#(set -x && nvcc main.cu -o main ${flag} && ./main 10 $1)

###### Notes ######
# sizes
#   - 4194304  ( 4*2^20, 4MB)
#   - 33554432 (32*2^20,32MB)
#
# Maximum optimization for CUDA programs
#   - Discussions: https://stackoverflow.com/a/43845984/4111149
#
# Generate ptx code
#   - `nvcc -ptx -o kernel.ptx -X^Cxas -O3,-v main.cu 


