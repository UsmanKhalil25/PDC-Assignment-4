# Assignment 4: NanoGPT149 Performance Analysis Report

## Students

- Muhammad Usman Khalil 22L-6873
- Ali Shafi 22L-6746

## Part 1: Naive Unfused Attention Analysis

### Performance Metrics

- Reference Implementation:

  - CPU Time: 270.906ms
  - Memory Usage: 4,718,592 bytes
  - Self CPU Time Total: 271.164ms

- Student Implementation:
  - CPU Time: 248.765ms
  - Memory Usage: 4,718,592 bytes
  - Self CPU Time Total: 248.877ms

### Raw output

```
(.venv) usman@DESKTOP-KCKDHHH:~/Code/PDC-Assignment-4$ python3 gpt149.py part1

Compiling code into a PyTorch module...


Running Part 1 Test: Naive Unfused Attention

-----RUNNING REFERENCE IMPLEMENTATION-----

WARNING:2025-05-11 17:23:45 39046:39046 init.cpp:155] function cbapi->getCuptiStatus() failed with error CUPTI_ERROR_NOT_INITIALIZED (15)
WARNING:2025-05-11 17:23:45 39046:39046 init.cpp:156] CUPTI initialization failed - CUDA profiler activities will be missing
INFO:2025-05-11 17:23:45 39046:39046 init.cpp:158] If you see CUPTI_ERROR_INSUFFICIENT_PRIVILEGES, refer to https://developer.nvidia.com/nvidia-development-tools-solutions-err-nvgpuctrperm-cupti
STAGE:2025-05-11 17:23:45 39046:39046 ActivityProfilerController.cpp:312] Completed Stage: Warm Up
STAGE:2025-05-11 17:23:46 39046:39046 ActivityProfilerController.cpp:318] Completed Stage: Collection
STAGE:2025-05-11 17:23:46 39046:39046 ActivityProfilerController.cpp:322] Completed Stage: Post Processing
manual attention == pytorch attention True
Manual Execution Time:  0.2709994316101074

-------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                           Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls
-------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                    aten::empty         0.04%     100.000us         0.04%     100.000us      33.333us       5.00 Mb       5.00 Mb             3
    REFERENCE - NAIVE ATTENTION        95.95%     260.174ms        99.90%     270.906ms     270.906ms       4.50 Mb      -1.00 Mb             1
                    aten::zeros         0.79%       2.141ms         2.63%       7.125ms       3.562ms       4.50 Mb           0 b             2
                    aten::clone         0.47%       1.288ms         0.89%       2.418ms       1.209ms       1.00 Mb           0 b             2
                model_inference         0.10%     258.000us       100.00%     271.164ms     271.164ms     512.00 Kb      -4.00 Mb             1
                  aten::flatten         0.42%       1.150ms         1.15%       3.119ms     623.800us     512.00 Kb           0 b             5
               aten::empty_like         0.01%      23.000us         0.02%      54.000us      54.000us     512.00 Kb           0 b             1
            aten::empty_strided         0.01%      29.000us         0.01%      29.000us      29.000us     512.00 Kb     512.00 Kb             1
                    aten::zero_         0.03%      77.000us         1.81%       4.915ms       2.458ms           0 b           0 b             2
                    aten::fill_         1.78%       4.838ms         1.78%       4.838ms       2.419ms           0 b           0 b             2
-------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 271.164ms

REFERENCE - NAIVE ATTENTION statistics
cpu time:  270.906ms
mem usage:  4718592 bytes
-----RUNNING STUDENT IMPLEMENTATION-----

STAGE:2025-05-11 17:23:54 39046:39046 ActivityProfilerController.cpp:312] Completed Stage: Warm Up
STAGE:2025-05-11 17:23:54 39046:39046 ActivityProfilerController.cpp:318] Completed Stage: Collection
STAGE:2025-05-11 17:23:54 39046:39046 ActivityProfilerController.cpp:322] Completed Stage: Post Processing
manual attention == pytorch attention True
Manual Execution Time:  0.24881362915039062

-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                         Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                  aten::empty         0.02%      51.000us         0.02%      51.000us      17.000us       5.00 Mb       5.00 Mb             3
    STUDENT - NAIVE ATTENTION        98.89%     246.122ms        99.95%     248.765ms     248.765ms       4.50 Mb      -1.00 Mb             1
                  aten::zeros         0.02%      51.000us         0.58%       1.434ms     717.000us       4.50 Mb           0 b             2
                  aten::clone         0.02%      40.000us         0.44%       1.094ms     547.000us       1.00 Mb           0 b             2
              model_inference         0.05%     112.000us       100.00%     248.877ms     248.877ms     512.00 Kb      -4.00 Mb             1
                aten::flatten         0.03%      75.000us         0.29%     727.000us     145.400us     512.00 Kb           0 b             5
             aten::empty_like         0.00%       9.000us         0.01%      27.000us      27.000us     512.00 Kb           0 b             1
          aten::empty_strided         0.00%      11.000us         0.00%      11.000us      11.000us     512.00 Kb     512.00 Kb             1
                  aten::zero_         0.01%      27.000us         0.54%       1.350ms     675.000us           0 b           0 b             2
                  aten::fill_         0.53%       1.323ms         0.53%       1.323ms     661.500us           0 b           0 b             2
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 248.877ms

STUDENT - NAIVE ATTENTION statistics
cpu time:  248.765ms
mem usage:  4718592 bytes
```

### Analysis

The naive implementation shows comparable performance between reference and student implementations, with the student version actually performing slightly better (about 8% faster). The memory usage remains identical at ~4.7MB, which is expected for the naive approach as it requires storing the full NxN attention matrix.

## Part 2: Blocked Matrix Multiply and Unfused Softmax Analysis

### Performance Metrics

- Reference Implementation:

  - CPU Time: 309.047ms
  - Memory Usage: 4,718,592 bytes
  - Self CPU Time Total: 309.193ms

- Student Implementation:
  - CPU Time: 210.923ms
  - Memory Usage: 4,718,592 bytes
  - Self CPU Time Total: 211.016ms

### Raw output

```
(.venv) usman@DESKTOP-KCKDHHH:~/Code/PDC-Assignment-4$ python3 gpt149.py part2


Compiling code into a PyTorch module...


Running Part 2 Test: Unfused Attention with Blocked Matmul

-----RUNNING REFERENCE IMPLEMENTATION-----

WARNING:2025-05-11 17:28:29 39149:39149 init.cpp:155] function cbapi->getCuptiStatus() failed with error CUPTI_ERROR_NOT_INITIALIZED (15)
WARNING:2025-05-11 17:28:29 39149:39149 init.cpp:156] CUPTI initialization failed - CUDA profiler activities will be missing
INFO:2025-05-11 17:28:29 39149:39149 init.cpp:158] If you see CUPTI_ERROR_INSUFFICIENT_PRIVILEGES, refer to https://developer.nvidia.com/nvidia-development-tools-solutions-err-nvgpuctrperm-cupti
STAGE:2025-05-11 17:28:29 39149:39149 ActivityProfilerController.cpp:312] Completed Stage: Warm Up
STAGE:2025-05-11 17:28:29 39149:39149 ActivityProfilerController.cpp:318] Completed Stage: Collection
STAGE:2025-05-11 17:28:29 39149:39149 ActivityProfilerController.cpp:322] Completed Stage: Post Processing
manual attention == pytorch attention True
Manual Execution Time:  0.30911779403686523

------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                            Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls
------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                     aten::empty         0.03%     107.000us         0.03%     107.000us      35.667us       5.00 Mb       5.00 Mb             3
    REFERENCE - BLOCKED MATMUL + UNFUSED SOFTMAX        97.62%     301.826ms        99.95%     309.047ms     309.047ms       4.50 Mb      -1.00 Mb             1
                                     aten::zeros         0.36%       1.113ms         1.86%       5.742ms       2.871ms       4.50 Mb           0 b             2
                                     aten::clone         0.02%      59.000us         0.44%       1.349ms     674.500us       1.00 Mb           0 b             2
                                 model_inference         0.05%     146.000us       100.00%     309.193ms     309.193ms     512.00 Kb      -4.00 Mb             1
                                   aten::flatten         0.03%      95.000us         0.32%     975.000us     195.000us     512.00 Kb           0 b             5
                                aten::empty_like         0.00%      12.000us         0.01%      29.000us      29.000us     512.00 Kb           0 b             1
                             aten::empty_strided         0.01%      32.000us         0.01%      32.000us      32.000us     512.00 Kb     512.00 Kb             1
                                     aten::zero_         0.02%      60.000us         1.47%       4.539ms       2.269ms           0 b           0 b             2
                                     aten::fill_         1.45%       4.479ms         1.45%       4.479ms       2.240ms           0 b           0 b             2
------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 309.193ms

REFERENCE - BLOCKED MATMUL + UNFUSED SOFTMAX statistics
cpu time:  309.047ms
mem usage:  4718592 bytes
-----RUNNING STUDENT IMPLEMENTATION-----

STAGE:2025-05-11 17:28:37 39149:39149 ActivityProfilerController.cpp:312] Completed Stage: Warm Up
STAGE:2025-05-11 17:28:37 39149:39149 ActivityProfilerController.cpp:318] Completed Stage: Collection
STAGE:2025-05-11 17:28:37 39149:39149 ActivityProfilerController.cpp:322] Completed Stage: Post Processing
manual attention == pytorch attention True
Manual Execution Time:  0.21096324920654297

----------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                          Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls
----------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                   aten::empty         0.04%      82.000us         0.04%      82.000us      27.333us       5.00 Mb       5.00 Mb             3
    STUDENT - BLOCKED MATMUL + UNFUSED SOFTMAX        98.12%     207.048ms        99.96%     210.923ms     210.923ms       4.50 Mb      -1.00 Mb             1
                                   aten::zeros         0.02%      42.000us         1.20%       2.541ms       1.270ms       4.50 Mb           0 b             2
                                   aten::clone         0.03%      61.000us         0.54%       1.142ms     571.000us       1.00 Mb           0 b             2
                               model_inference         0.04%      93.000us       100.00%     211.016ms     211.016ms     512.00 Kb      -4.00 Mb             1
                                 aten::flatten         0.05%     107.000us         0.42%     886.000us     177.200us     512.00 Kb           0 b             5
                              aten::empty_like         0.00%       9.000us         0.02%      39.000us      39.000us     512.00 Kb           0 b             1
                           aten::empty_strided         0.01%      12.000us         0.01%      12.000us      12.000us     512.00 Kb     512.00 Kb             1
                                   aten::zero_         0.01%      24.000us         1.16%       2.447ms       1.224ms           0 b           0 b             2
                                   aten::fill_         1.15%       2.423ms         1.15%       2.423ms       1.212ms           0 b           0 b             2
----------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 211.016ms

STUDENT - BLOCKED MATMUL + UNFUSED SOFTMAX statistics
cpu time:  210.923ms
mem usage:  4718592 bytes
```

### Analysis

The blocked implementation shows significant improvement in the student version, achieving approximately 32% better performance than the reference implementation. This demonstrates the effectiveness of cache-aware blocking strategies. The memory usage remains the same as Part 1 since we're still maintaining the full NxN matrix.

## Part 3: Fused Attention Analysis

### Performance Metrics

- Reference Implementation:

  - CPU Time: 248.826ms
  - Memory Usage: 557,056 bytes
  - Self CPU Time Total: 248.939ms

- Student Implementation:
  - CPU Time: 129.171ms
  - Memory Usage: 557,056 bytes
  - Self CPU Time Total: 129.234ms

### Raw output

```
(.venv) usman@DESKTOP-KCKDHHH:~/Code/PDC-Assignment-4$ python3 gpt149.py part3

Compiling code into a PyTorch module...


Running Part 3 Test: Fused Attention

-----RUNNING REFERENCE IMPLEMENTATION-----

WARNING:2025-05-11 17:31:08 39233:39233 init.cpp:155] function cbapi->getCuptiStatus() failed with error CUPTI_ERROR_NOT_INITIALIZED (15)
WARNING:2025-05-11 17:31:08 39233:39233 init.cpp:156] CUPTI initialization failed - CUDA profiler activities will be missing
INFO:2025-05-11 17:31:08 39233:39233 init.cpp:158] If you see CUPTI_ERROR_INSUFFICIENT_PRIVILEGES, refer to https://developer.nvidia.com/nvidia-development-tools-solutions-err-nvgpuctrperm-cupti
STAGE:2025-05-11 17:31:09 39233:39233 ActivityProfilerController.cpp:312] Completed Stage: Warm Up
STAGE:2025-05-11 17:31:09 39233:39233 ActivityProfilerController.cpp:318] Completed Stage: Collection
STAGE:2025-05-11 17:31:09 39233:39233 ActivityProfilerController.cpp:322] Completed Stage: Post Processing
manual attention == pytorch attention True
Manual Execution Time:  0.24887990951538086

-------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                           Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls
-------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                    aten::empty         0.03%      68.000us         0.03%      68.000us      22.667us       1.03 Mb       1.03 Mb             3
                    aten::clone         0.03%      69.000us         0.40%     994.000us     497.000us       1.00 Mb           0 b             2
    REFERENCE - FUSED ATTENTION        90.08%     224.243ms        99.95%     248.826ms     248.826ms     544.00 Kb      -1.00 Mb             1
                    aten::zeros         0.03%      67.000us         0.24%     602.000us     301.000us     544.00 Kb           0 b             2
                model_inference         0.05%     113.000us       100.00%     248.939ms     248.939ms     512.00 Kb     -32.00 Kb             1
                  aten::flatten         2.45%       6.091ms         3.29%       8.197ms      15.886us     512.00 Kb           0 b           516
               aten::empty_like         0.01%      15.000us         0.02%      38.000us      38.000us     512.00 Kb           0 b             1
            aten::empty_strided         0.01%      31.000us         0.01%      31.000us      31.000us     512.00 Kb     512.00 Kb             1
                    aten::zero_         0.02%      50.000us         0.20%     490.000us     245.000us           0 b           0 b             2
                    aten::fill_         0.18%     440.000us         0.18%     440.000us     440.000us           0 b           0 b             1
-------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 248.939ms

REFERENCE - FUSED ATTENTION statistics
cpu time:  248.826ms
mem usage:  557056 bytes
-----RUNNING STUDENT IMPLEMENTATION-----

STAGE:2025-05-11 17:31:17 39233:39233 ActivityProfilerController.cpp:312] Completed Stage: Warm Up
STAGE:2025-05-11 17:31:17 39233:39233 ActivityProfilerController.cpp:318] Completed Stage: Collection
STAGE:2025-05-11 17:31:17 39233:39233 ActivityProfilerController.cpp:322] Completed Stage: Post Processing
manual attention == pytorch attention True
Manual Execution Time:  0.12920331954956055

-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                         Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                  aten::empty         0.03%      35.000us         0.03%      35.000us       8.750us       1.04 Mb       1.04 Mb             4
                  aten::clone         0.03%      42.000us         0.76%     985.000us     492.500us       1.00 Mb           0 b             2
                  aten::zeros         0.03%      34.000us         0.42%     546.000us     182.000us     548.00 Kb           0 b             3
    STUDENT - FUSED ATTENTION        94.66%     122.334ms        99.95%     129.171ms     129.171ms     544.00 Kb      -1.00 Mb             1
              model_inference         0.05%      63.000us       100.00%     129.234ms     129.234ms     512.00 Kb     -32.00 Kb             1
                aten::flatten         0.97%       1.260ms         2.15%       2.782ms       5.381us     512.00 Kb           0 b           517
             aten::empty_like         0.01%       8.000us         0.01%      17.000us      17.000us     512.00 Kb           0 b             1
          aten::empty_strided         0.01%      12.000us         0.01%      12.000us      12.000us     512.00 Kb     512.00 Kb             1
                  aten::zero_         0.02%      20.000us         0.38%     486.000us     162.000us           0 b           0 b             3
                  aten::fill_         0.36%     466.000us         0.36%     466.000us     466.000us           0 b           0 b             1
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 129.234ms

STUDENT - FUSED ATTENTION statistics
cpu time:  129.171ms
mem usage:  557056 bytes
```

### Analysis

The fused implementation shows dramatic improvements in both performance and memory usage:

1. Memory usage reduced by ~88% (from 4.7MB to 557KB)
2. Student implementation is ~48% faster than the reference
3. Overall performance is significantly better than both previous implementations

This demonstrates the effectiveness of operation fusion in reducing memory overhead and improving cache utilization.

## Part 4: Flash Attention Analysis

### Performance Metrics

- Reference Implementation:

  - CPU Time: 601.97ms
  - Memory Usage: 524,288 bytes
  - Self CPU Time Total: 603.591ms

- Student Implementation:
  - CPU Time: 221.111ms
  - Memory Usage: 524,288 bytes
  - Self CPU Time Total: 222.672ms

### Raw output

```
(.venv) usman@DESKTOP-KCKDHHH:~/Code/PDC-Assignment-4$ python3 gpt149.py part4

Compiling code into a PyTorch module...


Running Part 4 Test: Flash Attention

-----RUNNING REFERENCE IMPLEMENTATION-----

WARNING:2025-05-11 17:33:55 39310:39310 init.cpp:155] function cbapi->getCuptiStatus() failed with error CUPTI_ERROR_NOT_INITIALIZED (15)
WARNING:2025-05-11 17:33:55 39310:39310 init.cpp:156] CUPTI initialization failed - CUDA profiler activities will be missing
INFO:2025-05-11 17:33:55 39310:39310 init.cpp:158] If you see CUPTI_ERROR_INSUFFICIENT_PRIVILEGES, refer to https://developer.nvidia.com/nvidia-development-tools-solutions-err-nvgpuctrperm-cupti
STAGE:2025-05-11 17:33:55 39310:39310 ActivityProfilerController.cpp:312] Completed Stage: Warm Up
STAGE:2025-05-11 17:33:56 39310:39310 ActivityProfilerController.cpp:318] Completed Stage: Collection
STAGE:2025-05-11 17:33:56 39310:39310 ActivityProfilerController.cpp:322] Completed Stage: Post Processing
manual attention == pytorch attention True
Manual Execution Time:  0.6035318374633789

-------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                           Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls
-------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                    aten::zeros         0.02%     142.000us         0.97%       5.867ms     419.071us       9.16 Mb           0 b            14
                    aten::empty         0.03%     196.000us         0.03%     196.000us      14.000us       9.16 Mb       9.16 Mb            14
                model_inference         0.08%     453.000us       100.00%     603.591ms     603.591ms     512.00 Kb    -679.00 Kb             1
    REFERENCE - FLASH ATTENTION        90.98%     549.126ms        99.73%     601.970ms     601.970ms     512.00 Kb      -8.00 Mb             1
                    aten::zero_         0.34%       2.079ms         8.89%      53.674ms     145.065us           0 b           0 b           370
                    aten::fill_         8.55%      51.595ms         8.55%      51.595ms     387.932us           0 b           0 b           133
-------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 603.591ms

REFERENCE - FLASH ATTENTION statistics
cpu time:  601.97ms
mem usage:  524288 bytes
-----RUNNING STUDENT IMPLEMENTATION-----

STAGE:2025-05-11 17:34:03 39310:39310 ActivityProfilerController.cpp:312] Completed Stage: Warm Up
STAGE:2025-05-11 17:34:03 39310:39310 ActivityProfilerController.cpp:318] Completed Stage: Collection
STAGE:2025-05-11 17:34:03 39310:39310 ActivityProfilerController.cpp:322] Completed Stage: Post Processing
manual attention == pytorch attention True
Manual Execution Time:  0.22261500358581543

-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                         Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                  aten::empty         0.03%      56.000us         0.03%      56.000us       4.308us       1.66 Mb       1.66 Mb            13
                  aten::zeros         0.03%      58.000us         0.79%       1.768ms     147.333us       1.16 Mb           0 b            12
                  aten::clone         0.03%      68.000us         0.38%     851.000us     425.500us       1.00 Mb           0 b             2
              model_inference         0.12%     268.000us       100.00%     222.672ms     222.672ms     512.00 Kb    -679.00 Kb             1
    STUDENT - FLASH ATTENTION        98.60%     219.553ms        99.30%     221.111ms     221.111ms     512.00 Kb      -1.00 Mb             1
                aten::flatten         0.07%     155.000us         0.32%     719.000us      14.979us     512.00 Kb           0 b            48
             aten::empty_like         0.01%      15.000us         0.01%      29.000us      29.000us     512.00 Kb           0 b             1
          aten::empty_strided         0.00%      10.000us         0.00%      10.000us      10.000us     512.00 Kb     512.00 Kb             1
                  aten::zero_         0.02%      41.000us         0.75%       1.668ms     139.000us           0 b           0 b            12
                  aten::fill_         0.73%       1.627ms         0.73%       1.627ms     542.333us           0 b           0 b             3
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 222.672ms

STUDENT - FLASH ATTENTION statistics
cpu time:  221.111ms
mem usage:  524288 bytes
```

### Analysis

The Flash Attention implementation shows interesting characteristics:

1. Memory usage is further reduced to ~524KB
2. Reference implementation is slower than Part 3, but student implementation maintains good performance
3. Student implementation is ~63% faster than the reference
4. The implementation successfully handles the block-wise computation of attention

## Overall Performance Comparison

### CPU Time Progression

1. Naive (Part 1): 248.765ms
2. Blocked (Part 2): 210.923ms
3. Fused (Part 3): 129.171ms
4. Flash (Part 4): 221.111ms

### Memory Usage Progression

1. Naive (Part 1): 4,718,592 bytes
2. Blocked (Part 2): 4,718,592 bytes
3. Fused (Part 3): 557,056 bytes
4. Flash (Part 4): 524,288 bytes

### Key Insights

1. Memory optimization had the most significant impact on performance
2. Operation fusion provided the best overall performance
3. Flash Attention, while more complex, successfully reduced memory footprint
4. Student implementations consistently outperformed reference implementations
5. The progression shows clear improvements in both memory efficiency and computational performance

This enhanced report provides a clear analysis of the performance characteristics and improvements across all four implementations while maintaining the original raw output data.
