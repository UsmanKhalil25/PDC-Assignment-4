### Question 1

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