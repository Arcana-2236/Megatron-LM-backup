# CoLA Memory Liveness Report

## Run Matrix

See `/home/zhengyangwang/offloading/ATC-Megatron/benchmarks/0525/memory_liveness_20260526T040751Z/RUN_MATRIX.csv` for status, commands, logs, snapshots, and parsed metrics.

## Memory Liveness Table

| model impl | runtime mode | snapshot phase | active allocated GB | inactive split GB | reserved inactive GB | reserved GB | allocated GB | reserved-allocated gap GB | max reserved-max allocated gap GB | largest free block GB | active block count | inactive split block count |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CoLA | DistOpt | post_init | 8.991 | 0.143 | 1.923 | 10.914 | 8.991 | 1.923 | 0.554 | 0.020 | 73 | 0 |
| CoLA | DistOpt | post_backward | 9.071 | 0.265 | 5.677 | 14.748 | 9.071 | 5.677 | 0.308 | 0.240 | 83 | 0 |
| CoLA | DistOpt | post_backward | 11.644 | 0.495 | 5.838 | 17.482 | 11.644 | 5.838 | 0.463 | 0.240 | 217 | 0 |
| CoLA | DistOpt | post_backward | 11.644 | 0.495 | 6.030 | 17.674 | 11.644 | 6.030 | 0.654 | 0.240 | 217 | 0 |
| CoLA | DistOpt | post_backward | 11.644 | 0.495 | 6.030 | 17.674 | 11.644 | 6.030 | 0.654 | 0.240 | 217 | 0 |
| CoLA | DistOpt | post_backward | 11.644 | 0.495 | 6.030 | 17.674 | 11.644 | 6.030 | 0.654 | 0.240 | 217 | 0 |
| CoLA | DistOpt | post_backward | 11.644 | 0.495 | 6.030 | 17.674 | 11.644 | 6.030 | 0.654 | 0.240 | 217 | 0 |
| CoLA | DistOpt | post_backward | 11.644 | 0.495 | 6.030 | 17.674 | 11.644 | 6.030 | 0.654 | 0.240 | 217 | 0 |
| CoLA | DistOpt | post_backward | 11.644 | 0.495 | 6.030 | 17.674 | 11.644 | 6.030 | 0.654 | 0.240 | 217 | 0 |
| CoLA | DistOpt | post_backward | 11.644 | 0.495 | 6.030 | 17.674 | 11.644 | 6.030 | 0.654 | 0.240 | 217 | 0 |
| CoLA | DistOpt | post_backward | 11.644 | 0.495 | 6.030 | 17.674 | 11.644 | 6.030 | 0.654 | 0.240 | 217 | 0 |
| CoLA | DistOpt | post_backward | 11.644 | 0.495 | 6.030 | 17.674 | 11.644 | 6.030 | 0.654 | 0.240 | 217 | 0 |
| CoLA | DistOpt | post_backward | 11.644 | 0.495 | 6.030 | 17.674 | 11.644 | 6.030 | 0.654 | 0.240 | 217 | 0 |
| CoLA | DistOpt | post_optimizer_step | 11.644 | 0.495 | 3.870 | 15.514 | 11.644 | 3.870 | 1.074 | 0.240 | 215 | 0 |
| CoLA | DistOpt | post_optimizer_step | 11.644 | 0.495 | 5.838 | 17.482 | 11.644 | 5.838 | 0.463 | 0.240 | 217 | 0 |
| CoLA | DistOpt | post_optimizer_step | 11.644 | 0.495 | 6.030 | 17.674 | 11.644 | 6.030 | 0.654 | 0.240 | 217 | 0 |
| CoLA | DistOpt | post_optimizer_step | 11.644 | 0.495 | 6.030 | 17.674 | 11.644 | 6.030 | 0.654 | 0.240 | 217 | 0 |
| CoLA | DistOpt | post_optimizer_step | 11.644 | 0.495 | 6.030 | 17.674 | 11.644 | 6.030 | 0.654 | 0.240 | 217 | 0 |
| CoLA | DistOpt | post_optimizer_step | 11.644 | 0.495 | 6.030 | 17.674 | 11.644 | 6.030 | 0.654 | 0.240 | 217 | 0 |
| CoLA | DistOpt | post_optimizer_step | 11.644 | 0.495 | 6.030 | 17.674 | 11.644 | 6.030 | 0.654 | 0.240 | 217 | 0 |
| CoLA | DistOpt | post_optimizer_step | 11.644 | 0.495 | 6.030 | 17.674 | 11.644 | 6.030 | 0.654 | 0.240 | 217 | 0 |
| CoLA | DistOpt | post_optimizer_step | 11.644 | 0.495 | 6.030 | 17.674 | 11.644 | 6.030 | 0.654 | 0.240 | 217 | 0 |
| CoLA | DistOpt | post_optimizer_step | 11.644 | 0.495 | 6.030 | 17.674 | 11.644 | 6.030 | 0.654 | 0.240 | 217 | 0 |
| CoLA | DistOpt | post_optimizer_step | 11.644 | 0.495 | 6.030 | 17.674 | 11.644 | 6.030 | 0.654 | 0.240 | 217 | 0 |
| CoLA | DistOpt | post_optimizer_step | 11.644 | 0.495 | 6.030 | 17.674 | 11.644 | 6.030 | 0.654 | 0.240 | 217 | 0 |
| CoLA | DistOpt | steady_state_after_iter_10 | 11.644 | 0.495 | 6.030 | 17.674 | 11.644 | 6.030 | 0.654 | 0.240 | 216 | 0 |
| FullRank | DistOpt | post_init | 20.728 | 0.236 | 4.977 | 25.705 | 20.728 | 4.977 | 1.997 | 0.104 | 41 | 0 |
| FullRank | DistOpt | post_backward | 20.809 | 0.215 | 5.898 | 26.707 | 20.809 | 5.898 | 1.158 | 0.240 | 51 | 0 |
| FullRank | DistOpt | post_backward | 26.743 | 0.472 | 5.720 | 32.463 | 26.743 | 5.720 | 0.979 | 0.191 | 121 | 0 |
| FullRank | DistOpt | post_backward | 26.743 | 0.472 | 5.720 | 32.463 | 26.743 | 5.720 | 0.979 | 0.191 | 121 | 0 |
| FullRank | DistOpt | post_backward | 26.743 | 0.472 | 5.720 | 32.463 | 26.743 | 5.720 | 0.979 | 0.191 | 121 | 0 |
| FullRank | DistOpt | post_backward | 26.743 | 0.472 | 5.720 | 32.463 | 26.743 | 5.720 | 0.979 | 0.191 | 121 | 0 |
| FullRank | DistOpt | post_backward | 26.743 | 0.472 | 5.720 | 32.463 | 26.743 | 5.720 | 0.979 | 0.191 | 121 | 0 |
| FullRank | DistOpt | post_backward | 26.743 | 0.472 | 5.720 | 32.463 | 26.743 | 5.720 | 0.979 | 0.191 | 121 | 0 |
| FullRank | DistOpt | post_backward | 26.743 | 0.472 | 5.720 | 32.463 | 26.743 | 5.720 | 0.979 | 0.191 | 121 | 0 |
| FullRank | DistOpt | post_backward | 26.743 | 0.472 | 5.720 | 32.463 | 26.743 | 5.720 | 0.979 | 0.191 | 121 | 0 |
| FullRank | DistOpt | post_backward | 26.743 | 0.472 | 5.720 | 32.463 | 26.743 | 5.720 | 0.979 | 0.191 | 121 | 0 |
| FullRank | DistOpt | post_backward | 26.743 | 0.472 | 5.720 | 32.463 | 26.743 | 5.720 | 0.979 | 0.191 | 121 | 0 |
| FullRank | DistOpt | post_backward | 26.743 | 0.472 | 5.720 | 32.463 | 26.743 | 5.720 | 0.979 | 0.191 | 121 | 0 |
| FullRank | DistOpt | post_optimizer_step | 26.743 | 0.472 | 3.644 | 30.387 | 26.743 | 3.644 | 3.644 | 0.104 | 119 | 0 |
| FullRank | DistOpt | post_optimizer_step | 26.743 | 0.472 | 5.720 | 32.463 | 26.743 | 5.720 | 0.979 | 0.191 | 121 | 0 |
| FullRank | DistOpt | post_optimizer_step | 26.743 | 0.472 | 5.720 | 32.463 | 26.743 | 5.720 | 0.979 | 0.191 | 121 | 0 |
| FullRank | DistOpt | post_optimizer_step | 26.743 | 0.472 | 5.720 | 32.463 | 26.743 | 5.720 | 0.979 | 0.191 | 121 | 0 |
| FullRank | DistOpt | post_optimizer_step | 26.743 | 0.472 | 5.720 | 32.463 | 26.743 | 5.720 | 0.979 | 0.191 | 121 | 0 |
| FullRank | DistOpt | post_optimizer_step | 26.743 | 0.472 | 5.720 | 32.463 | 26.743 | 5.720 | 0.979 | 0.191 | 121 | 0 |
| FullRank | DistOpt | post_optimizer_step | 26.743 | 0.472 | 5.720 | 32.463 | 26.743 | 5.720 | 0.979 | 0.191 | 121 | 0 |
| FullRank | DistOpt | post_optimizer_step | 26.743 | 0.472 | 5.720 | 32.463 | 26.743 | 5.720 | 0.979 | 0.191 | 121 | 0 |
| FullRank | DistOpt | post_optimizer_step | 26.743 | 0.472 | 5.720 | 32.463 | 26.743 | 5.720 | 0.979 | 0.191 | 121 | 0 |
| FullRank | DistOpt | post_optimizer_step | 26.743 | 0.472 | 5.720 | 32.463 | 26.743 | 5.720 | 0.979 | 0.191 | 121 | 0 |
| FullRank | DistOpt | post_optimizer_step | 26.743 | 0.472 | 5.720 | 32.463 | 26.743 | 5.720 | 0.979 | 0.191 | 121 | 0 |
| FullRank | DistOpt | post_optimizer_step | 26.743 | 0.472 | 5.720 | 32.463 | 26.743 | 5.720 | 0.979 | 0.191 | 121 | 0 |
| FullRank | DistOpt | steady_state_after_iter_10 | 26.743 | 0.472 | 5.720 | 32.463 | 26.743 | 5.720 | 0.979 | 0.191 | 120 | 0 |
| CoLA | DistOpt+offload | post_init | 7.704 | 0.004 | 2.660 | 10.363 | 7.704 | 2.660 | 0.003 | 0.191 | 6 | 0 |
| CoLA | DistOpt+offload | post_backward | 7.783 | 0.254 | 5.662 | 13.445 | 7.783 | 5.662 | 0.296 | 0.240 | 16 | 0 |
| CoLA | DistOpt+offload | post_backward | 7.783 | 0.254 | 6.045 | 13.828 | 7.783 | 6.045 | 0.679 | 0.383 | 18 | 0 |
| CoLA | DistOpt+offload | post_backward | 7.783 | 0.256 | 6.045 | 13.828 | 7.783 | 6.045 | 0.679 | 0.383 | 18 | 0 |
| CoLA | DistOpt+offload | post_backward | 7.783 | 0.254 | 6.045 | 13.828 | 7.783 | 6.045 | 0.679 | 0.383 | 18 | 0 |
| CoLA | DistOpt+offload | post_backward | 7.783 | 0.254 | 6.045 | 13.828 | 7.783 | 6.045 | 0.679 | 0.383 | 18 | 0 |
| CoLA | DistOpt+offload | post_backward | 7.783 | 0.254 | 6.045 | 13.828 | 7.783 | 6.045 | 0.679 | 0.383 | 18 | 0 |
| CoLA | DistOpt+offload | post_backward | 7.783 | 0.254 | 6.045 | 13.828 | 7.783 | 6.045 | 0.679 | 0.383 | 18 | 0 |
| CoLA | DistOpt+offload | post_backward | 7.783 | 0.254 | 6.045 | 13.828 | 7.783 | 6.045 | 0.679 | 0.383 | 18 | 0 |
| CoLA | DistOpt+offload | post_backward | 7.783 | 0.254 | 6.045 | 13.828 | 7.783 | 6.045 | 0.679 | 0.383 | 18 | 0 |
| CoLA | DistOpt+offload | post_backward | 7.783 | 0.254 | 6.045 | 13.828 | 7.783 | 6.045 | 0.679 | 0.383 | 18 | 0 |
| CoLA | DistOpt+offload | post_backward | 7.783 | 0.254 | 6.045 | 13.828 | 7.783 | 6.045 | 0.679 | 0.383 | 18 | 0 |
| CoLA | DistOpt+offload | post_backward | 7.783 | 0.254 | 6.045 | 13.828 | 7.783 | 6.045 | 0.679 | 0.383 | 18 | 0 |
| CoLA | DistOpt+offload | post_optimizer_step | 7.783 | 0.254 | 6.045 | 13.828 | 7.783 | 6.045 | 0.679 | 0.383 | 16 | 0 |
| CoLA | DistOpt+offload | post_optimizer_step | 7.783 | 0.254 | 6.045 | 13.828 | 7.783 | 6.045 | 0.679 | 0.383 | 18 | 0 |
| CoLA | DistOpt+offload | post_optimizer_step | 7.783 | 0.254 | 6.045 | 13.828 | 7.783 | 6.045 | 0.679 | 0.383 | 18 | 0 |
| CoLA | DistOpt+offload | post_optimizer_step | 7.783 | 0.254 | 6.045 | 13.828 | 7.783 | 6.045 | 0.679 | 0.383 | 18 | 0 |
| CoLA | DistOpt+offload | post_optimizer_step | 7.783 | 0.254 | 6.045 | 13.828 | 7.783 | 6.045 | 0.679 | 0.383 | 18 | 0 |
| CoLA | DistOpt+offload | post_optimizer_step | 7.783 | 0.254 | 6.045 | 13.828 | 7.783 | 6.045 | 0.679 | 0.383 | 18 | 0 |
| CoLA | DistOpt+offload | post_optimizer_step | 7.783 | 0.254 | 6.045 | 13.828 | 7.783 | 6.045 | 0.679 | 0.383 | 18 | 0 |
| CoLA | DistOpt+offload | post_optimizer_step | 7.783 | 0.254 | 6.045 | 13.828 | 7.783 | 6.045 | 0.679 | 0.383 | 18 | 0 |
| CoLA | DistOpt+offload | post_optimizer_step | 7.783 | 0.254 | 6.045 | 13.828 | 7.783 | 6.045 | 0.679 | 0.383 | 18 | 0 |
| CoLA | DistOpt+offload | post_optimizer_step | 7.783 | 0.254 | 6.045 | 13.828 | 7.783 | 6.045 | 0.679 | 0.383 | 18 | 0 |
| CoLA | DistOpt+offload | post_optimizer_step | 7.783 | 0.254 | 6.045 | 13.828 | 7.783 | 6.045 | 0.679 | 0.383 | 18 | 0 |
| CoLA | DistOpt+offload | post_optimizer_step | 7.783 | 0.254 | 6.045 | 13.828 | 7.783 | 6.045 | 0.679 | 0.383 | 18 | 0 |
| CoLA | DistOpt+offload | steady_state_after_iter_10 | 7.783 | 0.254 | 6.045 | 13.828 | 7.783 | 6.045 | 0.679 | 0.383 | 17 | 0 |
| FullRank | DistOpt+offload | post_init | 17.763 | 0.003 | 6.011 | 23.773 | 17.763 | 6.011 | 0.065 | 0.191 | 6 | 0 |
| FullRank | DistOpt+offload | post_backward | 17.842 | 0.107 | 6.191 | 24.033 | 17.842 | 6.191 | 0.325 | 0.240 | 16 | 0 |
| FullRank | DistOpt+offload | post_backward | 17.842 | 0.107 | 6.574 | 24.416 | 17.842 | 6.574 | 0.708 | 0.383 | 18 | 0 |
| FullRank | DistOpt+offload | post_backward | 17.842 | 0.109 | 6.574 | 24.416 | 17.842 | 6.574 | 0.708 | 0.383 | 18 | 0 |
| FullRank | DistOpt+offload | post_backward | 17.842 | 0.107 | 6.574 | 24.416 | 17.842 | 6.574 | 0.708 | 0.383 | 18 | 0 |
| FullRank | DistOpt+offload | post_backward | 17.842 | 0.107 | 6.574 | 24.416 | 17.842 | 6.574 | 0.708 | 0.383 | 18 | 0 |
| FullRank | DistOpt+offload | post_backward | 17.842 | 0.107 | 6.574 | 24.416 | 17.842 | 6.574 | 0.708 | 0.383 | 18 | 0 |
| FullRank | DistOpt+offload | post_backward | 17.842 | 0.107 | 6.574 | 24.416 | 17.842 | 6.574 | 0.708 | 0.383 | 18 | 0 |
| FullRank | DistOpt+offload | post_backward | 17.842 | 0.107 | 6.574 | 24.416 | 17.842 | 6.574 | 0.708 | 0.383 | 18 | 0 |
| FullRank | DistOpt+offload | post_backward | 17.842 | 0.107 | 6.574 | 24.416 | 17.842 | 6.574 | 0.708 | 0.383 | 18 | 0 |
| FullRank | DistOpt+offload | post_backward | 17.842 | 0.107 | 6.574 | 24.416 | 17.842 | 6.574 | 0.708 | 0.383 | 18 | 0 |
| FullRank | DistOpt+offload | post_backward | 17.842 | 0.107 | 6.574 | 24.416 | 17.842 | 6.574 | 0.708 | 0.383 | 18 | 0 |
| FullRank | DistOpt+offload | post_backward | 17.842 | 0.107 | 6.574 | 24.416 | 17.842 | 6.574 | 0.708 | 0.383 | 18 | 0 |
| FullRank | DistOpt+offload | post_optimizer_step | 17.842 | 0.107 | 6.574 | 24.416 | 17.842 | 6.574 | 0.708 | 0.383 | 16 | 0 |
| FullRank | DistOpt+offload | post_optimizer_step | 17.842 | 0.107 | 6.574 | 24.416 | 17.842 | 6.574 | 0.708 | 0.383 | 18 | 0 |
| FullRank | DistOpt+offload | post_optimizer_step | 17.842 | 0.107 | 6.574 | 24.416 | 17.842 | 6.574 | 0.708 | 0.383 | 18 | 0 |
| FullRank | DistOpt+offload | post_optimizer_step | 17.842 | 0.107 | 6.574 | 24.416 | 17.842 | 6.574 | 0.708 | 0.383 | 18 | 0 |
| FullRank | DistOpt+offload | post_optimizer_step | 17.842 | 0.107 | 6.574 | 24.416 | 17.842 | 6.574 | 0.708 | 0.383 | 18 | 0 |
| FullRank | DistOpt+offload | post_optimizer_step | 17.842 | 0.107 | 6.574 | 24.416 | 17.842 | 6.574 | 0.708 | 0.383 | 18 | 0 |
| FullRank | DistOpt+offload | post_optimizer_step | 17.842 | 0.107 | 6.574 | 24.416 | 17.842 | 6.574 | 0.708 | 0.383 | 18 | 0 |
| FullRank | DistOpt+offload | post_optimizer_step | 17.842 | 0.107 | 6.574 | 24.416 | 17.842 | 6.574 | 0.708 | 0.383 | 18 | 0 |
| FullRank | DistOpt+offload | post_optimizer_step | 17.842 | 0.107 | 6.574 | 24.416 | 17.842 | 6.574 | 0.708 | 0.383 | 18 | 0 |
| FullRank | DistOpt+offload | post_optimizer_step | 17.842 | 0.107 | 6.574 | 24.416 | 17.842 | 6.574 | 0.708 | 0.383 | 18 | 0 |
| FullRank | DistOpt+offload | post_optimizer_step | 17.842 | 0.107 | 6.574 | 24.416 | 17.842 | 6.574 | 0.708 | 0.383 | 18 | 0 |
| FullRank | DistOpt+offload | post_optimizer_step | 17.842 | 0.107 | 6.574 | 24.416 | 17.842 | 6.574 | 0.708 | 0.383 | 18 | 0 |
| FullRank | DistOpt+offload | steady_state_after_iter_10 | 17.842 | 0.107 | 6.574 | 24.416 | 17.842 | 6.574 | 0.708 | 0.383 | 17 | 0 |
| CoLA | FSDP | post_init | 3.224 | 0.032 | 0.034 | 3.258 | 3.224 | 0.034 | 2.679 | 0.017 | 56 | 0 |
| CoLA | FSDP | post_init | 0.935 | 0.112 | 0.114 | 1.049 | 0.935 | 0.114 | 0.378 | 0.091 | 16 | 0 |
| CoLA | FSDP | post_backward | 3.303 | 0.160 | 11.757 | 15.061 | 3.303 | 11.757 | 4.192 | 0.764 | 66 | 0 |
| CoLA | FSDP | post_backward | 1.015 | 0.237 | 3.454 | 4.469 | 1.015 | 3.454 | 0.873 | 0.764 | 26 | 0 |
| FullRank | FSDP | post_init | 7.412 | 0.073 | 0.075 | 7.486 | 7.412 | 0.075 | 6.313 | 0.036 | 56 | 0 |
| FullRank | FSDP | post_init | 1.633 | 0.044 | 0.046 | 1.680 | 1.633 | 0.046 | 1.252 | 0.036 | 16 | 0 |
| FullRank | FSDP | post_backward | 7.492 | 0.200 | 17.555 | 25.047 | 7.491 | 17.555 | 7.496 | 0.764 | 66 | 0 |
| FullRank | FSDP | post_backward | 1.713 | 0.170 | 5.369 | 7.082 | 1.713 | 5.369 | 1.813 | 0.764 | 26 | 0 |
| CoLA | FSDP+offload | post_init | 3.224 | 0.032 | 0.226 | 3.449 | 3.224 | 0.226 | 2.589 | 0.191 | 55 | 0 |
| CoLA | FSDP+offload | post_backward | 3.303 | 0.269 | 11.765 | 15.068 | 3.303 | 11.765 | 4.203 | 0.764 | 65 | 0 |
| CoLA | FSDP+offload | post_backward | 3.303 | 0.269 | 11.765 | 15.068 | 3.303 | 11.765 | 4.199 | 0.764 | 67 | 0 |
| CoLA | FSDP+offload | post_backward | 3.303 | 0.271 | 11.765 | 15.068 | 3.303 | 11.765 | 4.199 | 0.764 | 67 | 0 |
| CoLA | FSDP+offload | post_backward | 3.303 | 0.269 | 11.765 | 15.068 | 3.303 | 11.765 | 4.199 | 0.764 | 67 | 0 |
| CoLA | FSDP+offload | post_backward | 3.303 | 0.269 | 11.765 | 15.068 | 3.303 | 11.765 | 4.199 | 0.764 | 67 | 0 |
| CoLA | FSDP+offload | post_backward | 3.303 | 0.269 | 11.765 | 15.068 | 3.303 | 11.765 | 4.199 | 0.764 | 67 | 0 |
| CoLA | FSDP+offload | post_backward | 3.303 | 0.269 | 11.765 | 15.068 | 3.303 | 11.765 | 4.199 | 0.764 | 67 | 0 |
| CoLA | FSDP+offload | post_backward | 3.303 | 0.269 | 11.765 | 15.068 | 3.303 | 11.765 | 4.199 | 0.764 | 67 | 0 |
| CoLA | FSDP+offload | post_backward | 3.303 | 0.269 | 11.765 | 15.068 | 3.303 | 11.765 | 4.199 | 0.764 | 67 | 0 |
| CoLA | FSDP+offload | post_backward | 3.303 | 0.269 | 11.765 | 15.068 | 3.303 | 11.765 | 4.199 | 0.764 | 67 | 0 |
| CoLA | FSDP+offload | post_backward | 3.303 | 0.269 | 11.765 | 15.068 | 3.303 | 11.765 | 4.199 | 0.764 | 67 | 0 |
| CoLA | FSDP+offload | post_backward | 3.303 | 0.269 | 11.765 | 15.068 | 3.303 | 11.765 | 4.199 | 0.764 | 67 | 0 |
| CoLA | FSDP+offload | post_optimizer_step | 4.695 | 0.270 | 10.374 | 15.068 | 4.695 | 10.374 | 4.203 | 0.764 | 77 | 0 |
| CoLA | FSDP+offload | post_optimizer_step | 4.695 | 0.270 | 10.374 | 15.068 | 4.695 | 10.374 | 4.199 | 0.764 | 79 | 0 |
| CoLA | FSDP+offload | post_optimizer_step | 4.695 | 0.270 | 10.374 | 15.068 | 4.695 | 10.374 | 4.199 | 0.764 | 79 | 0 |
| CoLA | FSDP+offload | post_optimizer_step | 4.695 | 0.270 | 10.374 | 15.068 | 4.695 | 10.374 | 4.199 | 0.764 | 79 | 0 |
| CoLA | FSDP+offload | post_optimizer_step | 4.695 | 0.270 | 10.374 | 15.068 | 4.695 | 10.374 | 4.199 | 0.764 | 79 | 0 |
| CoLA | FSDP+offload | post_optimizer_step | 4.695 | 0.270 | 10.374 | 15.068 | 4.695 | 10.374 | 4.199 | 0.764 | 79 | 0 |
| CoLA | FSDP+offload | post_optimizer_step | 4.695 | 0.270 | 10.374 | 15.068 | 4.695 | 10.374 | 4.199 | 0.764 | 79 | 0 |
| CoLA | FSDP+offload | post_optimizer_step | 4.695 | 0.270 | 10.374 | 15.068 | 4.695 | 10.374 | 4.199 | 0.764 | 79 | 0 |
| CoLA | FSDP+offload | post_optimizer_step | 4.695 | 0.270 | 10.374 | 15.068 | 4.695 | 10.374 | 4.199 | 0.764 | 79 | 0 |
| CoLA | FSDP+offload | post_optimizer_step | 4.695 | 0.270 | 10.374 | 15.068 | 4.695 | 10.374 | 4.199 | 0.764 | 79 | 0 |
| CoLA | FSDP+offload | post_optimizer_step | 4.695 | 0.270 | 10.374 | 15.068 | 4.695 | 10.374 | 4.199 | 0.764 | 79 | 0 |
| CoLA | FSDP+offload | post_optimizer_step | 4.695 | 0.270 | 10.374 | 15.068 | 4.695 | 10.374 | 4.199 | 0.764 | 79 | 0 |
| CoLA | FSDP+offload | steady_state_after_iter_10 | 4.695 | 0.270 | 10.374 | 15.068 | 4.695 | 10.374 | 4.199 | 0.764 | 78 | 0 |
| FullRank | FSDP+offload | post_init | 7.412 | 0.073 | 0.266 | 7.678 | 7.412 | 0.266 | 6.121 | 0.191 | 55 | 0 |
| FullRank | FSDP+offload | post_backward | 7.491 | 0.309 | 17.081 | 24.572 | 7.491 | 17.081 | 7.021 | 0.764 | 65 | 0 |
| FullRank | FSDP+offload | post_backward | 7.492 | 0.309 | 18.007 | 25.498 | 7.492 | 18.007 | 7.947 | 0.764 | 67 | 0 |
| FullRank | FSDP+offload | post_backward | 7.492 | 0.311 | 18.007 | 25.498 | 7.492 | 18.007 | 7.947 | 0.764 | 67 | 0 |
| FullRank | FSDP+offload | post_backward | 7.492 | 0.309 | 18.007 | 25.498 | 7.492 | 18.007 | 7.947 | 0.764 | 67 | 0 |
| FullRank | FSDP+offload | post_backward | 7.492 | 0.309 | 18.007 | 25.498 | 7.492 | 18.007 | 7.947 | 0.764 | 67 | 0 |
| FullRank | FSDP+offload | post_backward | 7.492 | 0.309 | 18.007 | 25.498 | 7.492 | 18.007 | 7.947 | 0.764 | 67 | 0 |
| FullRank | FSDP+offload | post_backward | 7.492 | 0.309 | 18.007 | 25.498 | 7.492 | 18.007 | 7.947 | 0.764 | 67 | 0 |
| FullRank | FSDP+offload | post_backward | 7.492 | 0.309 | 18.007 | 25.498 | 7.492 | 18.007 | 7.947 | 0.764 | 67 | 0 |
| FullRank | FSDP+offload | post_backward | 7.492 | 0.309 | 18.007 | 25.498 | 7.492 | 18.007 | 7.947 | 0.764 | 67 | 0 |
| FullRank | FSDP+offload | post_backward | 7.492 | 0.309 | 18.007 | 25.498 | 7.492 | 18.007 | 7.947 | 0.764 | 67 | 0 |
| FullRank | FSDP+offload | post_backward | 7.492 | 0.309 | 18.007 | 25.498 | 7.492 | 18.007 | 7.947 | 0.764 | 67 | 0 |
| FullRank | FSDP+offload | post_backward | 7.492 | 0.309 | 18.007 | 25.498 | 7.492 | 18.007 | 7.947 | 0.764 | 67 | 0 |
| FullRank | FSDP+offload | post_optimizer_step | 9.028 | 0.317 | 15.544 | 24.572 | 9.028 | 15.544 | 7.021 | 0.764 | 71 | 0 |
| FullRank | FSDP+offload | post_optimizer_step | 9.028 | 0.317 | 16.470 | 25.498 | 9.028 | 16.470 | 7.947 | 0.764 | 73 | 0 |
| FullRank | FSDP+offload | post_optimizer_step | 9.028 | 0.317 | 16.470 | 25.498 | 9.028 | 16.470 | 7.947 | 0.764 | 73 | 0 |
| FullRank | FSDP+offload | post_optimizer_step | 9.028 | 0.317 | 16.470 | 25.498 | 9.028 | 16.470 | 7.947 | 0.764 | 73 | 0 |
| FullRank | FSDP+offload | post_optimizer_step | 9.028 | 0.317 | 16.470 | 25.498 | 9.028 | 16.470 | 7.947 | 0.764 | 73 | 0 |
| FullRank | FSDP+offload | post_optimizer_step | 9.028 | 0.317 | 16.470 | 25.498 | 9.028 | 16.470 | 7.947 | 0.764 | 73 | 0 |
| FullRank | FSDP+offload | post_optimizer_step | 9.028 | 0.317 | 16.470 | 25.498 | 9.028 | 16.470 | 7.947 | 0.764 | 73 | 0 |
| FullRank | FSDP+offload | post_optimizer_step | 9.028 | 0.317 | 16.470 | 25.498 | 9.028 | 16.470 | 7.947 | 0.764 | 73 | 0 |
| FullRank | FSDP+offload | post_optimizer_step | 9.028 | 0.317 | 16.470 | 25.498 | 9.028 | 16.470 | 7.947 | 0.764 | 73 | 0 |
| FullRank | FSDP+offload | post_optimizer_step | 9.028 | 0.317 | 16.470 | 25.498 | 9.028 | 16.470 | 7.947 | 0.764 | 73 | 0 |
| FullRank | FSDP+offload | post_optimizer_step | 9.028 | 0.317 | 16.470 | 25.498 | 9.028 | 16.470 | 7.947 | 0.764 | 73 | 0 |
| FullRank | FSDP+offload | post_optimizer_step | 9.028 | 0.317 | 16.470 | 25.498 | 9.028 | 16.470 | 7.947 | 0.764 | 73 | 0 |
| FullRank | FSDP+offload | steady_state_after_iter_10 | 9.028 | 0.317 | 16.470 | 25.498 | 9.028 | 16.470 | 7.947 | 0.764 | 72 | 0 |

## FullRank Vs CoLA

| runtime mode | snapshot phase | active allocated reduction GB | inactive split change GB | reserved inactive change GB | reserved-allocated gap change GB | active block count change | CoLA increases fragmentation? |
|---|---|---:|---:|---:|---:|---:|---|
| DistOpt | post_init | 11.737 | -0.093 | -3.054 | -3.054 | 32 | no |
| DistOpt | post_backward | 15.099 | 0.022 | 0.310 | 0.310 | 96 | no |
| DistOpt | post_optimizer_step | 15.099 | 0.022 | 0.310 | 0.310 | 96 | no |
| DistOpt | steady_state_after_iter_10 | 15.099 | 0.022 | 0.310 | 0.310 | 96 | no |
| DistOpt+offload | post_init | 10.059 | 0.000 | -3.351 | -3.351 | 0 | no |
| DistOpt+offload | post_backward | 10.059 | 0.147 | -0.529 | -0.529 | 0 | no |
| DistOpt+offload | post_optimizer_step | 10.059 | 0.147 | -0.529 | -0.529 | 0 | no |
| DistOpt+offload | steady_state_after_iter_10 | 10.059 | 0.147 | -0.529 | -0.529 | 0 | no |
| FSDP | post_init | 0.698 | 0.068 | 0.068 | 0.068 | 0 | no |
| FSDP | post_backward | 0.698 | 0.068 | -1.915 | -1.915 | 0 | no |
| FSDP+offload | post_init | 4.188 | -0.040 | -0.040 | -0.040 | 0 | no |
| FSDP+offload | post_backward | 4.188 | -0.040 | -6.242 | -6.242 | 0 | no |
| FSDP+offload | post_optimizer_step | 4.334 | -0.047 | -6.096 | -6.096 | 6 | no |
| FSDP+offload | steady_state_after_iter_10 | 4.334 | -0.047 | -6.096 | -6.096 | 6 | no |

## CoLA Runtime-System Overhead

| runtime mode | snapshot phase | active allocated delta vs CoLA DistOpt GB | inactive split delta GB | reserved inactive delta GB | reserved-allocated gap delta GB | likely cause |
|---|---|---:|---:|---:|---:|---|
| DistOpt | post_init | 0.000 | 0.000 | 0.000 | 0.000 | baseline |
| DistOpt+offload | post_init | -1.288 | -0.140 | 0.737 | 0.737 | allocator cache/headroom |
| FSDP | post_init | -8.056 | -0.031 | -1.809 | -1.809 | unclear |
| FSDP+offload | post_init | -5.768 | -0.111 | -1.697 | -1.697 | unclear |
| DistOpt | post_backward | 0.000 | 0.000 | 0.000 | 0.000 | baseline |
| DistOpt+offload | post_backward | -3.861 | -0.241 | 0.015 | 0.015 | unclear |
| FSDP | post_backward | -10.629 | -0.257 | -2.576 | -2.576 | unclear |
| FSDP+offload | post_backward | -8.341 | -0.226 | 5.735 | 5.735 | allocator cache/headroom |
| DistOpt | post_optimizer_step | 0.000 | 0.000 | 0.000 | 0.000 | baseline |
| DistOpt+offload | post_optimizer_step | -3.861 | -0.241 | 0.015 | 0.015 | unclear |
| FSDP+offload | post_optimizer_step | -6.949 | -0.224 | 4.344 | 4.344 | allocator cache/headroom |
| DistOpt | steady_state_after_iter_10 | 0.000 | 0.000 | 0.000 | 0.000 | baseline |
| DistOpt+offload | steady_state_after_iter_10 | -3.861 | -0.241 | 0.015 | 0.015 | unclear |
| FSDP+offload | steady_state_after_iter_10 | -6.949 | -0.224 | 4.344 | 4.344 | allocator cache/headroom |

## Interpretation

- `inactive_split_bytes` is the primary allocator-fragmentation proxy.
- `reserved - allocated` and `reserved inactive` are treated as allocator cache/headroom unless inactive split blocks or stack traces indicate fragmentation.
- FSDP/offload conclusions should be tied to active live bytes and stack traces when available, because those systems can create real staging/framework buffers.

### Answers

- Does CoLA itself increase allocator fragmentation under DistOpt? No. At `steady_state_after_iter_10`, CoLA reduces active allocated memory by 15.099 GB versus FullRank. `inactive_split_gb` changes only from 0.472 GB to 0.495 GB (+0.022 GB), and inactive split block count stays 0 in the parsed snapshot records. The small reserved-inactive increase (+0.310 GB) is allocator headroom/cache, not fragmentation by itself.
- Does CoLA reduce active live memory as expected? Yes. Steady-state active allocated memory drops from 26.743 GB to 11.644 GB under DistOpt, from 17.842 GB to 7.783 GB under DistOpt+offload, and from 9.028 GB to 4.695 GB under FSDP+offload.
- Does offload introduce extra staging/caching/reserved memory overhead? Under CoLA DistOpt+offload, active allocated memory is lower than CoLA DistOpt by 3.861 GB and inactive split is lower by 0.241 GB. Reserved-inactive is nearly unchanged at steady state (+0.015 GB), so this run does not show meaningful steady-state GPU allocator overhead from DistOpt optimizer CPU offload.
- Does FSDP introduce extra framework memory overhead or fragmentation? Megatron-FSDP without offload is unsupported for this study result: both FullRank and CoLA full 3B runs, plus both 3B-width 4-layer fallbacks, failed during the first `optimizer.step()` with CUDA illegal memory access / NCCL watchdog abort. Only `post_init` and first `post_backward` snapshots exist, so there is no valid `post_optimizer_step` or steady-state FSDP-only conclusion.
- Does FSDP+offload compound memory overhead? It compounds reserved headroom/cache relative to CoLA DistOpt and CoLA DistOpt+offload, but not allocator fragmentation. CoLA FSDP+offload steady state has active allocated 4.695 GB, inactive split 0.270 GB, reserved inactive 10.374 GB, and reserved-allocated gap 10.374 GB. Versus CoLA DistOpt, active drops by 6.949 GB and inactive split drops by 0.224 GB, while reserved inactive/gap rises by 4.344 GB. Versus CoLA DistOpt+offload, active drops by 3.089 GB, inactive split changes only +0.016 GB, and reserved inactive/gap rises by 4.329 GB. The best-supported classification is FSDP/offload allocator cache/headroom or framework staging/caching overhead, not fragmentation.
- Which metric best supports each answer? Use active allocated bytes for logical live-memory reduction, inactive split bytes for allocator fragmentation, and reserved inactive / reserved-allocated gap for allocator cache/headroom or staging overhead. Do not interpret the whole reserved-allocated gap as fragmentation.

### Caveats

- Top allocation stack trace attribution is limited by the stack trace content in `torch.cuda.memory_snapshot()` for this environment; the report preserves parsed stack tables under `parsed/top_allocation_stacks.csv` where available.
- FSDP-only rows are failed rows with partial snapshots, not measured steady-state rows. Their failure evidence is in the corresponding logs and `RUN_MATRIX.csv`.

## Raw Artifacts

- Raw snapshots: `raw/<run_id>/*.pickle`
- Per-phase stats: `raw/<run_id>/*.stats.json`
- Parsed tables: `parsed/*.csv` and `parsed/*.json`
- Figures: `figures/*.png` and `figures/*.pdf`
