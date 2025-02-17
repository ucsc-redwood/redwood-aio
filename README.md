# Redwood All-in-One

## Overview

This repository contains the source code for the Redwood project, which is a collection of benchmarks and tools for evaluating the performance of various programming models. And creating pipeline execution for applications on heterogeneous systems. 


## Developer Notes

### Collecting benchmarks

Baseline

```bash
xmake r bm-cifar-dense-omp --benchmark_repetitions=5 --benchmark_filter=Baseline --benchmark_format=json > ./data/raw_benchmarks/cifar-dense-baseline.json
```









