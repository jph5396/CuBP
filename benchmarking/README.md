# Benchmarking

## prerequisites 
1. Download the CPHD files listed here. 

|Image Name|link|Shape|Size (GB)|Notes| 
|----------|----|-----|----|-----|
|capella-egpyt.cphd|[link](https://capella-open-data.s3.amazonaws.com/data/2024/10/4/CAPELLA_C13_SP_CPHD_HH_20241004001939_20241004002012/CAPELLA_C13_SP_CPHD_HH_20241004001939_20241004002012.cphd) | ( 335,149 x 29,160 x 2 ) | 2.2 GB | Scene over the Pyramids of Giza |
| jeddah-tower.cphd | [link](http://umbra-open-data-catalog.s3.amazonaws.com/sar-data/tasks/Jeddah%20Tower%2C%20Saudi%20Arabia/3919a6cc-62e9-440e-a64c-598deed888d0/2024-12-22-07-43-39_UMBRA-08/2024-12-22-07-43-39_UMBRA-08_CPHD.cphd) | ( 84,866 x 71280 x 2 ) | 46 GB | Scene over the construction of the jeddah tower in Saudi Arabia, which will be the tallest building in the world once complete (at least as of May 2026). | 

2. Follow the install instructions listed in the repo's readme. 

## Running the benchmark
After downloading the data and installing the cubp package, the benchmarks can be repeated using the `./benchmark.sh` script in this repository. It is set up with presets to run from this directory. 

## Results
The benchmark was ran on an Nvidia DGX Spark with CUDA 13.0. 