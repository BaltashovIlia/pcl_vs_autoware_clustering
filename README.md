# Build

```bash
mkdir build 
cd build 
cmake .. -DCMAKE_BUILD_TYPE=Release
make 
```

# How to compare result 
```bash
./pcl_vs_autoware_compare ../test/sdc_filtered.pcd 
```

# How to benchmark
```bash
./pcl_vs_autoware_bench ../test/sdc_filtered.pcd 
```