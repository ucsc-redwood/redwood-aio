# Set configuration for Android devices (on a machine using ADB)
set-android:
    xmake f -p android -a arm64-v8a --ndk=~/Android/Sdk/ndk/28.0.13004108/ --android_sdk=~/Android/Sdk/ --ndk_sdkver=28 -c -v --use_vulkan=yes --use_cuda=no -m releasedbg

# Set configuration for NVIDIA Jetson Orin
set-jetson:
    xmake f -p linux -a arm64 --use_cuda=yes --use_vulkan=no -c -v -m releasedbg

# Set default configuration for PC
set-default:
    xmake f -p linux -a x86_64 -c -v --use_vulkan=no --use_cuda=yes -m releasedbg

# Compile Vulkan shader (need xxd)
compile-shader:
    make

# Convert raw google benchmark data (in ./data/raw_bm_results) to sqlite database
raw-to-db:
    python3 scripts/database/update_db.py

db-to-schedules:
    python3 scripts/analysis/gen_schedules.py --device 3A021JEHN02756 --app CifarDense 
    python3 scripts/analysis/gen_schedules.py --device 3A021JEHN02756 --app CifarSparse 
    python3 scripts/analysis/gen_schedules.py --device 3A021JEHN02756 --app Tree 
    
    python3 scripts/analysis/gen_schedules.py --device 9b034f1b --app CifarDense 
    python3 scripts/analysis/gen_schedules.py --device 9b034f1b --app CifarSparse 
    python3 scripts/analysis/gen_schedules.py --device 9b034f1b --app Tree 
    
    python3 scripts/analysis/gen_schedules.py --device ce0717178d7758b00b7e --app CifarDense 
    python3 scripts/analysis/gen_schedules.py --device ce0717178d7758b00b7e --app CifarSparse 
    python3 scripts/analysis/gen_schedules.py --device ce0717178d7758b00b7e --app Tree 

# python3 scripts/codegen/multi_schedule.py --in_dir data/generated-schedules/ --out_dir pipe/cifar-dense-cu/generated-code/ --device jetson --application CifarDense

# Generate pipeline code from sqlite database (in ./data/generated-schedules)
schedules-to-code:
    python3 scripts/analysis/gen_pipes.py --in_dir ./data/generated-schedules/ --out_dir pipe/cifar-dense-vk/generated-code/ --application CifarDense
    python3 scripts/analysis/gen_pipes.py --in_dir ./data/generated-schedules/ --out_dir pipe/cifar-sparse-vk/generated-code/ --application CifarSparse
    python3 scripts/analysis/gen_pipes.py --in_dir ./data/generated-schedules/ --out_dir pipe/tree-vk/generated-code/ --application Tree
    
    xmake format

# Remove all temporary files from Android devices, then push resources folder to devices
rm-android-tmp:
    adb -s 3A021JEHN02756 shell "rm -rf /data/local/tmp/*"
    adb -s 9b034f1b shell "rm -rf /data/local/tmp/*"
    adb -s ce0717178d7758b00b7e shell "rm -rf /data/local/tmp/*"
    
    xmake push-all-resources

# List all files in the temporary directory of Android devices
cat-android-tmp:
    adb -s 3A021JEHN02756 shell "ls -la /data/local/tmp"
    adb -s 9b034f1b shell "ls -la /data/local/tmp"
    adb -s ce0717178d7758b00b7e shell "ls -la /data/local/tmp"

#  ----------------------------------------------------------------------------
#  Device-specific benchmarks
#  ----------------------------------------------------------------------------

run-jetson-bm:
    xmake r bm-tree-cu --device jetson 
    xmake r bm-cifar-dense-cu --device jetson
    xmake r bm-cifar-sparse-cu --device jetson
    xmake r bm-tree-omp --device jetson 
    xmake r bm-cifar-dense-omp --device jetson
    xmake r bm-cifar-sparse-omp --device jetson

cuda-codegen-jetson:
    python3 scripts/codegen/multi_schedule.py --in_dir data/generated-schedules/ --out_dir pipe/cifar-dense-cu/generated-code/ --device jetson --application CifarDense
    python3 scripts/codegen/multi_schedule.py --in_dir data/generated-schedules/ --out_dir pipe/cifar-sparse-cu/generated-code/ --device jetson --application CifarSparse
    python3 scripts/codegen/multi_schedule.py --in_dir data/generated-schedules/ --out_dir pipe/tree-cu/generated-code/ --device jetson --application Tree
    xmake format

run-minipc-bm:
    xmake r bm-tree-vk --device minipc 
    xmake r bm-cifar-dense-vk --device minipc
    xmake r bm-cifar-sparse-vk --device minipc
    xmake r bm-tree-omp --device minipc 
    xmake r bm-cifar-dense-omp --device minipc
    xmake r bm-cifar-sparse-omp --device minipc

#  ----------------------------------------------------------------------------
#  Android
#  ----------------------------------------------------------------------------

run-android-bm:
    python3 scripts/collect_android_results.py




