#  ----------------------------------------------------------------------------
#  Setup Configuration
#  ----------------------------------------------------------------------------

# Set configuration for Android devices (on a machine using ADB)
# drwxrwxr-x 11 doremy doremy 4.0K Oct 16 12:23 26.1.10909125/
# drwxrwxr-x 11 doremy doremy 4.0K Oct 16 12:41 27.0.12077973/
# drwxrwxr-x 11 doremy doremy 4.0K Oct 16 13:12 28.0.12433566/
# drwxrwxr-x 11 doremy doremy 4.0K Dec  7 12:08 28.0.12674087/
# drwxrwxr-x 11 doremy doremy 4.0K Feb 17 00:08 28.0.13004108/
# drwxrwxr-x 11 doremy doremy 4.0K Mar 10 12:24 29.0.13113456/
set-android:
    xmake f -p android -a arm64-v8a --ndk=~/Android/Sdk/ndk/29.0.13113456/ --android_sdk=~/Android/Sdk/ --ndk_sdkver=29 -c -v --use_vulkan=yes --use_cuda=no -m release

# Set configuration for NVIDIA Jetson Orin
set-jetson:
    xmake f -p linux -a arm64 --use_cuda=yes --use_vulkan=no -c -v -m release

# Set default configuration for PC
set-default:
    xmake f -p linux -a x86_64 -c -v --use_vulkan=no --use_cuda=yes -m release

#  ----------------------------------------------------------------------------
#  Compile Shaders
#  ----------------------------------------------------------------------------

# Compile Vulkan shader (need xxd)
compile-shader:
    make

compile_commands:
    xmake project -k compile_commands
    sed -i 's/"-rdc=true",//g' compile_commands.json

#  ----------------------------------------------------------------------------
#  Benchmark Related
#  ----------------------------------------------------------------------------

# # Convert raw google benchmark data (in ./data/raw_bm_results) to sqlite database
# raw-to-db:
#     python3 scripts/database/update_db.py

# db-to-schedules:
#     python3 scripts/analysis/gen_schedules.py --device 3A021JEHN02756 --app CifarDense 
#     python3 scripts/analysis/gen_schedules.py --device 3A021JEHN02756 --app CifarSparse 
#     python3 scripts/analysis/gen_schedules.py --device 3A021JEHN02756 --app Tree 
    
#     python3 scripts/analysis/gen_schedules.py --device 9b034f1b --app CifarDense 
#     python3 scripts/analysis/gen_schedules.py --device 9b034f1b --app CifarSparse 
#     python3 scripts/analysis/gen_schedules.py --device 9b034f1b --app Tree 
    

# python3 scripts/codegen/multi_schedule.py --in_dir data/generated-schedules/ --out_dir pipe/cifar-dense-cu/generated-code/ --device jetson --application CifarDense

# # Generate pipeline code from sqlite database (in ./data/generated-schedules)
# schedules-to-code:
#     python3 scripts/analysis/gen_pipes.py --in_dir ./data/generated-schedules/ --out_dir pipe/cifar-dense-vk/generated-code/ --application CifarDense
#     python3 scripts/analysis/gen_pipes.py --in_dir ./data/generated-schedules/ --out_dir pipe/cifar-sparse-vk/generated-code/ --application CifarSparse
#     python3 scripts/analysis/gen_pipes.py --in_dir ./data/generated-schedules/ --out_dir pipe/tree-vk/generated-code/ --application Tree
    
#     xmake format

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

# #  ----------------------------------------------------------------------------
# #  Device-specific benchmarks
# #  ----------------------------------------------------------------------------

# run-jetson-bm:
#     xmake r bm-tree-cu --device jetson 
#     xmake r bm-cifar-dense-cu --device jetson
#     xmake r bm-cifar-sparse-cu --device jetson
#     xmake r bm-tree-omp --device jetson 
#     xmake r bm-cifar-dense-omp --device jetson
#     xmake r bm-cifar-sparse-omp --device jetson

# run-jetson-low-power-bm:
#     xmake r bm-tree-cu --device jetson-low-power 
#     xmake r bm-cifar-dense-cu --device jetson-low-power
#     xmake r bm-cifar-sparse-cu --device jetson-low-power
#     xmake r bm-tree-omp --device jetson-low-power 
#     xmake r bm-cifar-dense-omp --device jetson-low-power
#     xmake r bm-cifar-sparse-omp --device jetson-low-power

# cuda-codegen-jetson:
#     python3 scripts/codegen/multi_schedule.py --in_dir data/generated-schedules/ --out_dir pipe/cifar-dense-cu/generated-code/ --device jetson --application CifarDense
#     python3 scripts/codegen/multi_schedule.py --in_dir data/generated-schedules/ --out_dir pipe/cifar-sparse-cu/generated-code/ --device jetson --application CifarSparse
#     python3 scripts/codegen/multi_schedule.py --in_dir data/generated-schedules/ --out_dir pipe/tree-cu/generated-code/ --device jetson --application Tree
#     xmake format

# run-minipc-bm:
#     xmake r bm-tree-vk --device minipc 
#     xmake r bm-cifar-dense-vk --device minipc
#     xmake r bm-cifar-sparse-vk --device minipc
#     xmake r bm-tree-omp --device minipc 
#     xmake r bm-cifar-dense-omp --device minipc
#     xmake r bm-cifar-sparse-omp --device minipc

# run-pc-bm:
#     # xmake r bm-tree-vk --device pc 
#     # xmake r bm-cifar-dense-vk --device pc
#     # xmake r bm-cifar-sparse-vk --device pc
#     # xmake r bm-tree-omp --device pc 
#     xmake r bm-cifar-dense-omp --device pc
#     xmake r bm-cifar-sparse-omp --device pc
#     xmake r bm-cifar-dense-cu --device pc
#     xmake r bm-cifar-sparse-cu --device pc

# #  ----------------------------------------------------------------------------
# #  Android
# #  ----------------------------------------------------------------------------

# #    python3 scripts/collect_android_benchmarks.py --device 3A021JEHN02756 --benchmark bm-cifar-dense-omp
# #    python3 scripts/collect_android_benchmarks.py --device 3A021JEHN02756 --benchmark bm-cifar-sparse-omp
# #    python3 scripts/collect_android_benchmarks.py --device 3A021JEHN02756 --benchmark bm-tree-omp
# # python3 scripts/collect_android_benchmarks.py --device 9b034f1b --benchmark bm-cifar-dense-omp
# # python3 scripts/collect_android_benchmarks.py --device 9b034f1b --benchmark bm-cifar-sparse-omp
# # python3 scripts/collect_android_benchmarks.py --device 9b034f1b --benchmark bm-tree-omp
# #    python3 scripts/collect_android_benchmarks.py --device 3A021JEHN02756 --benchmark bm-cifar-dense-vk
# #    python3 scripts/collect_android_benchmarks.py --device 3A021JEHN02756 --benchmark bm-cifar-sparse-vk
# #    python3 scripts/collect_android_benchmarks.py --device 3A021JEHN02756 --benchmark bm-tree-vk

# run-android-bm:
#     python3 scripts/collect_android_benchmarks.py --device 9b034f1b --benchmark bm-cifar-dense-vk
#     python3 scripts/collect_android_benchmarks.py --device 9b034f1b --benchmark bm-cifar-sparse-vk
#     python3 scripts/collect_android_benchmarks.py --device 9b034f1b --benchmark bm-tree-vk


#  ----------------------------------------------------------------------------
#  from google benchmark output (json) to schedules (json)
#  ----------------------------------------------------------------------------

db-to-schedules:
    python3 scripts/gen_schedules.py -d jetson -a CifarDense -b ./data/stable-benchmark-out/ -o ./data/schedule_files
    python3 scripts/gen_schedules.py -d jetson -a CifarSparse -b ./data/stable-benchmark-out/ -o ./data/schedule_files
    python3 scripts/gen_schedules.py -d jetson -a Tree -b ./data/stable-benchmark-out/ -o ./data/schedule_files
    python3 scripts/gen_schedules.py -d jetson-low-power -a CifarDense -b ./data/stable-benchmark-out/ -o ./data/schedule_files
    python3 scripts/gen_schedules.py -d jetson-low-power -a CifarSparse -b ./data/stable-benchmark-out/ -o ./data/schedule_files
    python3 scripts/gen_schedules.py -d jetson-low-power -a Tree -b ./data/stable-benchmark-out/ -o ./data/schedule_files
    python3 scripts/gen_schedules.py -d 3A021JEHN02756 -a CifarDense -b ./data/stable-benchmark-out/ -o ./data/schedule_files
    python3 scripts/gen_schedules.py -d 3A021JEHN02756 -a CifarSparse -b ./data/stable-benchmark-out/ -o ./data/schedule_files
    python3 scripts/gen_schedules.py -d 3A021JEHN02756 -a Tree -b ./data/stable-benchmark-out/ -o ./data/schedule_files
    python3 scripts/gen_schedules.py -d 9b034f1b -a CifarDense -b ./data/stable-benchmark-out/ -o ./data/schedule_files
    python3 scripts/gen_schedules.py -d 9b034f1b -a CifarSparse -b ./data/stable-benchmark-out/ -o ./data/schedule_files
    python3 scripts/gen_schedules.py -d 9b034f1b -a Tree -b ./data/stable-benchmark-out/ -o ./data/schedule_files

