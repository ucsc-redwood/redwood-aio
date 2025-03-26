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

# run-jetsonlowpower-bm:
#     xmake r bm-tree-cu --device jetsonlowpower 
#     xmake r bm-cifar-dense-cu --device jetsonlowpower
#     xmake r bm-cifar-sparse-cu --device jetsonlowpower
#     xmake r bm-tree-omp --device jetsonlowpower 
#     xmake r bm-cifar-dense-omp --device jetsonlowpower
#     xmake r bm-cifar-sparse-omp --device jetsonlowpower

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
#  Run benchmarks 
#  ----------------------------------------------------------------------------

run-jetson-cu-bm:
    xmake r bm-cifar-dense-cu --device jetson
    xmake r bm-cifar-sparse-cu --device jetson
    xmake r bm-tree-cu --device jetson

run-jetson-omp-bm:
    xmake r bm-cifar-dense-omp --device jetson
    xmake r bm-cifar-sparse-omp --device jetson
    xmake r bm-tree-omp --device jetson

run-jetson-bm:
    just run-jetson-cu-bm
    just run-jetson-omp-bm

run-jetsonlowpower-cu-bm:
    xmake r bm-cifar-dense-cu --device jetsonlowpower
    xmake r bm-cifar-sparse-cu --device jetsonlowpower
    xmake r bm-tree-cu --device jetsonlowpower

run-jetsonlowpower-omp-bm:
    xmake r bm-cifar-dense-omp --device jetsonlowpower
    xmake r bm-cifar-sparse-omp --device jetsonlowpower
    xmake r bm-tree-omp --device jetsonlowpower

run-jetsonlowpower-bm:
    just run-jetsonlowpower-cu-bm
    just run-jetsonlowpower-omp-bm

run-android-vk-bm:
    python3 scripts/collect_android_benchmarks.py --device 3A021JEHN02756 --benchmark bm-cifar-dense-vk
    python3 scripts/collect_android_benchmarks.py --device 3A021JEHN02756 --benchmark bm-cifar-sparse-vk
    python3 scripts/collect_android_benchmarks.py --device 3A021JEHN02756 --benchmark bm-tree-vk

    python3 scripts/collect_android_benchmarks.py --device 9b034f1b --benchmark bm-cifar-dense-vk
    python3 scripts/collect_android_benchmarks.py --device 9b034f1b --benchmark bm-cifar-sparse-vk
    python3 scripts/collect_android_benchmarks.py --device 9b034f1b --benchmark bm-tree-vk

run-android-omp-bm:
    python3 scripts/collect_android_benchmarks.py --device 3A021JEHN02756 --benchmark bm-cifar-dense-omp
    python3 scripts/collect_android_benchmarks.py --device 3A021JEHN02756 --benchmark bm-cifar-sparse-omp
    python3 scripts/collect_android_benchmarks.py --device 3A021JEHN02756 --benchmark bm-tree-omp

    python3 scripts/collect_android_benchmarks.py --device 9b034f1b --benchmark bm-cifar-dense-omp
    python3 scripts/collect_android_benchmarks.py --device 9b034f1b --benchmark bm-cifar-sparse-omp
    python3 scripts/collect_android_benchmarks.py --device 9b034f1b --benchmark bm-tree-omp

run-android-bm:
    just run-android-vk-bm
    just run-android-omp-bm

#  ----------------------------------------------------------------------------
#  from google benchmark output (json) to schedules (json)
#  ----------------------------------------------------------------------------

db-to-schedules:
    python3 scripts/gen_schedules.py -d jetson -a CifarDense -b ./data/stable_bm_out_v2/ -o ./data/schedule_files_v2 --top 50
    python3 scripts/gen_schedules.py -d jetson -a CifarSparse -b ./data/stable_bm_out_v2/ -o ./data/schedule_files_v2 --top 50
    python3 scripts/gen_schedules.py -d jetson -a Tree -b ./data/stable_bm_out_v2/ -o ./data/schedule_files_v2 --top 50
    python3 scripts/gen_schedules.py -d jetsonlowpower -a CifarDense -b ./data/stable_bm_out_v2/ -o ./data/schedule_files_v2 --top 50
    python3 scripts/gen_schedules.py -d jetsonlowpower -a CifarSparse -b ./data/stable_bm_out_v2/ -o ./data/schedule_files_v2 --top 50
    python3 scripts/gen_schedules.py -d jetsonlowpower -a Tree -b ./data/stable_bm_out_v2/ -o ./data/schedule_files_v2 --top 50
    python3 scripts/gen_schedules.py -d 3A021JEHN02756 -a CifarDense -b ./data/stable_bm_out_v2/ -o ./data/schedule_files_v2 --top 50
    python3 scripts/gen_schedules.py -d 3A021JEHN02756 -a CifarSparse -b ./data/stable_bm_out_v2/ -o ./data/schedule_files_v2 --top 50
    python3 scripts/gen_schedules.py -d 3A021JEHN02756 -a Tree -b ./data/stable_bm_out_v2/ -o ./data/schedule_files_v2 --top 50
    python3 scripts/gen_schedules.py -d 9b034f1b -a CifarDense -b ./data/stable_bm_out_v2/ -o ./data/schedule_files_v2 --top 50
    python3 scripts/gen_schedules.py -d 9b034f1b -a CifarSparse -b ./data/stable_bm_out_v2/ -o ./data/schedule_files_v2 --top 50
    python3 scripts/gen_schedules.py -d 9b034f1b -a Tree -b ./data/stable_bm_out_v2/ -o ./data/schedule_files_v2 --top 50


# schedules-to-code:
#     python3 scripts/codegen/multi_schedule.py --in_dir data/schedule_files/ --out_dir ./tmp --device jetson --application CifarDense --out_name jetson_cifar_dense_schedules.hpp
#     python3 scripts/codegen/multi_schedule.py --in_dir data/schedule_files/ --out_dir ./tmp --device jetson --application CifarSparse --out_name jetson_cifar_sparse_schedules.hpp
#     python3 scripts/codegen/multi_schedule.py --in_dir data/schedule_files/ --out_dir ./tmp --device jetson --application Tree --out_name jetson_tree_schedules.hpp
#     python3 scripts/codegen/multi_schedule.py --in_dir data/schedule_files/ --out_dir ./tmp --device jetsonlowpower --application CifarDense --out_name jetson_low_power_cifar_dense_schedules.hpp
#     python3 scripts/codegen/multi_schedule.py --in_dir data/schedule_files/ --out_dir ./tmp --device jetsonlowpower --application CifarSparse --out_name jetson_low_power_cifar_sparse_schedules.hpp
#     python3 scripts/codegen/multi_schedule.py --in_dir data/schedule_files/ --out_dir ./tmp --device jetsonlowpower --application Tree --out_name jetson_low_power_tree_schedules.hpp
#     python3 scripts/codegen/multi_schedule.py --in_dir data/schedule_files/ --out_dir ./tmp --device 3A021JEHN02756 --application CifarDense --out_name 3A021JEHN02756_cifar_dense_schedules.hpp
#     python3 scripts/codegen/multi_schedule.py --in_dir data/schedule_files/ --out_dir ./tmp --device 3A021JEHN02756 --application CifarSparse --out_name 3A021JEHN02756_cifar_sparse_schedules.hpp
#     python3 scripts/codegen/multi_schedule.py --in_dir data/schedule_files/ --out_dir ./tmp --device 3A021JEHN02756 --application Tree --out_name 3A021JEHN02756_tree_schedules.hpp
#     python3 scripts/codegen/multi_schedule.py --in_dir data/schedule_files/ --out_dir ./tmp --device 9b034f1b --application CifarDense --out_name 9b034f1b_cifar_dense_schedules.hpp
#     python3 scripts/codegen/multi_schedule.py --in_dir data/schedule_files/ --out_dir ./tmp --device 9b034f1b --application CifarSparse --out_name 9b034f1b_cifar_sparse_schedules.hpp
#     python3 scripts/codegen/multi_schedule.py --in_dir data/schedule_files/ --out_dir ./tmp --device 9b034f1b --application Tree --out_name 9b034f1b_tree_schedules.hpp
    
#     xmake format

schedules-to-code-new:
    python3 scripts/codegen/new_vk.py data/schedule_files/ CifarDense pipe/new-cifar-dense-vk/generated_code.hpp
    python3 scripts/codegen/new_vk.py data/schedule_files/ CifarSparse pipe/new-cifar-sparse-vk/generated_code.hpp
    python3 scripts/codegen/new_vk.py data/schedule_files/ Tree pipe/new-tree-vk/generated_code.hpp

    python3 scripts/codegen/new_cu.py data/schedule_files/ CifarDense pipe/new-cifar-dense-cu/generated_code.cuh
    python3 scripts/codegen/new_cu.py data/schedule_files/ CifarSparse pipe/new-cifar-sparse-cu/generated_code.cuh
    python3 scripts/codegen/new_cu.py data/schedule_files/ Tree pipe/new-tree-cu/generated_code.cuh

    xmake format