
set-android:
    xmake f -p android -a arm64-v8a --ndk=~/Android/Sdk/ndk/28.0.12674087/ --android_sdk=~/Android/Sdk/ --ndk_sdkver=28 -c -v

set-jetson:
    xmake f -p linux -a arm64 -c -v

set-default:
    xmake f -p linux -a x86_64 -c

push-all-resources:
    xmake push-all-resources

run-bm-cu:
    xmake run bm-cifar-dense-cu
    xmake run bm-cifar-sparse-cu
    xmake run bm-tree-cu

run-bm-vk:
    xmake run bm-cifar-dense-vk
    xmake run bm-cifar-sparse-vk
    xmake run bm-tree-vk

run-bm-omp-android:
    xmake run bm-cifar-dense-omp
    xmake run bm-cifar-sparse-omp
    xmake run bm-tree-omp

run-bm-omp-jetson:
    xmake run bm-cifar-dense-omp -d jetson
    xmake run bm-cifar-sparse-omp -d jetson
    xmake run bm-tree-omp -d jetson
