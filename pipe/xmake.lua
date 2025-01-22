add_requires("cli11")

add_requires("concurrentqueue")

if not platform("android") then

target("cuda-omp")
    set_kind("binary")
    add_files("cuda-omp.cpp")

    add_deps("builtin-apps")
    add_deps("builtin-apps-cuda")
    
    add_packages("concurrentqueue")
    add_packages("cli11")
    add_packages("openmp")
target_end()

end
