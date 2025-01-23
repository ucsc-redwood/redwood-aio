if not is_plat("android") then
    -- set_toolchains("clang")
    add_requires("openmp")
end

add_requires("spdlog")

local source_files = {
    "cifar-dense/dense_appdata.cpp",
    "cifar-sparse/sparse_appdata.cpp",
    "tree/tree_appdata.cpp",
    -- "tree/omp/func_sort.cpp",
    "tree/omp/tree_kernel.cpp",
}

add_requires("glm")

target("builtin-apps")
    set_kind("static")

    add_files(source_files)

    -- Add openmp support
    if is_plat("android") then
        add_cxxflags("-fopenmp -static-openmp")
        add_ldflags("-fopenmp -static-openmp")
    else
        add_packages("openmp")
    end

    add_packages("spdlog")
    add_packages("glm")
target_end()

if not is_plat("android") then

target("builtin-apps-cuda")
    set_kind("static")

    add_files(source_files)

    add_files({
        "common/cuda/cu_mem_resource.cu",
        "cifar-dense/cuda/cu_dense_kernel.cu",
        "cifar-dense/cuda/cu_kernels.cu",
        "cifar-sparse/cuda/cu_dispatcher.cu",
        "cifar-sparse/cuda/cu_kernels.cu",
    })

    add_cugencodes("native")

    add_packages("spdlog")
target_end()

end
