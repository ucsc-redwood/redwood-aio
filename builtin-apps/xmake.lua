if not is_plat("android") then
    -- set_toolchains("clang")
    add_requires("openmp")
end

add_requires("spdlog")

local source_files = {
    "cifar-dense/dense_appdata.cpp",
    "cifar-sparse/sparse_appdata.cpp",
}

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
target_end()

target("builtin-apps-cuda")
    set_kind("static")

    add_files(source_files)

    add_files({
        "common/cuda/cu_mem_resource.cu",
        "cifar-dense/cuda/cu_dense_kernel.cu",
        "cifar-dense/cuda/cu_kernels.cu",
        "cifar-sparse/cuda/cu_dispatcher.cu",
        "cifar-sparse/cuda/cu_sparse_kernel.cu",
    })

    add_cugencodes("native")

    add_packages("spdlog")
target_end()


target("test_dense")
    set_kind("binary")
    set_default(false)
    add_files("cifar-dense/test_dense.cpp")
    add_deps("builtin-apps")

    -- Add openmp support
    if is_plat("android") then
        add_cxxflags("-fopenmp -static-openmp")
        add_ldflags("-fopenmp -static-openmp")
    else
        add_packages("openmp")
    end

    add_tests("default")
    add_tests("pass_output", {trim_output = true, pass_outputs = "Predicted Image: airplanes"})
    add_tests("fail_output", {trim_output = true, fail_outputs = "Predicted Image: birds"})

    add_tests("with_omp", {trim_output = true, args = {"true"}, pass_outputs = "Predicted Image: airplanes"})
target_end()

target("test_sparse")
    set_kind("binary")
    set_default(false)
    add_files("cifar-sparse/test_sparse.cpp")
    add_deps("builtin-apps")

    -- Add openmp support
    if is_plat("android") then
        add_cxxflags("-fopenmp -static-openmp")
        add_ldflags("-fopenmp -static-openmp")
    else
        add_packages("openmp")
    end

    add_tests("default")
    add_tests("pass_output", {trim_output = true, pass_outputs = "Predicted Image: deer"})
    add_tests("fail_output", {trim_output = true, fail_outputs = "Predicted Image: airplanes"})

    add_tests("with_omp", {trim_output = true, runargs = {"true"}, pass_outputs = "Predicted Image: deer"})
target_end()

set_policy("build.cuda.devlink", true)

target("test_dense_cuda")
    set_kind("binary")
    set_default(false)
    add_files("cifar-dense/test_dense_cuda.cu")
    add_deps("builtin-apps")
    add_deps("builtin-apps-cuda")

    add_cugencodes("native")

    add_packages("spdlog")

    add_tests("default")
    add_tests("pass_output", {trim_output = true, pass_outputs = "Predicted Image: airplanes"})
    add_tests("fail_output", {trim_output = true, fail_outputs = "Predicted Image: birds"})
target_end()
