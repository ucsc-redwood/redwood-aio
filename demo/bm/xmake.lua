
add_requires("benchmark")
add_requires("cli11")

target("bm")
    set_kind("binary")
    add_files("cifar_dense_baseline.cpp")
    add_packages("builtin-apps")
    add_packages("benchmark")
    add_packages("cli11")
    -- Add openmp support
    if is_plat("android") then
        add_cxxflags("-fopenmp -static-openmp")
        add_ldflags("-fopenmp -static-openmp")
    else
        add_packages("openmp")
    end

    if is_plat("android") then
      on_run(run_on_android)
    end
target_end()

target("bm-cifar-dense")
    set_kind("binary")
    add_files("cifar_dense_stages.cpp")
    add_packages("builtin-apps")
    add_packages("benchmark")
    add_packages("cli11")
    -- Add openmp support
    if is_plat("android") then
        add_cxxflags("-fopenmp -static-openmp")
        add_ldflags("-fopenmp -static-openmp")
    else
        add_packages("openmp")
    end

    if is_plat("android") then
      on_run(run_on_android)
    end
target_end()

