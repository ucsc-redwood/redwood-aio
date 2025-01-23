
if not is_plat("android") then
    add_requires("openmp")
end

target("bm-cifar-dense-stages")
    set_kind("binary")
    set_group("benchmarks")

    add_includedirs("$(projectdir)/builtin-apps/")
    add_files("stages.cpp")

    add_deps("builtin-apps")


    add_packages("benchmark")
    add_packages("cli11")

    add_packages("glm")

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


target("bm-cifar-dense-baselines")
    set_kind("binary")
    set_group("benchmarks")

    add_includedirs("$(projectdir)/builtin-apps/")
    add_files("baselines.cpp")

    add_deps("builtin-apps")


    add_packages("benchmark")
    add_packages("cli11")

    add_packages("glm")

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

