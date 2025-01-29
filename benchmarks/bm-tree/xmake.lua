target("bm-tree-stages")
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


target("bm-tree-baselines")
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


target("bm-tree-vk")
    set_kind("binary")
    set_group("benchmarks")

    add_includedirs("$(projectdir)/builtin-apps/")
    add_files("vk.cpp")

    add_deps("builtin-apps")


    add_packages("benchmark")
    add_packages("cli11")
    add_packages("spdlog")
    add_packages("glm")

    -- Add openmp support
    if is_plat("android") then
        add_cxxflags("-fopenmp -static-openmp")
        add_ldflags("-fopenmp -static-openmp")
    else
        add_packages("openmp")
    end

    add_packages("vulkan-hpp", "vulkan-memory-allocator")



    if is_plat("android") then
      on_run(run_on_android)
    end
target_end()


if not is_plat("android") then

    target("bm-tree-cuda")
        set_kind("binary")
        set_group("benchmarks")
    
        add_deps("builtin-apps")
        add_deps("builtin-apps-cuda")
        add_includedirs("$(projectdir)/builtin-apps/")
    
        add_files({
            "cuda.cu",
            "../../builtin-apps/common/cuda/cu_mem_resource.cu",
        })
    
        add_packages("spdlog")
        add_packages("benchmark")
        add_packages("cli11")
    
        add_cugencodes("native")
    
    
    target_end()
    
end 