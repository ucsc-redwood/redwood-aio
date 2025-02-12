-- ----------------------------------------------------------------
-- OMP
-- ----------------------------------------------------------------

target("bm-tree-omp")
    set_kind("binary")
    set_group("benchmarks")

    add_includedirs("$(projectdir)/builtin-apps/")
    add_includedirs("$(projectdir)")

    add_files("omp.cpp")

    add_deps("builtin-apps")


    add_packages("benchmark")
    -- add_packages("cli11")

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


-- ----------------------------------------------------------------
-- VK
-- ----------------------------------------------------------------

target("bm-tree-vk")
    set_kind("binary")
    set_group("benchmarks")

    add_includedirs("$(projectdir)/builtin-apps/")
    add_includedirs("$(projectdir)")
    add_files("vk.cpp")

    add_deps("builtin-apps")
    add_deps("builtin-apps-vulkan")


    add_packages("benchmark")
    -- add_packages("cli11")
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


-- ----------------------------------------------------------------
-- CUDA
-- ----------------------------------------------------------------

-- if not is_plat("android") then
if has_config("cuda") then
    
    target("bm-tree-cu")
        set_kind("binary")
        set_group("benchmarks")
    
        add_deps("builtin-apps")
        add_deps("builtin-apps-cuda")
        add_includedirs("$(projectdir)/builtin-apps/")
        add_includedirs("$(projectdir)")
    
        add_files({
            "cuda.cu",
            -- "../../builtin-apps/common/cuda/cu_mem_resource.cu",
        })
    
        add_packages("spdlog")
        add_packages("benchmark")
        -- add_packages("cli11")
    
        add_cugencodes("native")
    
    
    target_end()
    
end 