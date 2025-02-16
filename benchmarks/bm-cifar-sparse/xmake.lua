-- -- ----------------------------------------------------------------
-- -- Omp
-- -- ----------------------------------------------------------------

-- target("bm-cifar-sparse-omp")
--     set_kind("binary")
--     set_group("benchmarks")

--     add_includedirs("$(projectdir)/builtin-apps/")
--     add_includedirs("$(projectdir)")

--     add_rules("benchmark_config", "common_flags", "run_on_android")
--     add_files("omp.cpp")

--     add_deps("builtin-apps")


--     add_packages("benchmark")
--     -- add_packages("cli11")

--     add_packages("glm")

--     -- Add openmp support
--     if is_plat("android") then
--         add_cxxflags("-fopenmp -static-openmp")
--         add_ldflags("-fopenmp -static-openmp")
--     else
--         add_packages("openmp")
--     end


-- target_end()

-- -- ----------------------------------------------------------------
-- -- Vulkan
-- -- ----------------------------------------------------------------

-- target("bm-cifar-sparse-vk")
--     set_kind("binary")
--     set_group("benchmarks")

--     add_includedirs("$(projectdir)/builtin-apps/")
--     add_includedirs("$(projectdir)")

--     add_rules("benchmark_config", "common_flags", "vulkan_config", "run_on_android")
--     add_files("vk.cpp")

--     add_deps("builtin-apps", "builtin-apps-vulkan")


--     add_packages("benchmark")
--     -- add_packages("cli11")
--     add_packages("spdlog")
--     add_packages("glm")
--     add_packages("vulkan-hpp", "vulkan-memory-allocator")

--     -- Add openmp support
--     if is_plat("android") then
--         add_cxxflags("-fopenmp -static-openmp")
--         add_ldflags("-fopenmp -static-openmp")
--     else
--         add_packages("openmp")
--     end


-- target_end()

-- -- ----------------------------------------------------------------
-- -- CUDA
-- -- ----------------------------------------------------------------

-- -- if not is_plat("android") then
-- if has_config("cuda") then
--     target("bm-cifar-sparse-cu")
--         set_kind("binary")
--         set_group("benchmarks")
    
--         add_deps("builtin-apps", "builtin-apps-cuda")
--         add_includedirs("$(projectdir)/builtin-apps/")
--         add_includedirs("$(projectdir)")

--         add_rules("benchmark_config")
--         add_files({
--             "cuda.cu",
--         })
    
--         add_packages("spdlog")
--         add_packages("benchmark")
        
--         add_cugencodes("native")
--     target_end()
-- end 


-- ------------------------------------------------------------
-- OMP benchmarks
-- ------------------------------------------------------------

target("bm-cifar-sparse-omp") do
    add_rules("benchmark_config", "common_flags", "run_on_android")
    add_files({
        "omp.cpp",
    })
    add_deps("builtin-apps")
end


-- ------------------------------------------------------------
-- VK benchmarks
-- ------------------------------------------------------------

target("bm-cifar-sparse-vk") do
    add_rules("benchmark_config", "common_flags", "vulkan_config", "run_on_android")
    add_files({
        "vk.cpp",
    })
    add_deps("builtin-apps", "builtin-apps-vulkan")
end


-- ------------------------------------------------------------
-- CUDA benchmarks
-- ------------------------------------------------------------

if has_config("cuda") then
    target("bm-cifar-sparse-cu") do
        add_rules("benchmark_config")
        add_files({
            "cuda.cu",
        })
        add_deps("builtin-apps", "builtin-apps-cuda")
        add_cugencodes("native")
    end
end 