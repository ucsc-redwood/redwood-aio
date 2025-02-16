-- -- ----------------------------------------------------------------
-- -- Benchmark: OMP
-- -- ----------------------------------------------------------------

-- target("bm-tree-omp") do
--     add_rules("benchmark_config", "common_flags", "run_on_android")

--     add_files({
--         "omp.cpp",
--     })

--     add_deps("builtin-apps")
-- target_end()


-- -- ----------------------------------------------------------------
-- -- Benchmark: VK
-- -- ----------------------------------------------------------------

-- target("bm-tree-vk")
--     add_rules("benchmark_config", "common_flags", "vulkan_config", "run_on_android")

--     add_files({
--         "vk.cpp",
--     })

--     add_deps("builtin-apps", "builtin-apps-vulkan")
-- target_end()


-- -- ----------------------------------------------------------------
-- -- Benchmark: CUDA
-- -- ----------------------------------------------------------------

-- -- if not is_plat("android") then
-- if has_config("cuda") then
--     target("bm-tree-cu")
--         add_rules("benchmark_config")

--         add_files({
--             "cuda.cu",
--         })

--         add_deps("builtin-apps", "builtin-apps-cuda")
--         add_cugencodes("native")
--     target_end()
-- end 

-- ------------------------------------------------------------
-- OMP benchmarks
-- ------------------------------------------------------------

target("bm-tree-omp") do
    add_rules("benchmark_config", "common_flags", "run_on_android")
    add_files({
        "omp.cpp",
    })
    add_deps("builtin-apps")
end


-- ------------------------------------------------------------
-- VK benchmarks
-- ------------------------------------------------------------

target("bm-tree-vk") do
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
    target("bm-tree-cu") do
        add_rules("benchmark_config")
        add_files({
            "cuda.cu",
        })
        add_deps("builtin-apps", "builtin-apps-cuda")
        add_cugencodes("native")
    end
end 