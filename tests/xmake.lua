
-- ----------------------------------------------------------------
-- Common test configuration
-- ----------------------------------------------------------------

-- currently using:
-- - gtest-v1.15.2:

add_requires("gtest")

rule("test_config")
    on_load(function (target)
        target:set("kind", "binary")
        target:set("group", "test")
        target:add("includedirs", "$(projectdir)/builtin-apps/")
        target:add("includedirs", "$(projectdir)")
        target:add("packages", "gtest")
    end)
rule_end()

-- ----------------------------------------------------------------
-- OMP-based Tree Tests
-- ----------------------------------------------------------------

target("test-omp-tree") do
    add_rules("test_config", "common_flags", "run_on_android")
    add_files({
        "test_omp_tree.cpp",
    })
    add_deps("builtin-apps")
end 

-- ----------------------------------------------------------------
-- CUDA-based Tree Tests
-- ----------------------------------------------------------------

-- if has_config("cuda") then
--     target("test-cu-tree") do
--         add_rules("test_config")
--         add_files("test_cu_tree.cu")
--         add_deps("builtin-apps", "builtin-apps-cuda")
        
--         -- CUDA-specific OpenMP flags
--         add_cuflags("-Xcompiler", "-fopenmp", {force = true})
--         add_ldflags("-fopenmp", {force = true})
--         add_packages("openmp")
--         add_cugencodes("native")
--     end
-- end

-- ----------------------------------------------------------------
-- VK-based Tree Tests
-- ----------------------------------------------------------------

target("test-vk-tree") do
    add_rules("test_config", "common_flags", "vulkan_config", "run_on_android")
    add_files({
        "test_vk_tree.cpp",
    })
    add_deps("builtin-apps", "builtin-apps-vulkan")
end

-- ----------------------------------------------------------------
-- VK primitive tests (e.g., radix sort, prefix sum)
-- ----------------------------------------------------------------

target("test-vk-sort") do
    add_rules("test_config", "common_flags", "vulkan_config", "run_on_android")
    add_files({
        "test_vk_sort.cpp",
    })
    add_deps("builtin-apps", "builtin-apps-vulkan")
end


-- target("test-vk-prefix-sum")
--     set_kind("binary")
--     set_group("test")

--     add_includedirs("$(projectdir)/builtin-apps/")
--     add_files("test_vk_prefix_sum.cpp")

--     add_deps("builtin-apps")

--     add_packages("benchmark")
--     add_packages("gtest")

--     add_packages("cli11")
--     add_packages("spdlog")
--     add_packages("glm")

--     -- Add openmp support
--     if is_plat("android") then
--         add_cxxflags("-fopenmp -static-openmp")
--         add_ldflags("-fopenmp -static-openmp")
--     else
--         add_packages("openmp")
--     end

--     add_packages("vulkan-hpp", "vulkan-memory-allocator")

--     if is_plat("android") then
--       on_run(run_on_android)
--     end
-- target_end()



-- target("test-vk-prefix-sum-v2")
--     set_kind("binary")
--     set_group("test")

--     add_includedirs("$(projectdir)/builtin-apps/")
--     add_files("test_vk_prefix_sum_v2.cpp")

--     add_deps("builtin-apps")

--     add_packages("benchmark")
--     add_packages("gtest")

--     add_packages("cli11")
--     add_packages("spdlog")
--     add_packages("glm")

--     -- Add openmp support
--     if is_plat("android") then
--         add_cxxflags("-fopenmp -static-openmp")
--         add_ldflags("-fopenmp -static-openmp")
--     else
--         add_packages("openmp")
--     end

--     add_packages("vulkan-hpp", "vulkan-memory-allocator")

--     if is_plat("android") then
--       on_run(run_on_android)
--     end
-- target_end()
