
-- ----------------------------------------------------------------
-- Common test configuration
-- ----------------------------------------------------------------

add_requires("gtest")

rule("test_config")
    on_load(function (target)
        target:set("kind", "binary")
        target:set("group", "test")
        target:add("includedirs", "$(projectdir)/builtin-apps/")
        target:add("includedirs", "$(projectdir)")
        target:add("packages", "gtest")
        target:add("packages", "spdlog")
        target:add("packages", "glm")
    end)
rule_end()

-- ----------------------------------------------------------------
-- Tree Tests
-- ----------------------------------------------------------------

target("test-omp-tree")
    add_rules("test_config", "common_flags", "vulkan_config", "run_on_android")
    add_files("test_omp_tree.cpp")
    add_deps("builtin-apps")

if has_config("cuda") then
target("test-cu-tree")
    add_rules("test_config", "cuda_config")
    add_files("test_cu_tree.cu")
    add_deps("builtin-apps", "builtin-apps-cuda")
    
    -- CUDA-specific OpenMP flags
    add_cuflags("-Xcompiler", "-fopenmp", {force = true})
    add_ldflags("-fopenmp", {force = true})
    add_packages("openmp")
target_end()
end

target("test-vk-tree")
    add_rules("test_config", "common_flags", "vulkan_config", "run_on_android")
    add_files("test_vk_tree.cpp")
    add_deps("builtin-apps", "builtin-apps-vulkan")
target_end()

-- ----------------------------------------------------------------
-- VK Tests
-- ----------------------------------------------------------------

target("test-vk-sort")
    add_rules("test_config", "common_flags", "vulkan_config", "run_on_android")
    add_files("test_vk_sort.cpp")
    add_deps("builtin-apps", "builtin-apps-vulkan")
target_end()


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
