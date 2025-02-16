-- Copyright (c) 2025 Yanwen Xu (yxu83@ucsc.edu). MIT License.

add_rules("mode.debug", "mode.release")

set_languages("c++20")
set_warnings("allextra")

-- For consistency across Windows, Linux
if not is_plat("android") then
    set_toolchains("clang")
end

-- ----------------------------------------------------------------
-- Common packages used in the project
-- ----------------------------------------------------------------

-- currently using:
-- - cli11-v2.4.2
-- - spdlog-v1.14.1
-- - glm-1.0.1
add_requires("spdlog")  -- everything
add_requires("cli11") -- all binaries
add_requires("glm") -- tree applications

-- OpenMP is handled differently on Android
if not is_plat("android") then
    add_requires("openmp")
end

-- Common configurations
rule("common_flags")
    on_load(function (target)
        -- OpenMP flags for Android (special case)
        if is_plat("android") then
            target:add("cxxflags", "-fopenmp -static-openmp")
            target:add("ldflags", "-fopenmp -static-openmp")
        else
            target:add("packages", "openmp")
        end

        -- Add common packages to the target
        target:add("packages", "cli11")
        target:add("packages", "spdlog")
        target:add("packages", "glm")
    end)
rule_end()

-- ----------------------------------------------------------------
-- Vulkan configuration
-- ----------------------------------------------------------------

rule("vulkan_config") 
    on_load(function (target)
        target:add("packages", "vulkan-headers")
        target:add("packages", "vulkan-hpp")
        target:add("packages", "vulkan-memory-allocator")
    end)
rule_end()

-- ----------------------------------------------------------------
-- Android configuration
-- ----------------------------------------------------------------

includes("android.lua")

rule("run_on_android")
    if is_plat("android") then
      on_run(run_on_android)
    end
rule_end()

-- ----------------------------------------------------------------
-- Projects
-- ----------------------------------------------------------------

includes("builtin-apps/common/vulkan") -- KISS-VK library
includes("builtin-apps")
includes("tests")
includes("benchmarks")
includes("pipe")
includes("utility")
