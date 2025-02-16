add_rules("mode.debug", "mode.release")

set_languages("c++20")
set_warnings("allextra")

if not is_plat("android") then
    set_toolchains("clang")
end

add_requires("benchmark")

if not is_plat("android") then
    add_requires("openmp")
end

add_requires("spdlog")
add_requires("glm")

includes("android.lua")

rule("run_on_android")
    if is_plat("android") then
      on_run(run_on_android)
    end
rule_end()

includes("pipe")
includes("builtin-apps")
includes("builtin-apps/common/vulkan")
includes("tests")
includes("utility")
includes("play")

includes("benchmarks")

-- Common configurations
rule("common_flags")
    on_load(function (target)
        -- OpenMP flags for Android
        if is_plat("android") then
            target:add("cxxflags", "-fopenmp -static-openmp")
            target:add("ldflags", "-fopenmp -static-openmp")
        else
            target:add("packages", "openmp")
        end
    end)
rule_end()

-- Vulkan configuration
rule("vulkan_config") 
    on_load(function (target)
        target:add("packages", "vulkan-hpp")
        target:add("packages", "vulkan-memory-allocator")
    end)
rule_end()

-- Common test configuration
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

-- CUDA configuration
rule("cuda_config")
    on_load(function (target)
        target:add("cugencodes", "native")
    end)
rule_end()
