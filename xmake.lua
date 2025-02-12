add_rules("mode.debug", "mode.release")

set_languages("c++20")
set_warnings("allextra")

set_toolchains("clang")

add_requires("benchmark")

-- add_requires("concurrentqueue")

if not is_plat("android") then
    add_requires("openmp")
end

add_requires("spdlog")
add_requires("glm")

includes("android.lua")

includes("pipe")
includes("builtin-apps")
includes("builtin-apps/common/vulkan")
includes("tests")
includes("utility")

includes("benchmarks")
