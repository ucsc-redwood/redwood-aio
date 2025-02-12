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

includes("pipe")
includes("builtin-apps")
includes("builtin-apps/common/vulkan")
includes("tests")
includes("utility")
includes("play")

includes("benchmarks")
