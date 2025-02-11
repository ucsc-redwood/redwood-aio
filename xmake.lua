add_rules("mode.debug", "mode.release")

set_languages("c++20")
set_warnings("allextra")


add_requires("benchmark", {system = false})
add_requires("cli11", {system = false})

add_requires("concurrentqueue", {system = false})

if not is_plat("android") then
    add_requires("openmp")
end

add_requires("spdlog", {system = false})
add_requires("glm", {system = false})

includes("android.lua")

-- includes("pipe")
includes("builtin-apps")
includes("tests")
includes("utility")
includes("builtin-apps/common/vulkan")

includes("benchmarks")
