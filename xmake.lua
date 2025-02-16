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
    -- on_run(function (target)
    --     os.runv("adb", "shell", "su", "-c", target:targetfile())
    -- end)
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
