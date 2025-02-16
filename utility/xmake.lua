-- ----------------------------------------------------------------
-- Utility
-- ----------------------------------------------------------------

add_requires("volk")

rule("utility_config")
    on_load(function (target)
        target:set("kind", "binary")
        target:set("group", "utility")
    end)
rule_end()

-- ----------------------------------------------------------------
-- Utility Target: Find the current GPU's Warp Size
-- ----------------------------------------------------------------

target("query-warpsize") do
    add_rules("utility_config", "vulkan_config", "run_on_android")
    add_files({
      "query_warpsize.cpp",
    })
    add_packages("volk")
end

-- ----------------------------------------------------------------
-- Utility Target: Query the current CPU Information
-- ----------------------------------------------------------------

if is_plat("linux") or is_plat("android") then
    target("query-cpuinfo") do
        add_rules("utility_config", "run_on_android")
        add_files({
          "query_cpuinfo.cpp",
        })
    end
end

-- ----------------------------------------------------------------
-- Utility Target: try pinning threads to all cores and verify if it works
-- ----------------------------------------------------------------

target("test-affinity") do
    add_rules("utility_config", "run_on_android")
    add_files({
      "test_affinity.cpp",
    })
end

-- ----------------------------------------------------------------
-- Utility Target: Query the cache line size of the current CPU
-- ----------------------------------------------------------------

target("query-cacheline") do
    add_rules("utility_config", "run_on_android")
    add_files({
      "query_cacheline.cpp",
    })
end
