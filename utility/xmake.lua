add_requires("volk")


rule("utility_config")
    on_load(function (target)
        target:set("kind", "binary")
        target:set("group", "utility")
    end)
rule_end()


-- ----------------------------------------------------------------
-- Utility Targets
-- ----------------------------------------------------------------

target("query-warpsize")
    add_rules("utility_config", "vulkan_config", "run_on_android")
    add_files({
      "query_warpsize.cpp",
    })
    add_packages("volk")
target_end()

if is_plat("linux") or is_plat("android") then
target("query-cpuinfo")
    add_rules("utility_config", "run_on_android")
    add_files({
      "query_cpuinfo.cpp",
    })
target_end()
end

target("test-affinity")
    add_rules("utility_config", "run_on_android")
    add_files({
      "test_affinity.cpp",
    })
target_end()

target("query-cacheline")
    add_rules("utility_config", "run_on_android")
    add_files({
      "query_cacheline.cpp",
    })
target_end()
