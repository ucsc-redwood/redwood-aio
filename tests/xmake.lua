-- Copyright (c) 2025 Yanwen Xu (yxu83@ucsc.edu). MIT License.

-- ----------------------------------------------------------------
-- Common test configuration
-- ----------------------------------------------------------------

-- currently using:
-- - gtest-v1.15.2:

add_requires("gtest")

rule("test_config")
on_load(function(target)
	target:set("kind", "binary")
	target:set("group", "test")
	target:add("includedirs", "$(projectdir)/builtin-apps/")
	target:add("includedirs", "$(projectdir)")
	target:add("packages", "gtest")
end)
rule_end()

-- ----------------------------------------------------------------
-- Test targets
-- ----------------------------------------------------------------

-- --- OpenMP Tree ---

target("test-omp-tree")
do
    add_rules("test_config", "common_flags", "run_on_android")
	add_files({
		"test_omp_tree.cpp",
	})
	add_deps("builtin-apps")
end

-- --- Vulkan Tree ---

target("test-vk-tree")
do
    add_rules("test_config", "common_flags", "vulkan_config", "run_on_android")
	add_files({
		"test_vk_tree.cpp",
	})
	add_deps("builtin-apps", "builtin-apps-vulkan")
end

