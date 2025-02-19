-- Copyright (c) 2025 Yanwen Xu (yxu83@ucsc.edu). MIT License.

-- ----------------------------------------------------------------
-- Playground: Testing Vulkan Radix Sort Implementation
-- ----------------------------------------------------------------

target("try-sort")
do
	add_rules("vulkan_config", "common_flags", "run_on_android")
	set_kind("binary")
	add_files({
		"main.cpp",
	})

	add_includedirs("$(projectdir)")
	add_deps("builtin-apps-vulkan")
	add_packages("spdlog")
end
