-- Copyright (c) 2025 Yanwen Xu (yxu83@ucsc.edu). MIT License.

target("play-safe-tree-vk")
do
	add_rules("common_flags", "vulkan_config", "run_on_android")

    add_includedirs("$(projectdir)")
    
	add_files({
		"main.cpp",
	})

	add_deps("builtin-apps-vulkan", "builtin-apps")
end
