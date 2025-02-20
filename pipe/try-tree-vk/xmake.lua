-- Copyright (c) 2025 Yanwen Xu (yxu83@ucsc.edu). MIT License.

target("try-pipe-tree-vk")
do
	add_rules("pipe_config", "common_flags", "vulkan_config", "run_on_android")

	add_files({
		"main.cpp",
	})

	add_deps("builtin-apps-vulkan", "builtin-apps")
end
