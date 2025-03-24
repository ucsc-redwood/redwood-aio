-- Copyright (c) 2025 Yanwen Xu (yxu83@ucsc.edu). MIT License.

target("pipe-cifar-dense-vk")
do
	add_rules("pipe_config", "common_flags", "vulkan_config", "run_on_android")

	add_headerfiles({
		"run_stages.hpp",
		"task.hpp",
		"generated_codes.hpp",
		"generated-code/*.hpp", -- generated code
	})

	add_files({
		"main.cpp",
		"task.cpp",
		"generated-code/*.cpp", -- generated code
	})

	add_deps("builtin-apps", "builtin-apps-vulkan")
end
