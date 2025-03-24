-- Copyright (c) 2025 Yanwen Xu (yxu83@ucsc.edu). MIT License.

target("try-pipe-tree-cu")
do
	add_rules("pipe_config", "common_flags")

	add_headerfiles({
		"task.hpp",
		"run_stages.hpp",
		"../templates.hpp",
	})

	add_files({
		"main.cu",
		"task.cpp",
	})

	add_deps("builtin-apps", "builtin-apps-cuda")
	add_cugencodes("native")
end
