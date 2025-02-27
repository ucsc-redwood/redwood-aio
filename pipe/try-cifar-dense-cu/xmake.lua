-- Copyright (c) 2025 Yanwen Xu (yxu83@ucsc.edu). MIT License.

target("try-pipe-cifar-dense-cu")
do
	add_rules("pipe_config", "common_flags")

	add_headerfiles({
		"run_stages.hpp",
		"task.hpp",
	})

	add_files({
		"main.cu",
		"task.cpp",
	})

	add_deps("builtin-apps", "builtin-apps-cuda")

    add_cugencodes("native")
end
