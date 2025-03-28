-- Copyright (c) 2025 Yanwen Xu (yxu83@ucsc.edu). MIT License.

-- Benchmark for the new pipeline

target("bm-new-pipe-cifar-dense-cu")
do
	add_rules("pipe_config", "common_flags")

	add_headerfiles({
		"task.hpp",
		"run_stages.hpp",
		"../templates.hpp",
		"../templates_cu.hpp",
		"generated_code.cuh",
	})

	add_files({
		"bm_main.cu",
		"task.cpp",
	})

	add_deps("builtin-apps", "builtin-apps-cuda")
	add_cugencodes("native")

	add_packages("benchmark")
end

-- Single execution of the pipeline for debugging or testing purposes

target("pipe-cifar-dense-cu")
do
	add_rules("pipe_config", "common_flags")

	add_headerfiles({
		"task.hpp",
		"run_stages.hpp",
		"../templates.hpp",
		"../templates_cu.hpp",
		"generated_code_non_bm.cuh",
	})

	add_files({
		"main.cu",
		"task.cpp",
	})

	add_deps("builtin-apps", "builtin-apps-cuda")
	add_cugencodes("native")

	add_links("nvToolsExt")
end
