-- Copyright (c) 2025 Yanwen Xu (yxu83@ucsc.edu). MIT License.

target("new-pipe-cifar-dense-cu")
do
	add_rules("pipe_config", "common_flags", "run_on_android")

	add_headerfiles({
		"task.hpp",
		"run_stages.hpp",
		"../templates.hpp",
		"../templates_cu.hpp",
		-- "generated_code.hpp",
	})

	add_files({
		"main.cu",
		"task.cpp",
	})

	add_deps("builtin-apps", "builtin-apps-cuda")
	add_cugencodes("native")

	add_packages("benchmark")
end


-- target("bm-new-pipe-cifar-dense-cu")
-- do
-- 	add_rules("pipe_config", "common_flags", "run_on_android")

-- 	add_headerfiles({
-- 		"task.hpp",
-- 		"run_stages.hpp",
-- 		"../templates.hpp",
-- 		"../templates_cu.hpp",
-- 		"generated_code.hpp",
-- 	})

-- 	add_files({
-- 		"bm_main.cu",
-- 		"task.cpp",
-- 	})

-- 	add_deps("builtin-apps", "builtin-apps-cuda")
-- 	add_cugencodes("native")

-- 	add_packages("benchmark")
-- end
