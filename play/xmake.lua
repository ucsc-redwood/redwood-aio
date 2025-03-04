-- Copyright (c) 2025 Yanwen Xu (yxu83@ucsc.edu). MIT License.

target("play-tree")
do
	add_rules("common_flags", "run_on_android")

    add_includedirs("$(projectdir)")
    
	add_files({
		"main.cpp",
	})

	add_deps("builtin-apps")
end
