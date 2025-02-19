-- Copyright (c) 2025 Yanwen Xu (yxu83@ucsc.edu). MIT License.

-- ----------------------------------------------------------------
-- Vulkan
-- ----------------------------------------------------------------

target("bm-vk-prefix-sum")
do
	set_kind("binary")
	set_group("benchmarks")

	add_includedirs("$(projectdir)/builtin-apps/")
	add_files("bm_vk_prefix_sum.cpp")

	add_deps("builtin-apps")

	add_packages("benchmark")
	-- add_packages("cli11")
	add_packages("spdlog")
	add_packages("glm")
	add_packages("vulkan-hpp", "vulkan-memory-allocator")

	-- Add openmp support
	if is_plat("android") then
		add_cxxflags("-fopenmp -static-openmp")
		add_ldflags("-fopenmp -static-openmp")
	else
		add_packages("openmp")
	end

	add_rules("run_on_android")
end
