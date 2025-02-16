-- ------------------------------------------------------------
-- OMP benchmarks
-- ------------------------------------------------------------

target("bm-cifar-dense-omp")
    add_rules("benchmark_config", "common_flags", "run_on_android")
    add_files("omp.cpp")
    add_deps("builtin-apps")
target_end()


-- ------------------------------------------------------------
-- VK benchmarks
-- ------------------------------------------------------------

target("bm-cifar-dense-vk")
    add_rules("benchmark_config", "common_flags", "vulkan_config", "run_on_android")
    add_files("vk.cpp")
    add_deps("builtin-apps", "builtin-apps-vulkan")
target_end()


-- ------------------------------------------------------------
-- CUDA benchmarks
-- ------------------------------------------------------------

-- if not is_plat("android") then
if has_config("cuda") then
target("bm-cifar-dense-cu")
    add_rules("benchmark_config")
    add_files("cuda.cu")
    add_deps("builtin-apps", "builtin-apps-cuda")
    add_cugencodes("native")
target_end()

end 