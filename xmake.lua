add_rules("mode.debug", "mode.release")

set_languages("c++20")
set_warnings("allextra")


add_requires("benchmark")
add_requires("cli11")

add_requires("concurrentqueue")

if not is_plat("android") then
    add_requires("openmp")
end

add_requires("spdlog")
add_requires("glm")


-- local has_cuda = true

-- if has_cuda then
--     os.exec("echo '======================cuda======================'")
-- end

includes("android.lua")

includes("pipe")
includes("builtin-apps")

-- includes("play/safe-pipe")
-- includes("play/usm")
-- includes("play/test-tree")
includes("play/threads")
includes("play/cuda-pipe")
includes("play/cuda-graph")
includes("play/test-usm")
includes("play/test-cuda-tree")
includes("play/test-tree-vk")
includes("play/vk-radix-sort")
includes("play/vk-prefix-sum")

includes("utility")

includes("builtin-apps/common/vulkan")


includes("benchmarks")
