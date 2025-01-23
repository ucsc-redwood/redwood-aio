add_rules("mode.debug", "mode.release")

set_languages("c++20")
set_warnings("allextra")


-- local has_cuda = true

-- if has_cuda then
--     os.exec("echo '======================cuda======================'")
-- end

includes("android.lua")

includes("pipe")
includes("builtin-apps")

includes("play/usm")
includes("play/safe-pipe")
includes("play/test-tree")

includes("builtin-apps/common/vulkan")


includes("benchmarks")
