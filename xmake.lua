add_rules("mode.debug", "mode.release")

set_languages("c++20")
set_warnings("allextra")

includes("android.lua")

includes("pipe")
includes("builtin-apps")

includes("play/usm")
includes("play/safe-pipe")
includes("play/test-tree")

includes("builtin-apps/common/vulkan")
