add_rules("mode.debug", "mode.release")

set_languages("c++20")
set_warnings("allextra")

includes("android.lua")

includes("pipe")
includes("builtin-apps")
-- includes("pipe")

includes("play/usm")
includes("play/vk")
includes("play/vk-pipe")

includes("builtin-apps/common/vulkan")
