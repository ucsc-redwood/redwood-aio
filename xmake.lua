add_rules("mode.debug", "mode.release")

set_languages("c++20")
set_warnings("allextra")

-- set_policy("build.cuda.devlink", true)

includes("builtin-apps")
includes("play")
includes("pipe")

includes("play/usm")
includes("builtin-apps/common/vulkan")
