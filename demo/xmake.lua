add_rules("mode.debug", "mode.release")

set_languages("c++20")
set_warnings("allextra")
-- if not is_plat("android") then
--     set_toolchains("clang")
-- end

add_repositories("local-repo ../local-repo")

add_requires("builtin-apps")

if not is_plat("android") then
    add_requires("openmp")
end

includes("../android.lua")

includes("hello-world")
includes("bm")
includes("cuda-pipe")