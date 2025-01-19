add_rules("mode.debug", "mode.release")

set_languages("c++20")
set_warnings("allextra")
if not is_plat("android") then
    set_toolchains("clang")
end

add_repositories("local-repo ../local-repo")

add_requires("builtin-apps")

target("demo")
    set_kind("binary")
    add_files("src/*.cpp")
    add_packages("builtin-apps")
