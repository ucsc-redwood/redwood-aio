add_requires("cli11")

add_requires("concurrentqueue")

if not platform("android") then


add_requires("openmp")

target("usm")
    set_kind("binary")
    add_files("main.cu")

    add_packages("concurrentqueue")
    add_packages("cli11")
    add_packages("openmp")
target_end()

end
