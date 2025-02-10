target("auto-pipe")
    set_kind("binary")
    add_files("main.cpp")

    add_packages("spdlog")
    add_packages("cli11")

    if is_plat("android") then
      on_run(run_on_android)
    end