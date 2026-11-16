# Copyright (c) 2026 Minmin Gong
#

add_executable(AIHoloImagerDelight
    Main.cpp
)

target_link_libraries(AIHoloImagerDelight
    PRIVATE
        cxxopts
        AIHoloImagerLib
)

set_target_properties(AIHoloImagerDelight PROPERTIES FOLDER "DelightApp")
