# Copyright (c) 2026 Minmin Gong
#

if(is_ci_env)
    add_custom_target(DeployLoMaModels)
else()
    set(url "https://github.com/davnords/storage/releases/download/loma/")
    set(local_dir "${CMAKE_CURRENT_SOURCE_DIR}/LoMaModels/")
    file(MAKE_DIRECTORY ${local_dir})

    set(deployed_files )
    # LoMa-B already carries the frozen DINOv2 and the DaD detector weights, so no other
    # checkpoint has to be downloaded.
    foreach(file "loma_B.pt")
        DownloadFile("${url}${file}" "${local_dir}${file}")

        set(output_file ${aihi_output_dir}/Models/LoMa/${file})
        set(src_file LoMaModels/${file})
        add_custom_command(OUTPUT ${output_file}
            COMMAND ${CMAKE_COMMAND} -E create_hardlink ${src_file} ${output_file}
            COMMENT "Deploying ${src_file}..."
            DEPENDS ${src_file}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            VERBATIM COMMAND_EXPAND_LISTS
        )

        list(APPEND deployed_files ${output_file})
    endforeach()

    add_custom_target(DeployLoMaModels DEPENDS ${deployed_files})
endif()

set_target_properties(DeployLoMaModels PROPERTIES FOLDER "External")
