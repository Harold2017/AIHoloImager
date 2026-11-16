// Copyright (c) 2026 Minmin Gong
//

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <format>
#include <iostream>

#ifndef _DEBUG
    #define CXXOPTS_NO_RTTI
#endif
#include <cxxopts.hpp>

#include "AIHoloImager/AIHoloImager.hpp"

int main(int argc, char* argv[])
{
    cxxopts::Options options("AIHoloImagerDelight", "AIHoloImagerDelight: Remove the lighting from images.");
    // clang-format off
    options.add_options()
        ("H,help", "Produce help message.")
        ("I,input-path", "The directory that contains the input images, or a single image file.", cxxopts::value<std::string>())
        ("O,output-path", "The directory to save the delighted images (\"<input-dir>/Delighted\" by default).", cxxopts::value<std::string>())
        ("D,device", "The computation device for inferencing (cuda or cpu; cuda by default).", cxxopts::value<std::string>())
        ("S,max-size", "The maximum width or height to delight at (2048 by default, 0 to disable). Larger images are scaled down and then back.", cxxopts::value<uint32_t>()->default_value("2048"))
        ("gpu-debug", "Enable GPU system debugging information.")
        ("api", "Select which API to use (auto, d3d12, or vulkan; auto by default).", cxxopts::value<std::string>())
        ("v,version", "Version.");
    // clang-format on

    const auto vm = options.parse(argc, argv);

    if ((argc <= 1) || (vm.count("help") > 0))
    {
        std::cout << std::format("{}\n", options.help());
        return 0;
    }
    if (vm.count("version") > 0)
    {
        std::cout << "AIHoloImagerDelight, Version 0.8.0\n";
        return 0;
    }

    std::filesystem::path input_path;
    if (vm.count("input-path") > 0)
    {
        input_path = vm["input-path"].as<std::string>();
    }
    else
    {
        std::cerr << std::format("ERROR: MUST have a input path\n");
        return 1;
    }

    if (!std::filesystem::exists(input_path))
    {
        std::cerr << std::format("ERROR: COULDN'T find {}\n", input_path.string());
        return 1;
    }

    std::filesystem::path output_path;
    if (vm.count("output-path") > 0)
    {
        output_path = vm["output-path"].as<std::string>();
    }
    else
    {
        std::filesystem::path input_dir = input_path;
        if (std::filesystem::is_regular_file(input_path))
        {
            input_dir = input_path.parent_path();
        }
        output_path = input_dir / "Delighted";
    }

    auto device = AIHoloImager::AIHoloImager::DeviceType::Cuda;
    if (vm.count("device") > 0)
    {
        std::string device_name = vm["device"].as<std::string>();
        std::transform(
            device_name.begin(), device_name.end(), device_name.begin(), [](char c) { return static_cast<char>(std::tolower(c)); });

        if (device_name == "cpu")
        {
            device = AIHoloImager::AIHoloImager::DeviceType::Cpu;
        }
        else if (device_name == "cuda")
        {
            device = AIHoloImager::AIHoloImager::DeviceType::Cuda;
        }
        else
        {
            std::cerr << std::format("ERROR: Unsupported device {}. Use cpu or cuda.\n", device_name);
            return 1;
        }
    }

    const bool gpu_debug = (vm.count("gpu-debug") > 0);
    const uint32_t max_size = vm["max-size"].as<uint32_t>();

    auto api = AIHoloImager::AIHoloImager::Api::Auto;
    if (vm.count("api") > 0)
    {
        std::string api_name = vm["api"].as<std::string>();
        std::transform(api_name.begin(), api_name.end(), api_name.begin(), [](char c) { return static_cast<char>(std::tolower(c)); });

        if (api_name == "d3d12")
        {
            api = AIHoloImager::AIHoloImager::Api::D3D12;
        }
        else if (api_name == "vulkan")
        {
            api = AIHoloImager::AIHoloImager::Api::Vulkan;
        }
        else if (api_name == "auto")
        {
            api = AIHoloImager::AIHoloImager::Api::Auto;
        }
        else
        {
            std::cerr << std::format("ERROR: Unsupported API {}. Use auto, d3d12, or vulkan.\n", api_name);
            return 1;
        }
    }

    auto start = std::chrono::high_resolution_clock::now();

    try
    {
        AIHoloImager::AIHoloImager imager(device, api, output_path / "Tmp", gpu_debug);
        imager.Delight(input_path, output_path, max_size);
    }
    catch (std::exception& ex)
    {
        std::cout << "ERROR: " << ex.what() << '\n';
        return 1;
    }

    auto elapse = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start);

    std::cout << std::format("The delighted images are saved to {}\n", output_path.string());
    std::cout << "Total time: " << elapse.count() << " s\n";

    return 0;
}
