// Copyright (c) 2024-2026 Minmin Gong
//

#include "AIHoloImager/AIHoloImager.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <format>
#include <iostream>
#include <stdexcept>
#include <vector>

#ifdef AIHI_ENABLE_D3D12
    #include "Base/MiniWindows.hpp"

    #include <directx/d3d12.h>
#endif
#ifdef AIHI_ENABLE_VULKAN
    #include <volk.h>
#endif

#include "AIHoloImager/Texture.hpp"
#include "AIHoloImagerInternal.hpp"
#include "Base/ErrorHandling.hpp"
#include "Delighter/Delighter.hpp"
#include "Gpu/GpuSystem.hpp"
#include "MaskGen/MaskGenerator.hpp"
#include "MeshGen/MeshGenerator.hpp"
#include "MeshOpt/MeshOptimizer.hpp"
#include "Python/PythonSystem.hpp"
#include "SfM/StructureFromMotion.hpp"
#include "Util/ExifMetadata.hpp"
#include "Util/PerfProfiler.hpp"

namespace AIHoloImager
{
    namespace
    {
        bool IsImageFile(const std::filesystem::path& path)
        {
            static constexpr std::string_view ImageExtensions[] = {".jpg", ".jpeg", ".png", ".bmp", ".tga"};

            std::string ext = path.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), [](char c) { return static_cast<char>(std::tolower(c)); });

            return std::ranges::find(ImageExtensions, std::string_view(ext)) != std::end(ImageExtensions);
        }

        std::vector<std::filesystem::path> RetrieveImagePaths(const std::filesystem::path& input_path)
        {
            std::vector<std::filesystem::path> ret;

            if (std::filesystem::is_directory(input_path))
            {
                for (const auto& dir_entry : std::filesystem::directory_iterator{input_path})
                {
                    if (dir_entry.is_regular_file() && IsImageFile(dir_entry.path()))
                    {
                        ret.push_back(dir_entry.path());
                    }
                }

                std::ranges::sort(ret);
            }
            else if (std::filesystem::is_regular_file(input_path))
            {
                ret.push_back(input_path);
            }

            return ret;
        }

        // Keeps the PNG files lossless, and converts the rest into JPEG.
        std::filesystem::path DelightedImagePath(const std::filesystem::path& output_dir, const std::filesystem::path& input_path)
        {
            std::string ext = input_path.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), [](char c) { return static_cast<char>(std::tolower(c)); });

            return (output_dir / input_path.stem()).replace_extension(ext == ".png" ? ".png" : ".jpg");
        }

        // Bilinear resize of an 8-bit RGBA image.
        Texture ResizeImage(const Texture& image, uint32_t width, uint32_t height)
        {
            constexpr uint32_t NumChannels = 4;

            const uint32_t src_width = image.Width();
            const uint32_t src_height = image.Height();
            const std::byte* src = image.Data();

            Texture ret(width, height, image.Format());
            std::byte* dst = ret.Data();

            for (uint32_t y = 0; y < height; ++y)
            {
                const float sy = ((static_cast<float>(y) + 0.5f) * static_cast<float>(src_height) / static_cast<float>(height)) - 0.5f;
                const uint32_t sy0 = static_cast<uint32_t>(std::min(std::max(std::floor(sy), 0.0f), static_cast<float>(src_height - 1)));
                const uint32_t sy1 = std::min(sy0 + 1, src_height - 1);
                const float fy = std::min(std::max(sy - static_cast<float>(sy0), 0.0f), 1.0f);

                for (uint32_t x = 0; x < width; ++x)
                {
                    const float sx = ((static_cast<float>(x) + 0.5f) * static_cast<float>(src_width) / static_cast<float>(width)) - 0.5f;
                    const uint32_t sx0 = static_cast<uint32_t>(std::min(std::max(std::floor(sx), 0.0f), static_cast<float>(src_width - 1)));
                    const uint32_t sx1 = std::min(sx0 + 1, src_width - 1);
                    const float fx = std::min(std::max(sx - static_cast<float>(sx0), 0.0f), 1.0f);

                    const std::byte* p00 = src + (static_cast<size_t>(sy0) * src_width + sx0) * NumChannels;
                    const std::byte* p01 = src + (static_cast<size_t>(sy0) * src_width + sx1) * NumChannels;
                    const std::byte* p10 = src + (static_cast<size_t>(sy1) * src_width + sx0) * NumChannels;
                    const std::byte* p11 = src + (static_cast<size_t>(sy1) * src_width + sx1) * NumChannels;

                    std::byte* dst_pixel = dst + (static_cast<size_t>(y) * width + x) * NumChannels;
                    for (uint32_t c = 0; c < NumChannels; ++c)
                    {
                        const float top = std::lerp(
                            static_cast<float>(std::to_integer<uint8_t>(p00[c])), static_cast<float>(std::to_integer<uint8_t>(p01[c])), fx);
                        const float bottom = std::lerp(
                            static_cast<float>(std::to_integer<uint8_t>(p10[c])), static_cast<float>(std::to_integer<uint8_t>(p11[c])), fx);

                        dst_pixel[c] = static_cast<std::byte>(std::min(std::max(std::lerp(top, bottom, fy) + 0.5f, 0.0f), 255.0f));
                    }
                }
            }

            return ret;
        }
    } // namespace

    AIHoloImagerInternal::~AIHoloImagerInternal() noexcept = default;

    class AIHoloImager::Impl : public AIHoloImagerInternal
    {
    public:
        Impl(DeviceType device, Api api, const std::filesystem::path& tmp_dir, bool gpu_debug)
            : exe_dir_(RetrieveExeDir()), tmp_dir_(tmp_dir), gpu_system_(Convert(api), ConfirmDevice, true, gpu_debug),
              python_system_(GetDeviceName(device), exe_dir_), tensor_converter_(gpu_system_, python_system_.ComputeDeviceName())
        {
        }

        static std::filesystem::path RetrieveExeDir()
        {
            char exe_path[MAX_PATH];
            GetModuleFileNameA(nullptr, exe_path, sizeof(exe_path));
            return std::filesystem::path(exe_path).parent_path();
        }

        static GpuSystem::Api Convert(AIHoloImager::Api api)
        {
            switch (api)
            {
#ifdef AIHI_ENABLE_D3D12
            case AIHoloImager::Api::D3D12:
                return GpuSystem::Api::D3D12;
#endif
#ifdef AIHI_ENABLE_VULKAN
            case AIHoloImager::Api::Vulkan:
                return GpuSystem::Api::Vulkan;
#endif

            case AIHoloImager::Api::Auto:
                return GpuSystem::Api::Auto;

            default:
                return GpuSystem::Api::Auto;
            }
        }

        static bool ConfirmDevice(GpuSystem::Api api, void* device)
        {
            switch (api)
            {
#ifdef AIHI_ENABLE_D3D12
            case GpuSystem::Api::D3D12:
            {
                D3D12_FEATURE_DATA_D3D12_OPTIONS1 options1{};
                reinterpret_cast<ID3D12Device*>(device)->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &options1, sizeof(options1));
                if (!options1.WaveOps || (options1.WaveLaneCountMin < 16))
                {
                    return false;
                }

                return true;
            }
#endif
#ifdef AIHI_ENABLE_VULKAN
            case GpuSystem::Api::Vulkan:
            {
                VkPhysicalDeviceSubgroupProperties subgroup_properties{
                    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
                };

                VkPhysicalDeviceProperties2 device_properties{
                    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                    .pNext = &subgroup_properties,
                };

                vkGetPhysicalDeviceProperties2(reinterpret_cast<VkPhysicalDevice>(device), &device_properties);

                if (device_properties.properties.limits.maxColorAttachments < 3)
                {
                    return false;
                }

                if (((subgroup_properties.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) == 0) ||
                    (subgroup_properties.subgroupSize < 16))
                {
                    return false;
                }

                return true;
            }
#endif

            default:
                Unreachable("Invalid API");
            }
        }

        static std::string_view GetDeviceName(DeviceType device)
        {
            switch (device)
            {
            case DeviceType::Cuda:
                return "cuda";

            default:
                return "cpu";
            }
        }

        const std::filesystem::path& ExeDir() noexcept override
        {
            return exe_dir_;
        }

        const std::filesystem::path& TmpDir() noexcept override
        {
            return tmp_dir_;
        }

        GpuSystem& GpuSystemInstance() noexcept override
        {
            return gpu_system_;
        }

        PythonSystem& PythonSystemInstance() noexcept override
        {
            return python_system_;
        }

        PerfProfiler& PerfProfilerInstance() noexcept override
        {
            return profiler_;
        }

        TensorConverter& TensorConverterInstance() noexcept override
        {
            return tensor_converter_;
        }

        Mesh Generate(const std::filesystem::path& input_path, uint32_t texture_size, bool no_delight, FeatureMatcher matcher)
        {
            StructureFromMotion::Result sfm_result;
            {
                StructureFromMotion sfm(*this, matcher);
                sfm_result = sfm.Process(input_path, true);
            }

            {
                MaskGenerator mask_gen(*this);
                for (auto& projection : sfm_result.projections)
                {
                    mask_gen.Generate(projection);
                }
            }
            if (!no_delight)
            {
                Delighter delighter(*this);
                for (size_t i = 0; i < sfm_result.projections.size(); ++i)
                {
                    auto& projection = sfm_result.projections[i];

                    delighter.Process(projection);

#ifdef AIHI_KEEP_INTERMEDIATES
                    const auto output_dir = tmp_dir_ / "Delighter";
                    std::filesystem::create_directories(output_dir);

                    const std::string name = projection.image_path.empty() ? std::to_string(i) : projection.image_path.stem().string();
                    SaveTexture(delighter.ReadBackImage(projection), output_dir / std::format("Delighted_{}.jpg", name));
#endif
                }
            }

            MeshGenerator::Result mg_result;
            {
                MeshGenerator mesh_gen(*this);
                mg_result = mesh_gen.Generate(sfm_result);
            }

            GpuMesh result_mesh;
            {
                MeshOptimizer mesh_opt(*this);
                result_mesh = mesh_opt.Optimize(sfm_result, mg_result, texture_size);
            }

            profiler_.Output(std::cout);

            return ToMesh(gpu_system_, result_mesh);
        }

        void Delight(const std::filesystem::path& input_path, const std::filesystem::path& output_dir, uint32_t max_size)
        {
            const std::vector<std::filesystem::path> image_paths = RetrieveImagePaths(input_path);
            if (image_paths.empty())
            {
                throw std::runtime_error(std::format("No image found in {}", input_path.string()));
            }

            if (std::filesystem::weakly_canonical(input_path) == std::filesystem::weakly_canonical(output_dir))
            {
                throw std::runtime_error("The output directory MUST be different from the input one");
            }
            std::filesystem::create_directories(output_dir);

            Delighter delighter(*this);

            for (size_t i = 0; i < image_paths.size(); ++i)
            {
                const std::filesystem::path& image_path = image_paths[i];
                std::cout << std::format("[{}/{}] Delighting {}...\n", i + 1, image_paths.size(), image_path.filename().string());

                PerfRegion delight_perf(profiler_, std::format("Delight {}", image_path.filename().string()));

                Texture image_cpu = LoadTexture(image_path);
                if (!image_cpu.Valid())
                {
                    std::cerr << std::format("WARNING: Unable to load {}, skipped.\n", image_path.string());
                    continue;
                }

                const uint32_t full_width = image_cpu.Width();
                const uint32_t full_height = image_cpu.Height();

                Texture delight_input;
                const bool resized = (max_size > 0) && (std::max(full_width, full_height) > max_size);
                if (resized)
                {
                    const float scale = static_cast<float>(max_size) / static_cast<float>(std::max(full_width, full_height));
                    const uint32_t width = std::max(1U, static_cast<uint32_t>(std::lround(static_cast<float>(full_width) * scale)));
                    const uint32_t height = std::max(1U, static_cast<uint32_t>(std::lround(static_cast<float>(full_height) * scale)));
                    delight_input = ResizeImage(image_cpu, width, height);
                }
                else
                {
                    delight_input = std::move(image_cpu);
                }

                GpuTexture2D gpu_image(gpu_system_, delight_input.Width(), delight_input.Height(), 1, GpuFormat::RGBA8_UNorm_SRGB,
                    GpuResourceFlag::ShaderResource | GpuResourceFlag::Shareable, std::format("delight_input_{}", i));
                {
                    auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Copy);
                    cmd_list.Upload(gpu_image, 0, delight_input.Data(), delight_input.DataSize());
                    gpu_system_.Execute(std::move(cmd_list));
                }

                AIHoloImagerInternal::ProjectionDesc projection;
                projection.image = std::make_shared<GpuTexture2D>(std::move(gpu_image));
                delighter.Process(projection);

                Texture delighted_image = delighter.ReadBackImage(projection);
                if (resized)
                {
                    delighted_image = ResizeImage(delighted_image, full_width, full_height);
                }

                const std::filesystem::path output_path = DelightedImagePath(output_dir, image_path);
                SaveTexture(delighted_image, output_path);

                const auto exif = LoadExifMetadata(image_path);
                if (exif && !SaveExifMetadata(output_path, *exif))
                {
                    std::cerr << std::format("WARNING: Unable to copy the Exif metadata to {}.\n", output_path.string());
                }
            }

            profiler_.Output(std::cout);
        }

    private:
        std::filesystem::path exe_dir_;
        std::filesystem::path tmp_dir_;

        GpuSystem gpu_system_;
        PythonSystem python_system_;
        PerfProfiler profiler_;

        TensorConverter tensor_converter_;
    };

    AIHoloImager::AIHoloImager(DeviceType device, Api api, const std::filesystem::path& tmp_dir, bool gpu_debug)
        : impl_(std::make_unique<Impl>(device, api, tmp_dir, gpu_debug))
    {
    }
    AIHoloImager::AIHoloImager(AIHoloImager&& rhs) noexcept = default;
    AIHoloImager::~AIHoloImager() noexcept = default;

    AIHoloImager& AIHoloImager::operator=(AIHoloImager&& rhs) noexcept = default;

    Mesh AIHoloImager::Generate(const std::filesystem::path& input_path, uint32_t texture_size, bool no_delight, FeatureMatcher matcher)
    {
        return impl_->Generate(input_path, texture_size, no_delight, matcher);
    }

    void AIHoloImager::Delight(const std::filesystem::path& input_path, const std::filesystem::path& output_dir, uint32_t max_size)
    {
        impl_->Delight(input_path, output_dir, max_size);
    }
} // namespace AIHoloImager
