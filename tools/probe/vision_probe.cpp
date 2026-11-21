// Encodes an image with the Hy4 Flash vision tower and writes the embedding
// rows as raw FP32 [rows][3072]. A binary PPM (P6, sides multiples of 32) is
// encoded as is; any other image is decoded and resized as prompts do
// (PrepareImage), and RESIZED.ppm, when given, receives those pixels.
//
//   hy4_vision_probe MMPROJ.gguf IMAGE OUT.f32 [RESIZED.ppm]
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "runtime/hy4_vision.hpp"
#include "vision_prompt.hpp"

int main(int argc, char** argv) {
  if (argc != 4 && argc != 5) {
    std::fprintf(stderr, "usage: %s MMPROJ.gguf IMAGE OUT.f32 [RESIZED.ppm]\n",
                 argv[0]);
    return 2;
  }
  std::ifstream file(argv[2], std::ios::binary);
  const std::vector<std::uint8_t> bytes{std::istreambuf_iterator<char>(file),
                                        {}};
  std::uint32_t width = 0, height = 0;
  std::vector<std::uint8_t> rgb;
  if (bytes.size() > 2 && bytes[0] == 'P' && bytes[1] == '6') {
    std::istringstream header(std::string(bytes.begin(), bytes.end()));
    std::string magic;
    std::uint32_t max = 0;
    header >> magic >> width >> height >> max;
    header.get();
    const auto at = static_cast<std::size_t>(header.tellg());
    const std::size_t size = std::size_t{width} * height * 3;
    if (max != 255 || at + size > bytes.size()) {
      std::fprintf(stderr, "not an 8-bit P6 PPM\n");
      return 1;
    }
    rgb.assign(bytes.begin() + static_cast<std::ptrdiff_t>(at),
               bytes.begin() + static_cast<std::ptrdiff_t>(at + size));
  } else {
    try {
      auto image = gufo::models::hy4_flash::PrepareImage(bytes, false);
      width = image.width;
      height = image.height;
      rgb = std::move(image.pixels);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "%s\n", e.what());
      return 1;
    }
  }
  if (argc == 5) {
    std::ofstream resized(argv[4], std::ios::binary);
    resized << "P6\n" << width << ' ' << height << "\n255\n";
    resized.write(reinterpret_cast<const char*>(rgb.data()),
                  static_cast<std::streamsize>(rgb.size()));
  }

  hipStream_t stream = nullptr;
  (void)hipStreamCreate(&stream);
  std::string error;
  auto encoder = gufo::models::hy4_flash::VisionEncoder::Load(argv[1], 3072,
                                                              stream, &error);
  if (!encoder) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  for (int run = 0; run < 2; ++run) {
    const auto t0 = std::chrono::steady_clock::now();
    auto rows = encoder->Encode(rgb.data(), width, height, &error);
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
    if (!rows) {
      std::fprintf(stderr, "%s\n", error.c_str());
      return 1;
    }
    std::printf("%ux%u: %u rows in %.1f ms\n", width, height, rows->rows(), ms);
    if (run == 1) {
      std::vector<float> host(std::size_t{rows->rows()} * rows->width());
      (void)hipMemcpy(host.data(), rows->data(), host.size() * sizeof(float),
                      hipMemcpyDeviceToHost);
      std::ofstream out(argv[3], std::ios::binary);
      out.write(reinterpret_cast<const char*>(host.data()),
                static_cast<std::streamsize>(host.size() * sizeof(float)));
    }
  }
  return 0;
}
