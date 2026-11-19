// Encodes a binary PPM (P6, sides multiples of 32) with the Hy4 Flash vision
// tower and writes the embedding rows as raw FP32 [rows][3072].
//
//   hy4_vision_probe MMPROJ.gguf IMAGE.ppm OUT.f32
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "runtime/hy4_vision.hpp"

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: %s MMPROJ.gguf IMAGE.ppm OUT.f32\n", argv[0]);
    return 2;
  }
  std::ifstream file(argv[2], std::ios::binary);
  std::string magic;
  std::uint32_t width = 0, height = 0, max = 0;
  file >> magic >> width >> height >> max;
  file.get();
  if (magic != "P6" || max != 255) {
    std::fprintf(stderr, "not an 8-bit P6 PPM\n");
    return 1;
  }
  std::vector<std::uint8_t> rgb(std::size_t{width} * height * 3);
  file.read(reinterpret_cast<char*>(rgb.data()),
            static_cast<std::streamsize>(rgb.size()));

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
