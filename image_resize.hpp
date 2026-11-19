// Antialiased bicubic RGB8 resize, copied from gemma4-gufo-extension
// (vision/resize.hpp), which took it from the Gufo fork's feat/gemma
// src/core/image.cpp (83b764b4).
#ifndef HY4FLASH_IMAGE_RESIZE_HPP_
#define HY4FLASH_IMAGE_RESIZE_HPP_

#include <cstdint>
#include <gufo/ext/image.hpp>

namespace gufo::models::hy4_flash {

/// Antialiased bicubic RGB8 resize, bit-exact with PyTorch/torchvision uint8
/// `resize(..., BICUBIC, antialias=True)` (separable, horizontal first).
[[nodiscard]] core::Image ResizeBicubic(const core::Image& image,
                                        std::uint32_t width,
                                        std::uint32_t height);

}  // namespace gufo::models::hy4_flash

#endif  // HY4FLASH_IMAGE_RESIZE_HPP_
