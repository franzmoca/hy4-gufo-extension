#ifndef HY4FLASH_RUNNER_HPP_
#define HY4FLASH_RUNNER_HPP_

#include <memory>
#include <string>

#include <gufo/ext/text.hpp>

namespace gufo::models::hy4_flash {

/// `TextProvider::create` for `gufo serve`.
std::shared_ptr<ext::TextModelRunner> CreateHy4TextRunner(const ext::TextRunnerOptions& options,
                                                          std::string* error);

}  // namespace gufo::models::hy4_flash

#endif  // HY4FLASH_RUNNER_HPP_
