#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <gufo/ext/platform/mapped_file.hpp>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace gufo::models::hy4_flash {

/// Reads Engram table rows from the GGUF shard with direct I/O on a pool of
/// resident readers, keeping a bounded cache of rows. The table stays out of
/// the page cache, and a prompt's random rows are read many at a time.
class EngramReader {
public:
  /// One gather: its rows arrive in `out` once Wait returns true.
  class Gather;

  ~EngramReader();
  EngramReader(const EngramReader&) = delete;
  EngramReader& operator=(const EngramReader&) = delete;

  /// Borrows `file` (the mapped shard's handle) and keeps its own handle.
  [[nodiscard]] static std::unique_ptr<EngramReader> Open(
      platform::PlatformFile file, std::uint64_t file_offset,
      std::uint64_t rows, std::uint32_t row_bytes, std::string* error);

  /// Starts reading rows `rows` (table row indices) into `out`, row i at
  /// out + i * row_bytes. Gathers are served in the order they start, an
  /// `urgent` one before every queued read; `out` must stay alive until Wait.
  [[nodiscard]] std::shared_ptr<Gather> Start(
      std::span<const std::uint64_t> rows, std::uint8_t* out,
      bool urgent = false);
  /// Queues low-priority background reads of `rows` into the row cache so a
  /// subsequent Start() hits RAM without blocking on NVMe I/O.
  void Prefetch(std::span<const std::uint64_t> rows);
  /// Waits for `gather`; false when a read failed.
  [[nodiscard]] bool Wait(Gather& gather);

private:
  struct Job {
    std::uint64_t row;
    std::uint8_t* dst;
    Gather* gather;
  };
  struct CacheEntry {
    std::atomic_flag busy = ATOMIC_FLAG_INIT;
    std::uint64_t row = 0;
    bool valid = false;
  };

  EngramReader() = default;
  bool ReadCached(std::uint64_t row, std::uint8_t* dst);
  bool ReadOne(std::uint64_t row, std::uint8_t* dst,
               std::vector<std::uint8_t>& buf);
  void Worker();

  platform::PlatformFile file_{platform::kInvalidFile};
  bool direct_ = false;
  std::uint64_t base_offset_ = 0;
  std::uint64_t rows_ = 0;
  std::size_t row_bytes_ = 0;

  std::unique_ptr<CacheEntry[]> cache_entries_;
  std::vector<std::uint8_t> cache_rows_;
  std::size_t cache_count_ = 0;

  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable wake_;
  std::condition_variable done_;
  std::deque<Job> jobs_;
  bool stop_ = false;
};

}  // namespace gufo::models::hy4_flash
