#include "runtime/hy4_engram.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <system_error>
#include <unordered_map>
#include <utility>

namespace gufo::models::hy4_flash {
namespace {

constexpr std::size_t kPage = 4096;
// Enough reads in flight to keep an NVMe drive's queues full.
constexpr std::size_t kWorkers = 128;
// Jobs a reader takes at once from a large gather.
constexpr std::size_t kReadBatch = 8;
constexpr std::size_t kCacheBytes = 128 * 1024 * 1024;

}  // namespace

class EngramReader::Gather {
public:
  std::size_t pending = 0;
  bool failed = false;
  // Repeated rows: copied from the first occurrence once the reads finish.
  std::vector<std::pair<const std::uint8_t*, std::uint8_t*>> copies;
};

EngramReader::~EngramReader() {
  {
    const std::lock_guard lock(mutex_);
    stop_ = true;
  }
  wake_.notify_all();
  for (auto& worker : workers_)
    worker.join();
  if (file_ != platform::kInvalidFile)
    platform::CloseFile(file_);
}

std::unique_ptr<EngramReader> EngramReader::Open(platform::PlatformFile file,
                                                 std::uint64_t file_offset,
                                                 std::uint64_t rows,
                                                 std::uint32_t row_bytes,
                                                 std::string* error) {
  std::unique_ptr<EngramReader> r(new EngramReader());
  r->base_offset_ = file_offset;
  r->rows_ = rows;
  r->row_bytes_ = row_bytes;
  r->cache_count_ = std::bit_floor(kCacheBytes / row_bytes);
  r->cache_entries_ = std::make_unique<CacheEntry[]>(r->cache_count_);
  r->cache_rows_.resize(r->cache_count_ * row_bytes);
  r->file_ = platform::ReopenRandomReadFile(file, r->direct_);
  if (!platform::FileValid(r->file_)) {
    r->direct_ = false;
    r->file_ = platform::DuplicateFile(file);
  }
  if (r->file_ == platform::kInvalidFile) {
    if (error)
      *error = "hy4: cannot open the Engram table for reading";
    return nullptr;
  }
  for (std::size_t i = 0; i < kWorkers; ++i)
    r->workers_.emplace_back([raw = r.get()] { raw->Worker(); });
  return r;
}

bool EngramReader::ReadCached(std::uint64_t row, std::uint8_t* dst) {
  const std::size_t slot =
      (row * 0x9e3779b97f4a7c15ULL >> 20) & (cache_count_ - 1);
  CacheEntry& entry = cache_entries_[slot];
  if (entry.busy.test_and_set(std::memory_order_acquire))
    return false;
  const bool hit = entry.valid && entry.row == row;
  if (hit)
    std::memcpy(dst, cache_rows_.data() + slot * row_bytes_, row_bytes_);
  entry.busy.clear(std::memory_order_release);
  return hit;
}

bool EngramReader::ReadOne(std::uint64_t row, std::uint8_t* dst,
                           std::vector<std::uint8_t>& buf) {
  if (row >= rows_)
    return false;
  if (ReadCached(row, dst))
    return true;
  const std::uint64_t offset = base_offset_ + row * row_bytes_;
  const std::uint64_t begin = direct_ ? offset & ~(kPage - 1) : offset;
  const std::uint64_t end =
      direct_ ? (offset + row_bytes_ + kPage - 1) & ~(kPage - 1)
              : offset + row_bytes_;
  const std::size_t length = end - begin;
  if (buf.size() < length + kPage)
    buf.resize(length + kPage);
  // Direct I/O needs a page-aligned buffer.
  const auto misalign = reinterpret_cast<std::uintptr_t>(buf.data()) % kPage;
  std::uint8_t* base =
      misalign == 0 ? buf.data() : buf.data() + (kPage - misalign);
  // The aligned window may run past the end of the file: only the row's own
  // bytes have to arrive.
  const std::size_t needed = (offset - begin) + row_bytes_;
  std::size_t got = 0;
  while (got < needed) {
    const std::size_t n =
        platform::ReadFileAt(file_, base + got, length - got, begin + got);
    if (n == 0)
      return false;
    got += n;
  }
  const std::uint8_t* src = base + (offset - begin);
  std::memcpy(dst, src, row_bytes_);
  // A busy slot is skipped: readers never wait for another row.
  const std::size_t slot =
      (row * 0x9e3779b97f4a7c15ULL >> 20) & (cache_count_ - 1);
  CacheEntry& entry = cache_entries_[slot];
  if (!entry.busy.test_and_set(std::memory_order_acquire)) {
    std::memcpy(cache_rows_.data() + slot * row_bytes_, src, row_bytes_);
    entry.row = row;
    entry.valid = true;
    entry.busy.clear(std::memory_order_release);
  }
  return true;
}

void EngramReader::Worker() {
  std::vector<std::uint8_t> buf;
  std::unique_lock lock(mutex_);
  for (;;) {
    wake_.wait(lock, [&] { return stop_ || !jobs_.empty(); });
    if (stop_)
      return;
    Job batch[kReadBatch];
    const std::size_t count = std::min(
        jobs_.size() >= kWorkers * kReadBatch ? kReadBatch : 1, jobs_.size());
    for (std::size_t i = 0; i < count; ++i) {
      batch[i] = jobs_.front();
      jobs_.pop_front();
    }
    lock.unlock();
    bool ok[kReadBatch];
    for (std::size_t i = 0; i < count; ++i)
      ok[i] = ReadOne(batch[i].row, batch[i].dst, buf);
    lock.lock();
    bool finished = false;
    for (std::size_t i = 0; i < count; ++i) {
      Gather& g = *batch[i].gather;
      g.failed = g.failed || !ok[i];
      finished = --g.pending == 0 || finished;
    }
    if (finished)
      done_.notify_all();
  }
}

std::shared_ptr<EngramReader::Gather> EngramReader::Start(
    std::span<const std::uint64_t> rows, std::uint8_t* out, bool urgent) {
  auto gather = std::make_shared<Gather>();
  std::unordered_map<std::uint64_t, std::uint8_t*> first;
  first.reserve(rows.size());
  std::vector<Job> jobs;
  jobs.reserve(rows.size());
  for (std::size_t i = 0; i < rows.size(); ++i) {
    std::uint8_t* dst = out + i * row_bytes_;
    const auto [it, fresh] = first.try_emplace(rows[i], dst);
    if (fresh) {
      jobs.push_back({rows[i], dst, gather.get()});
    } else {
      gather->copies.emplace_back(it->second, dst);
    }
  }
  // Rows in file order, and cached rows without waking a reader.
  std::erase_if(jobs,
                [&](const Job& job) { return ReadCached(job.row, job.dst); });
  std::ranges::sort(jobs, {}, &Job::row);
  {
    const std::lock_guard lock(mutex_);
    gather->pending = jobs.size();
    jobs_.insert(urgent ? jobs_.begin() : jobs_.end(), jobs.begin(),
                 jobs.end());
  }
  if (jobs.size() >= kWorkers) {
    wake_.notify_all();
  } else {
    for (std::size_t i = 0; i < jobs.size(); ++i)
      wake_.notify_one();
  }
  return gather;
}

bool EngramReader::Wait(Gather& gather) {
  {
    std::unique_lock lock(mutex_);
    done_.wait(lock, [&] { return gather.pending == 0; });
  }
  if (gather.failed)
    return false;
  for (const auto& [src, dst] : gather.copies)
    std::memcpy(dst, src, row_bytes_);
  return true;
}

}  // namespace gufo::models::hy4_flash
