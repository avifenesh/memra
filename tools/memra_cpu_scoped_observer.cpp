// Test-only I/O observer. Production builds do not define this macro or export these hooks.
#define MEMRA_CPU_SCOPED_TEST_OBSERVER
#include "memra_cpu_experts.cpp"

namespace {
std::mutex observer_mutex;
std::condition_variable observer_cv;
int observer_pause_mode = 0;
std::atomic<std::uint64_t> observer_entries {0};
std::atomic<std::uint64_t> primary_reads {0}, alternate_reads {0}, primary_bytes {0}, alternate_bytes {0};
std::atomic<std::uint64_t> completed_primary {0}, completed_alternate {0};
}
extern "C" void memra_cpu_test_before_scoped_read(std::int32_t alternate, std::size_t length) {
    std::unique_lock<std::mutex> lock(observer_mutex);
    observer_entries.fetch_add(1, std::memory_order_relaxed);
    (alternate ? alternate_reads : primary_reads).fetch_add(1,std::memory_order_relaxed);
    (alternate ? alternate_bytes : primary_bytes).fetch_add(length,std::memory_order_relaxed);
    observer_cv.wait(lock, [alternate] { return observer_pause_mode == 0 || (observer_pause_mode == 2 && alternate == 0); });
}
extern "C" void memra_cpu_test_pause_scoped_reads(std::int32_t paused) {
    {
        std::lock_guard<std::mutex> lock(observer_mutex);
        observer_pause_mode = paused;
        if (observer_pause_mode) {
            observer_entries.store(0,std::memory_order_relaxed);
            primary_reads.store(0);alternate_reads.store(0);primary_bytes.store(0);alternate_bytes.store(0);
            completed_primary.store(0);completed_alternate.store(0);
        }
    }
    observer_cv.notify_all();
}
extern "C" std::uint64_t memra_cpu_test_scoped_read_entries() {
    return observer_entries.load(std::memory_order_relaxed);
}
extern "C" int memra_cpu_test_take_annex(const memra_scoped_disk_info_v1 * info,
                                        void * out, std::size_t len) noexcept try {
    if (!info || !out || len != info->len) return EINVAL;
    const CacheKey key {FileKey {InodeKey {info->device,info->inode},info->file_bytes,
                       info->ctime_seconds,info->ctime_nanoseconds},info->offset,len};
    auto bytes = PrefetchAnnex::instance().take(key);
    if (!bytes) return EAGAIN;
    std::memcpy(out,bytes->data,len);
    return 0;
} catch (...) { return EIO; }

extern "C" void memra_cpu_test_scoped_io_counts(std::uint64_t * primary, std::uint64_t * alternate,
                                               std::uint64_t * pbytes, std::uint64_t * abytes) {
    *primary=primary_reads.load();*alternate=alternate_reads.load();
    *pbytes=primary_bytes.load();*abytes=alternate_bytes.load();
}

extern "C" void memra_cpu_test_after_scoped_prefetch_job(std::int32_t alternate) {
    (alternate ? completed_alternate : completed_primary).fetch_add(1);
}
extern "C" void memra_cpu_test_completed_halves(std::uint64_t * primary, std::uint64_t * alternate) {
    *primary=completed_primary.load();*alternate=completed_alternate.load();
}
extern "C" std::int32_t memra_cpu_test_signed_prefetch_inflight() {
    return prefetch_inflight().load();
}
