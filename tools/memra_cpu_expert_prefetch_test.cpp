// Focused process-level tests for the CPU expert companion's speculative-prefetch accounting
// (memra#586: a mirrored projection is two I/O jobs but one in-flight charge). Including the
// production translation unit keeps the test on the exact IoPool, PrefetchState and
// prefetch_inflight() implementation and lets it read the SIGNED in-flight counter directly,
// without the public stats function's clamp to zero and without a test hook in the stable
// companion ABI (the shm test's pattern, tools/memra_cpu_expert_shm_test.cpp).
//
// The one pread call in the translation unit (pread_exact) is renamed at preprocessing time to
// memra_test_pread, which forwards to the real pread but can HOLD a read at its entry and FAIL a
// read on demand. The hold is the deterministic observation point: a worker that enters a held
// read has finished every job it took before it (bookkeeping runs before the next job is taken),
// so "every alternate half held and the queue empty" means every primary half's bookkeeping is
// complete, and the signed counter read at that moment is the projections' charge, not a race.
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

ssize_t memra_test_pread(int fd, void * buffer, size_t length, off_t offset);
#define pread memra_test_pread
#include "memra_cpu_experts.cpp"
#undef pread

namespace prefetch_test {

constexpr std::size_t kProjectionBytes = 64 * 1024;  // 4096-aligned and >= 8192: two halves
constexpr std::size_t kProjections = 4;               // three in the batch, one refused then retried
constexpr std::size_t kSentinelOffset = kProjections * kProjectionBytes;
constexpr std::size_t kSentinelBytes = 4096;          // direct, below the split floor: one job
constexpr std::size_t kFixtureBytes = kSentinelOffset + kSentinelBytes;

struct Hooks {
    std::mutex mutex;
    std::condition_variable cv;
    std::uint64_t alternate_device = 0;  // st_dev of the mirror filesystem; 0 = no mirror
    bool hold_alternate = false;
    bool release_alternate = false;
    bool fail_alternate = false;
    bool hold_primary = false;
    bool release_primary = false;
    int held_alternate = 0;
    int held_primary = 0;
    int completed_primary = 0;
    int completed_alternate = 0;
};

Hooks & hooks() {
    static Hooks instance;
    return instance;
}

void release_everything() {
    auto & h = hooks();
    std::lock_guard<std::mutex> lock(h.mutex);
    h.hold_alternate = false;
    h.hold_primary = false;
    h.release_alternate = true;
    h.release_primary = true;
    h.cv.notify_all();
}

template <class Predicate>
void wait_for(Predicate predicate, const char * what) {
    auto & h = hooks();
    std::unique_lock<std::mutex> lock(h.mutex);
    if (!h.cv.wait_for(lock, std::chrono::seconds(20), predicate)) {
        throw std::runtime_error(std::string("timed out waiting for ") + what);
    }
}

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error(message);
}

void expect(bool condition, const std::string & message) {
    if (!condition) fail(message);
}

void write_exact_at(int fd, const void * source, std::size_t length, std::uint64_t offset) {
    const auto * bytes = static_cast<const std::uint8_t *>(source);
    std::size_t done = 0;
    while (done < length) {
        const ssize_t count = pwrite(
            fd, bytes + done, length - done, static_cast<off_t>(offset + done));
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) fail("test pwrite failed: " + std::string(std::strerror(errno)));
        done += static_cast<std::size_t>(count);
    }
}

std::vector<std::uint8_t> fixture_bytes() {
    std::vector<std::uint8_t> fixture(kFixtureBytes);
    std::uint32_t state = 0x5851f42du;
    for (auto & byte : fixture) {
        state = state * 1664525u + 1013904223u;
        byte = static_cast<std::uint8_t>(state >> 24);
    }
    return fixture;
}

void make_fixture(const char * path) {
    const int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (fd < 0) fail(std::string("cannot create fixture ") + path + ": " + std::strerror(errno));
    const auto fixture = fixture_bytes();
    write_exact_at(fd, fixture.data(), fixture.size(), 0);
    if (fsync(fd) != 0) fail("fixture fsync failed");
    close(fd);
}

int open_readonly(const char * path) {
    const int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) fail(std::string("cannot open ") + path + ": " + std::strerror(errno));
    return fd;
}

std::uint64_t device_of(const char * path) {
    struct stat value {};
    if (stat(path, &value) != 0) {
        fail(std::string("cannot stat ") + path + ": " + std::strerror(errno));
    }
    return static_cast<std::uint64_t>(value.st_dev);
}

// The v2 mirror map line the companion parses: source FileKey, alternate FileKey, alternate path.
void write_map(const char * source_path, const char * mirror_path, const char * map_path) {
    const int source_fd = open_readonly(source_path);
    const int mirror_fd = open_readonly(mirror_path);
    const FileKey source = file_key(source_fd);
    const FileKey mirror = file_key(mirror_fd);
    close(source_fd);
    close(mirror_fd);
    expect(source.inode.device != mirror.inode.device,
           "the mirror must live on a different device than the source");
    expect(source.size == mirror.size, "source and mirror sizes differ");
    std::ofstream map(map_path, std::ios::trunc);
    if (!map) fail(std::string("cannot write mirror map ") + map_path);
    map << source.inode.device << '\t' << source.inode.inode << '\t' << source.size << '\t'
        << source.ctime_seconds << '\t' << source.ctime_nanoseconds << '\t'
        << mirror.inode.device << '\t' << mirror.inode.inode << '\t' << mirror.size << '\t'
        << mirror.ctime_seconds << '\t' << mirror.ctime_nanoseconds << '\t'
        << mirror_path << '\n';
    if (!map) fail("mirror map write failed");
    std::printf("MAP_OK source_dev=%llu mirror_dev=%llu\n",
                static_cast<unsigned long long>(source.inode.device),
                static_cast<unsigned long long>(mirror.inode.device));
}

memra_cpu_projection_v2 projection(int fd, std::uint64_t offset, std::size_t bytes) {
    memra_cpu_projection_v2 desc {};
    desc.weights = nullptr;
    desc.qtype = QT_F32;
    desc.in_features = 16;
    desc.out_features = static_cast<std::int32_t>(bytes / 64);
    desc.row_bytes = 64;
    desc.byte_len = bytes;
    desc.file_fd = fd;
    desc.file_offset = offset;
    desc.scale = 1.0f;
    return desc;
}

CacheKey key_of(int fd, const memra_cpu_projection_v2 & desc) {
    return CacheKey { file_key(fd), desc.file_offset, desc.byte_len };
}

std::int32_t submit(const memra_cpu_projection_v2 * descs, std::int32_t count) {
    std::array<char, 512> error {};
    const std::int32_t submitted =
        memra_cpu_expert_prefetch_v2(descs, count, error.data(), error.size());
    if (submitted < 0) fail(std::string("prefetch refused: ") + error.data());
    return submitted;
}

int inflight_signed() {
    return prefetch_inflight().load(std::memory_order_seq_cst);
}

// The bounded poll is only ever used AFTER the deterministic observations have passed and only
// where the buggy accounting cannot pass through the expected value on its way down (stated at
// each call site). It confirms the landing, it is not the balancing oracle.
template <class Predicate>
bool poll(Predicate predicate, std::chrono::milliseconds budget) {
    const auto deadline = std::chrono::steady_clock::now() + budget;
    for (;;) {
        if (predicate()) return true;
        if (std::chrono::steady_clock::now() >= deadline) return predicate();
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}

// Takes the key out of the annex once it lands and checks its bytes against the fixture.
void take_and_verify(int fd, const memra_cpu_projection_v2 & desc, const char * label) {
    const CacheKey key = key_of(fd, desc);
    std::shared_ptr<AlignedBytes> landed;
    poll([&] {
        landed = PrefetchAnnex::instance().take(key);
        return landed != nullptr;
    }, std::chrono::seconds(10));
    expect(landed != nullptr, std::string(label) + ": prefetched projection never landed");
    const auto fixture = fixture_bytes();
    expect(landed->size() >= desc.byte_len, std::string(label) + ": landed buffer too small");
    expect(std::memcmp(landed->data, fixture.data() + desc.file_offset, desc.byte_len) == 0,
           std::string(label) + ": landed bytes differ from the file");
}

void observe(const char * label, int expected) {
    const int observed = inflight_signed();
    std::printf("%s: inflight_signed=%d expected=%d\n", label, observed, expected);
    if (observed != expected) {
        fail(std::string("REGRESSION memra#586 at ") + label + ": inflight_signed="
             + std::to_string(observed) + " expected " + std::to_string(expected));
    }
}

void drain_to_zero(const char * label) {
    poll([] { return inflight_signed() == 0; }, std::chrono::seconds(2));
    observe(label, 0);
}

// Cell 1, the barrier and the cap. Three mirrored projections (two halves each) under
// MEMRA_CPU_EXPERT_IO_THREADS=3 and MEMRA_CPU_EXPERT_PREFETCH_MAX_INFLIGHT=3: every alternate half
// is held, so once three are held all three workers are blocked, the queue is empty and every
// primary half's bookkeeping is complete. The projections must still be charged (3), the cap must
// refuse a fourth, and after release the counter must drain to exactly zero and admit the retry.
int run_barrier(const char * source_path, const char * mirror_path) {
    auto & h = hooks();
    h.alternate_device = device_of(mirror_path);
    const int fd = open_readonly(source_path);
    std::array<memra_cpu_projection_v2, kProjections> descs {};
    for (std::size_t index = 0; index < kProjections; ++index) {
        descs[index] = projection(fd, index * kProjectionBytes, kProjectionBytes);
    }
    {
        std::lock_guard<std::mutex> lock(h.mutex);
        h.hold_alternate = true;
        h.release_alternate = false;
    }
    expect(submit(descs.data(), 3) == 3, "barrier: three projections were not all submitted");
    wait_for([&] { return h.held_alternate == 3; }, "three alternate halves to be held");
    int completed_primary = 0;
    int held_alternate = 0;
    {
        // Snapshot under the hook mutex: the workers write these counters under it.
        std::lock_guard<std::mutex> lock(h.mutex);
        completed_primary = h.completed_primary;
        held_alternate = h.held_alternate;
    }
    std::printf("barrier: primary halves completed=%d alternate halves held=%d\n",
                completed_primary, held_alternate);
    observe("barrier: primaries complete, alternates held", 3);
    const std::int32_t excess = submit(&descs[3], 1);
    std::printf("barrier: cap 3 with 3 charged: extra projection admitted=%d expected=0\n", excess);
    expect(excess == 0, "REGRESSION memra#586: the cap admitted excess work while three "
                        "projections were still in flight");
    {
        std::lock_guard<std::mutex> lock(h.mutex);
        h.release_alternate = true;
        h.cv.notify_all();
    }
    for (std::size_t index = 0; index < 3; ++index) {
        take_and_verify(fd, descs[index], "barrier batch");
    }
    // Sound here: with the defect the counter is already at most -1 once the last complete_read
    // ran (three releases of the primaries took it to zero, the alternates take it below), so a
    // read of zero after the three takes can only be the repaired accounting.
    drain_to_zero("barrier: drained after the three landed");
    expect(submit(&descs[3], 1) == 1, "barrier: the refused projection was not admitted on retry");
    take_and_verify(fd, descs[3], "barrier retry");
    drain_to_zero("barrier: drained after the retry landed");
    close(fd);
    std::printf("BARRIER_OK\n");
    return 0;
}

// Cell 2, the failure path and the retry. One mirrored projection under a single I/O worker:
// its alternate half is held, then made to fail with EIO. A direct single-job sentinel is
// queued behind it and held at ITS read entry: the single worker reaches that entry only after
// the failed projection's bookkeeping (abort_read plus the release of its charge), so the
// counter read there is deterministic: the sentinel's own charge and nothing else.
int run_failure(const char * source_path, const char * mirror_path) {
    auto & h = hooks();
    h.alternate_device = device_of(mirror_path);
    const int fd = open_readonly(source_path);
    const memra_cpu_projection_v2 mirrored = projection(fd, 0, kProjectionBytes);
    const memra_cpu_projection_v2 sentinel = projection(fd, kSentinelOffset, kSentinelBytes);
    {
        std::lock_guard<std::mutex> lock(h.mutex);
        h.hold_alternate = true;
        h.release_alternate = false;
    }
    expect(submit(&mirrored, 1) == 1, "failure: the mirrored projection was not submitted");
    wait_for([&] { return h.held_alternate == 1; }, "the alternate half to be held");
    observe("failure: primary half complete, alternate half held", 1);
    {
        std::lock_guard<std::mutex> lock(h.mutex);
        h.fail_alternate = true;
        h.hold_primary = true;
        h.release_primary = false;
    }
    expect(submit(&sentinel, 1) == 1, "failure: the sentinel was not submitted");
    observe("failure: sentinel queued behind the held half", 2);
    {
        std::lock_guard<std::mutex> lock(h.mutex);
        h.release_alternate = true;
        h.cv.notify_all();
    }
    wait_for([&] { return h.held_primary == 1; }, "the sentinel's read to be held");
    observe("failure: alternate half failed with EIO, sentinel read entered", 1);
    expect(PrefetchAnnex::instance().take(key_of(fd, mirrored)) == nullptr,
           "failure: a failed projection was published to the annex");
    {
        std::lock_guard<std::mutex> lock(h.mutex);
        h.hold_primary = false;
        h.release_primary = true;
        h.fail_alternate = false;
        h.hold_alternate = false;
        h.cv.notify_all();
    }
    take_and_verify(fd, sentinel, "failure sentinel");
    // Sound here: the deterministic observations above already decided the cell; the sentinel is
    // one job with one release in either accounting, so this only confirms its landing.
    drain_to_zero("failure: drained after the sentinel landed");
    expect(submit(&mirrored, 1) == 1, "failure: the failed projection was not admitted on retry");
    take_and_verify(fd, mirrored, "failure retry");
    drain_to_zero("failure: drained after the retry landed");
    close(fd);
    std::printf("FAILURE_PATH_OK\n");
    return 0;
}

// Cell 3, the non-mirrored control. Buffered reads (no MEMRA_CPU_EXPERT_IO, no mirror map): one
// job per projection, held at its entry under three workers; the charge equals the held count
// and drains to exactly zero. This path is unchanged by the repair and must read the same.
int run_parity(const char * source_path) {
    auto & h = hooks();
    const int fd = open_readonly(source_path);
    std::array<memra_cpu_projection_v2, 3> descs {};
    for (std::size_t index = 0; index < descs.size(); ++index) {
        descs[index] = projection(fd, index * kProjectionBytes, kProjectionBytes);
    }
    {
        std::lock_guard<std::mutex> lock(h.mutex);
        h.hold_primary = true;
        h.release_primary = false;
    }
    expect(submit(descs.data(), 3) == 3, "parity: three projections were not all submitted");
    wait_for([&] { return h.held_primary == 3; }, "three buffered reads to be held");
    observe("parity: three single-job reads held", 3);
    {
        std::lock_guard<std::mutex> lock(h.mutex);
        h.release_primary = true;
        h.cv.notify_all();
    }
    for (const auto & desc : descs) take_and_verify(fd, desc, "parity");
    drain_to_zero("parity: drained");
    close(fd);
    std::printf("PARITY_OK\n");
    return 0;
}

// Cell 4, the submit side (review round on #612). Two projections: the first is valid and takes
// its charge and annex claim inside the submit loop; the second names an fd that is not open, so
// file_key() throws before it is charged and before any job reaches the pool. The call must
// return -1, release the first projection's charge (inflight 0) and its annex claim (the same
// projection is admitted on retry, lands, and matches the file). Before the guard, the charge
// and the claim leaked for the life of the process.
int run_submit_throw(const char * source_path) {
    const int fd = open_readonly(source_path);
    std::array<memra_cpu_projection_v2, 2> descs {};
    descs[0] = projection(fd, 0, kProjectionBytes);
    descs[1] = projection(1 << 20, kProjectionBytes, kProjectionBytes);  // not an open fd
    std::array<char, 512> error {};
    const std::int32_t rc =
        memra_cpu_expert_prefetch_v2(descs.data(), 2, error.data(), error.size());
    std::printf("submit-throw: prefetch returned %d error=\"%s\"\n", rc, error.data());
    expect(rc < 0, "submit-throw: the call with an unopenable fd did not fail");
    observe("submit-throw: after the failed call", 0);
    expect(submit(descs.data(), 1) == 1,
           "REGRESSION memra#586 (submit side): the first projection's annex claim leaked, "
           "the retry was refused");
    take_and_verify(fd, descs[0], "submit-throw");
    drain_to_zero("submit-throw: after the retry landed");
    std::printf("submit-throw: PASS\n");
    return 0;
}

// Cell 5, the claim taken in the throwing iteration itself (review round 2 on #612). Under
// O_DIRECT with a mirror map that does not list this source, the loop takes the annex claim
// (begin_read) and then mirror resolve throws for the same projection, before a runtime exists
// and before any charge. The call must return -1 and the key must not stay claimed: begin_read
// on it afterwards must succeed (the fixture releases that probe claim again). Before the fix,
// the key read as speculated for the life of the process and no retry was ever admitted.
int run_submit_throw_claim(const char * source_path) {
    const int fd = open_readonly(source_path);
    memra_cpu_projection_v2 desc = projection(fd, 0, kProjectionBytes);
    std::array<char, 512> error {};
    const std::int32_t rc = memra_cpu_expert_prefetch_v2(&desc, 1, error.data(), error.size());
    std::printf("submit-throw-claim: prefetch returned %d error=\"%s\"\n", rc, error.data());
    expect(rc < 0, "submit-throw-claim: the call under a mirror map without this source did not fail");
    observe("submit-throw-claim: after the failed call", 0);
    const CacheKey key = key_of(fd, desc);
    const bool reclaimable = PrefetchAnnex::instance().begin_read(key);
    std::printf("submit-throw-claim: key claimable after the failed call=%d expected=1\n",
                reclaimable ? 1 : 0);
    if (reclaimable) PrefetchAnnex::instance().abort_read(key);
    expect(reclaimable, "REGRESSION memra#586 (submit side): the throwing projection's own annex "
                        "claim leaked, the key reads as speculated");
    std::printf("submit-throw-claim: PASS\n");
    return 0;
}

int run(int argc, char ** argv) {
    if (argc < 2) fail("usage: make-fixture PATH | write-map SOURCE MIRROR MAP | "
                       "barrier SOURCE MIRROR | failure SOURCE MIRROR | parity SOURCE | "
                       "submit-throw SOURCE | submit-throw-claim SOURCE");
    const std::string mode = argv[1];
    if (mode == "make-fixture" && argc == 3) {
        make_fixture(argv[2]);
        std::printf("FIXTURE_OK\n");
        return 0;
    }
    if (mode == "write-map" && argc == 5) {
        write_map(argv[2], argv[3], argv[4]);
        return 0;
    }
    if (mode == "barrier" && argc == 4) return run_barrier(argv[2], argv[3]);
    if (mode == "failure" && argc == 4) return run_failure(argv[2], argv[3]);
    if (mode == "parity" && argc == 3) return run_parity(argv[2]);
    if (mode == "submit-throw" && argc == 3) return run_submit_throw(argv[2]);
    if (mode == "submit-throw-claim" && argc == 3) return run_submit_throw_claim(argv[2]);
    fail("unknown mode or argument count: " + mode);
}

}  // namespace prefetch_test

ssize_t memra_test_pread(int fd, void * buffer, size_t length, off_t offset) {
    auto & h = prefetch_test::hooks();
    struct stat value {};
    if (fstat(fd, &value) != 0) return -1;  // errno from fstat
    const bool alternate = h.alternate_device != 0
        && static_cast<std::uint64_t>(value.st_dev) == h.alternate_device;
    bool fail_read = false;
    {
        std::unique_lock<std::mutex> lock(h.mutex);
        if (alternate ? h.hold_alternate : h.hold_primary) {
            int & held = alternate ? h.held_alternate : h.held_primary;
            ++held;
            h.cv.notify_all();
            h.cv.wait(lock, [&] { return alternate ? h.release_alternate : h.release_primary; });
            --held;
        }
        fail_read = alternate && h.fail_alternate;
    }
    if (fail_read) {
        errno = EIO;
        return -1;
    }
    const ssize_t count = ::pread(fd, buffer, length, offset);
    if (count >= 0) {
        std::lock_guard<std::mutex> lock(h.mutex);
        (alternate ? h.completed_alternate : h.completed_primary) += 1;
    }
    return count;
}

int main(int argc, char ** argv) {
    int status = 0;
    try {
        status = prefetch_test::run(argc, argv);
    } catch (const std::exception & exception) {
        std::fprintf(stderr, "FAIL: %s\n", exception.what());
        status = 1;
    }
    // Never exit with a held read: the pool's destructor joins its workers.
    prefetch_test::release_everything();
    return status;
}
