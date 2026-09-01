// Focused process-level tests for the persistent CPU expert cache. Including the production
// translation unit keeps the test on the exact ShmArena/WeightCache implementation without
// adding test hooks to the stable companion ABI.
#include "memra_cpu_experts.cpp"

namespace {

constexpr std::size_t kFixtureBytes = 64 * 1024;

struct TestHeader {
    std::uint64_t magic;
    std::uint32_t version;
    std::uint32_t state;
    std::uint64_t segment_bytes;
    std::uint64_t entry_count;
};

static_assert(sizeof(TestHeader) == 32, "unexpected shm cache header layout");

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error(message);
}

void read_exact_at(int fd, void * destination, std::size_t length, std::uint64_t offset) {
    auto * bytes = static_cast<std::uint8_t *>(destination);
    std::size_t done = 0;
    while (done < length) {
        const ssize_t count = pread(
            fd, bytes + done, length - done, static_cast<off_t>(offset + done));
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) fail("test pread failed: " + std::string(std::strerror(errno)));
        done += static_cast<std::size_t>(count);
    }
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

int open_fixture(const char * path) {
    int fd = open(path, O_RDWR | O_CREAT | O_EXCL, 0600);
    if (fd >= 0) {
        std::array<std::uint8_t, kFixtureBytes> fixture {};
        std::uint32_t state = 0x12345678u;
        for (auto & byte : fixture) {
            state = state * 1664525u + 1013904223u;
            byte = static_cast<std::uint8_t>(state >> 24);
        }
        write_exact_at(fd, fixture.data(), fixture.size(), 0);
        if (fsync(fd) != 0) fail("test fixture fsync failed");
        return fd;
    }
    if (errno != EEXIST) fail("cannot create test fixture: " + std::string(std::strerror(errno)));
    fd = open(path, O_RDONLY);
    if (fd < 0) fail("cannot reopen test fixture: " + std::string(std::strerror(errno)));
    struct stat st {};
    if (fstat(fd, &st) != 0 || st.st_size != static_cast<off_t>(kFixtureBytes)) {
        fail("existing test fixture has the wrong size");
    }
    return fd;
}

std::shared_ptr<AlignedBytes> read_fixture(int fd) {
    auto bytes = std::make_shared<AlignedBytes>();
    bytes->resize(kFixtureBytes, 4096);
    read_exact_at(fd, bytes->data, kFixtureBytes, 0);
    return bytes;
}

void verify_fixture(int fd, const AlignedBytes & cached) {
    std::array<std::uint8_t, kFixtureBytes> source {};
    read_exact_at(fd, source.data(), source.size(), 0);
    if (std::memcmp(source.data(), cached.data, source.size()) != 0) {
        fail("cache hit differs from the source fixture");
    }
}

void exercise_cache(const char * path, const std::string & expectation) {
    const int fd = open_fixture(path);
    const CacheKey key { file_key(fd), 0, kFixtureBytes };
    {
        WeightCache cache;
        const bool shared = ShmArena::instance().enabled();
        if (expectation == "private" && shared) fail("unsafe shm object was adopted");
        if (expectation != "private" && !shared) fail("expected shm arena is disabled");

        auto found = cache.find(key);
        if (expectation == "warm") {
            if (!found) fail("expected persisted warm hit was a miss");
            verify_fixture(fd, *found);
            std::puts("WARM_HIT_OK");
        } else {
            if (found) fail("expected cache miss was a warm hit");
            auto source = read_fixture(fd);
            cache.insert(key, source);
            found = cache.find(key);
            if (!found) fail("source re-read did not populate the cache");
            verify_fixture(fd, *found);
            if (expectation == "seed") {
                std::puts("SEED_OK");
            } else if (expectation == "miss") {
                std::puts("MISS_REREAD_OK");
            } else if (expectation == "private") {
                if (ShmArena::instance().contains(found->data)) {
                    fail("private fallback returned a shared-memory block");
                }
                std::puts("PRIVATE_FALLBACK_OK");
            } else {
                fail("unknown cache expectation " + expectation);
            }
        }
    }
    close(fd);
}

void precreate(const char * name, mode_t mode) {
    const int fd = shm_open(name, O_RDWR | O_CREAT | O_EXCL, mode);
    if (fd < 0) fail("test shm precreate failed: " + std::string(std::strerror(errno)));
    if (fchmod(fd, mode) != 0) fail("test shm fchmod failed");
    close(fd);
}

void mutate(const char * name, const std::string & kind) {
    const int fd = shm_open(name, O_RDWR, 0);
    if (fd < 0) fail("cannot open test shm: " + std::string(std::strerror(errno)));
    TestHeader header {};
    read_exact_at(fd, &header, sizeof(header), 0);
    if (header.version != 2 || header.state != 1 || header.entry_count == 0) {
        fail("test shm does not contain a clean v2 entry");
    }
    ShmArena::PersistedEntry row {};
    read_exact_at(fd, &row, sizeof(row), 4096);
    if (kind == "past-end") {
        row.shm_offset = header.segment_bytes + 4096;
        write_exact_at(fd, &row, sizeof(row), 4096);
    } else if (kind == "overflow") {
        row.shm_offset = std::numeric_limits<std::uint64_t>::max() - row.pool_bytes + 1;
        write_exact_at(fd, &row, sizeof(row), 4096);
    } else if (kind == "checksum") {
        std::uint8_t byte = 0;
        read_exact_at(fd, &byte, sizeof(byte), row.shm_offset);
        byte ^= 0x80;
        write_exact_at(fd, &byte, sizeof(byte), row.shm_offset);
    } else {
        fail("unknown shm mutation " + kind);
    }
    if (fsync(fd) != 0) fail("test shm fsync failed");
    close(fd);
}

} // namespace

int main(int argc, char ** argv) try {
    if (argc < 2) fail("missing test command");
    const std::string command = argv[1];
    if (command == "cache" && argc == 4) {
        exercise_cache(argv[2], argv[3]);
    } else if (command == "precreate" && argc == 4) {
        char * end = nullptr;
        const unsigned long mode = std::strtoul(argv[3], &end, 8);
        if (end == argv[3] || *end != '\0' || mode > 0777) fail("invalid test shm mode");
        precreate(argv[2], static_cast<mode_t>(mode));
    } else if (command == "mutate" && argc == 4) {
        mutate(argv[2], argv[3]);
    } else if (command == "unlink" && argc == 3) {
        if (shm_unlink(argv[2]) != 0 && errno != ENOENT) {
            fail("test shm unlink failed: " + std::string(std::strerror(errno)));
        }
    } else {
        fail("invalid test command");
    }
    return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "shm test failure: %s\n", error.what());
    return 1;
}
