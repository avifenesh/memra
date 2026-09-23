#ifndef MEMRA_CPU_SCOPED_READER_H
#define MEMRA_CPU_SCOPED_READER_H
#include "../crates/memra-gguf/include/memra_scoped_disk_v1.h"
#include <cerrno>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

// Native ownership wrapper for one prepared range reader. No fd is obtained even on a cache miss.
// A shared_ptr to this object may move into detached I/O state; its destructor releases after drain.
class MemraScopedReader {
public:
    explicit MemraScopedReader(const memra_scoped_disk_reader_v1 & source) : reader_(source) {
        if (reader_.version != 1 || !reader_.context || !reader_.retain || !reader_.release
            || !reader_.read_at || !reader_.info) {
            throw std::runtime_error("invalid scoped CPU reader ABI");
        }
        reader_.retain(reader_.context);
        const int status = reader_.info(reader_.context, &info_);
        if (status || info_.len == 0 || info_.len > std::numeric_limits<std::size_t>::max()
            || info_.offset > info_.file_bytes || info_.len > info_.file_bytes - info_.offset
            || info_.direct > 1) {
            reader_.release(reader_.context);
            throw std::runtime_error(status ? std::string("scoped CPU reader metadata: ")
                + std::strerror(status) : "invalid scoped CPU reader geometry");
        }
    }
    MemraScopedReader(const MemraScopedReader &) = delete;
    MemraScopedReader & operator=(const MemraScopedReader &) = delete;
    ~MemraScopedReader() { reader_.release(reader_.context); }
    const memra_scoped_disk_info_v1 & info() const { return info_; }
    int read_exact(void * destination, std::size_t offset, std::size_t length) const {
        if ((!destination && length) || offset > info_.len || length > info_.len - offset) return EINVAL;
        auto * bytes = static_cast<std::uint8_t *>(destination);
        std::size_t done = 0;
        while (done < length) {
            std::size_t count = 0;
            const int status = reader_.read_at(reader_.context, bytes + done, length - done,
                                               offset + done, &count);
            if (status == EINTR) continue;
            if (status) return status;
            if (!count || count > length - done) return EIO;
            done += count;
        }
        return 0;
    }
private:
    memra_scoped_disk_reader_v1 reader_;
    memra_scoped_disk_info_v1 info_ {};
};
#endif
