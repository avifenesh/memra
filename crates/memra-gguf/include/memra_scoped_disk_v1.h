#ifndef MEMRA_SCOPED_DISK_V1_H
#define MEMRA_SCOPED_DISK_V1_H
#include <stddef.h>
#include <stdint.h>

/* Internal process-native ABI; no file descriptor or whole-mapping access. */
struct memra_scoped_disk_info_v1 {
    uint64_t device, inode, file_bytes;
    int64_t ctime_seconds, ctime_nanoseconds;
    uint64_t offset, len;
    uint32_t direct;
};

/* The descriptor borrows context. Retain before detached use, then release exactly once
 * after all reads finish. read_at returns an errno-style status and initializes bytes_read
 * to zero on failure. Offsets are relative to the authorized extent, including direct I/O.
 * Non-null pointers must address valid objects/buffers; callbacks may run concurrently on
 * separate destination buffers. Source files must remain immutable while loaded. */
struct memra_scoped_disk_reader_v1 {
    uint32_t version;
    const void *context;
    void (*retain)(const void *);
    void (*release)(const void *);
    int32_t (*read_at)(const void *, uint8_t *, size_t, uint64_t, size_t *);
    int32_t (*info)(const void *, struct memra_scoped_disk_info_v1 *);
};
#endif
