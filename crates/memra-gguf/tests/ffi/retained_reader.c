#include "memra_scoped_disk_v1.h"
#include <errno.h>
#include <stdlib.h>

struct retained_reader { struct memra_scoped_disk_reader_v1 reader; };

void *memra_ffi_test_capture(const struct memra_scoped_disk_reader_v1 *reader) {
    if (!reader || reader->version != 1 || !reader->context || !reader->retain ||
        !reader->release || !reader->read_at || !reader->info) return NULL;
    struct retained_reader *state = malloc(sizeof(*state));
    if (!state) return NULL;
    state->reader = *reader;
    state->reader.retain(state->reader.context);
    return state;
}

int memra_ffi_test_read_and_release(void *opaque, uint8_t *out, size_t len) {
    if (!opaque || !out || !len) return EINVAL;
    struct retained_reader *state = opaque;
    const struct memra_scoped_disk_reader_v1 *reader = &state->reader;
    struct memra_scoped_disk_info_v1 info = {0};
    int error = reader->info(reader->context, &info);
    if (!error && (info.len != len || info.direct != 0 || info.file_bytes < len)) error = EINVAL;
    size_t done = 0;
    while (!error && done < len) {
        size_t count = 0;
        error = reader->read_at(reader->context, out + done, len - done, done, &count);
        if (error == EINTR) { error = 0; continue; }
        if (!error && count == 0) error = EIO;
        done += count;
    }
    if (!error) {
        size_t count = 999;
        if (reader->read_at(reader->context, out, len, 1, &count) != EINVAL || count != 0) error = EIO;
        if (reader->read_at(reader->context, out, len, UINT64_MAX, &count) != EINVAL || count != 0) error = EIO;
    }
    reader->release(reader->context);
    free(state);
    return error;
}
