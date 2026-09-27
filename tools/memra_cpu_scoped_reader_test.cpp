// Tiny CPU-only controls on the actual companion translation unit. No CUDA or model load.
#include "memra_cpu_experts.cpp"

namespace {
void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}
struct ReaderStats {
    std::atomic<int> reads {0}, destroyed {0};
    std::mutex mutex;
    std::condition_variable cv;
    bool paused = false;
    void resume() {
        { std::lock_guard<std::mutex> lock(mutex); paused = false; }
        cv.notify_all();
    }
};
struct ResumeReads {
    std::shared_ptr<ReaderStats> stats;
    ~ResumeReads() { stats->resume(); }
};
template<class Predicate> bool wait_until(Predicate done) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!done() && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return done();
}
struct ReaderContext {
    std::atomic<int> refs {1};
    int fd;
    std::uint64_t offset;
    std::size_t len;
    bool direct, fail_info = false, fail_read = false;
    std::shared_ptr<ReaderStats> stats;
};
void retain_reader(const void * raw) {
    const_cast<ReaderContext *>(static_cast<const ReaderContext *>(raw))->refs.fetch_add(1);
}
void release_reader(const void * raw) {
    auto * ctx = const_cast<ReaderContext *>(static_cast<const ReaderContext *>(raw));
    if (ctx->refs.fetch_sub(1) == 1) {
        close(ctx->fd);
        ctx->stats->destroyed.fetch_add(1);
        delete ctx;
    }
}
std::int32_t reader_info(const void * raw, memra_scoped_disk_info_v1 * out) {
    auto * ctx = static_cast<const ReaderContext *>(raw);
    if (ctx->fail_info) return EIO;
    const auto key = file_key(ctx->fd);
    *out = {key.inode.device, key.inode.inode, key.size, key.ctime_seconds,
            key.ctime_nanoseconds, ctx->offset, ctx->len, std::uint32_t(ctx->direct)};
    return 0;
}
std::int32_t read_reader(const void * raw, std::uint8_t * dst, std::size_t len,
                        std::uint64_t offset, std::size_t * read_bytes) {
    auto * ctx = static_cast<const ReaderContext *>(raw);
    *read_bytes = 0;
    if (offset > ctx->len || len > ctx->len - offset) return EINVAL;
    ctx->stats->reads.fetch_add(1);
    {
        std::unique_lock<std::mutex> lock(ctx->stats->mutex);
        ctx->stats->cv.wait(lock, [&] { return !ctx->stats->paused; });
    }
    if (ctx->fail_read) return EIO;
    const auto count = pread(ctx->fd, dst, len, ctx->offset + offset);
    if (count < 0) return errno;
    *read_bytes = std::size_t(count);
    return 0;
}
memra_scoped_disk_reader_v1 make_reader(int fd, std::uint64_t offset, std::size_t len,
        const std::shared_ptr<ReaderStats> & stats) {
    const bool direct = direct_io_enabled();
    const std::string path = "/proc/self/fd/" + std::to_string(fd);
    const int owned = open(path.c_str(), O_RDONLY | O_CLOEXEC | (direct ? O_DIRECT : 0));
    require(owned >= 0, "fixture reader reopen failed");
    auto * ctx = new ReaderContext {{1}, owned, offset, len, direct, false, false, stats};
    return {1, ctx, retain_reader, release_reader, read_reader, reader_info};
}
std::vector<std::uint8_t> weights(int in, int out) {
    std::vector<std::uint8_t> bytes(std::size_t(in / 32) * 34 * out);
    for (std::size_t at = 0; at < bytes.size(); at += 34) {
        bytes[at] = 0; bytes[at + 1] = 0x3c; // q8_0 d=1
        for (int lane = 0; lane < 32; ++lane) bytes[at + 2 + lane] = std::uint8_t((int(at / 34) + lane) % 7 - 3);
    }
    return bytes;
}
void write_all(int fd, const void * raw, std::size_t size) {
    const auto * bytes = static_cast<const std::uint8_t *>(raw);
    while (size) {
        const auto n = write(fd, bytes, size);
        if (n < 0 && errno == EINTR) continue;
        require(n > 0, "fixture write failed");
        bytes += n; size -= std::size_t(n);
    }
}
int fixture(const std::string & path, const std::array<std::vector<std::uint8_t>, 3> & bytes) {
    const int fd = open(path.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    require(fd >= 0, "fixture create failed");
    std::array<std::uint8_t, 4096> prefix {};
    write_all(fd, prefix.data(), prefix.size());
    for (const auto & slab : bytes) write_all(fd, slab.data(), slab.size());
    require(fsync(fd) == 0, "fixture fsync failed");
    return fd;
}
memra_cpu_projection_v2 projection(int in, int out, int fd, std::uint64_t offset) {
    return {nullptr, QT_Q8_0, in, out, std::size_t(in/32)*34,
            std::size_t(in/32)*34*out, fd, offset, 0.03125f};
}
void run(const std::string & root) {
    const int in = 128, ff = 512;
    const auto gate = weights(in, ff), down = weights(ff, in);
    require(gate.size() % 4096 == 0 && gate.size() == down.size(), "fixture alignment");
    const std::array<std::vector<std::uint8_t>, 3> bytes {gate, gate, down};
    const auto raw_path=root+"/raw.bin", scoped_path=root+"/scoped.bin", prefetch_path=root+"/prefetch.bin";
    const int raw_fd = fixture(raw_path, bytes), scoped_fd = fixture(scoped_path, bytes);
    memra_cpu_expert_v2 raw {projection(in,ff,raw_fd,4096), projection(in,ff,raw_fd,4096+gate.size()),
                            projection(ff,in,raw_fd,4096+2*gate.size()), 0.5f};
    memra_cpu_expert_v2 scoped=raw;
    scoped.gate.file_fd=scoped.up.file_fd=scoped.down.file_fd=-1;
    scoped.gate.file_offset=scoped.up.file_offset=scoped.down.file_offset=0;
    auto stats=std::make_shared<ReaderStats>();
    std::array<memra_scoped_disk_reader_v1,3> handles;
    std::array<const memra_scoped_disk_reader_v1 *,3> readers;
    for (int i=0;i<3;++i) { handles[i]=make_reader(scoped_fd,4096+i*gate.size(),gate.size(),stats); readers[i]=&handles[i]; }
    close(scoped_fd);
    std::vector<float> input(in), raw_out(in), scoped_out(in);
    for (int i=0;i<in;++i) input[i]=float(i%13-6)*0.01f;
    std::array<char,1024> error {};
    require(memra_cpu_experts_abi_version()==2,"legacy ABI changed");
    require(memra_cpu_moe_token_v2(&raw,1,input.data(),raw_out.data(),4,error.data(),error.size())==0,error.data());
    require(memra_cpu_moe_token_scoped_v1(&scoped,1,input.data(),scoped_out.data(),4,error.data(),error.size(),readers.data())==0,error.data());
    require(std::memcmp(raw_out.data(),scoped_out.data(),in*sizeof(float))==0,"token bytes differ");
    const int cold_reads=stats->reads.load();require(cold_reads>=3,"scoped cold path did not read");
    require(memra_cpu_moe_token_scoped_v1(&scoped,1,input.data(),scoped_out.data(),4,error.data(),error.size(),readers.data())==0,error.data());
    require(stats->reads.load()==cold_reads,"scoped warm cache missed");
    require(std::memcmp(raw_out.data(),scoped_out.data(),in*sizeof(float))==0,"warm token bytes differ");
    std::vector<float> inputs=input;inputs.insert(inputs.end(),input.begin(),input.end());
    std::array<float,2> route {0.5f,0.25f};raw_out.resize(2*in);scoped_out.resize(2*in);
    require(memra_cpu_expert_rows_v2(&raw,inputs.data(),2,route.data(),raw_out.data(),4,error.data(),error.size())==0,error.data());
    require(memra_cpu_expert_rows_scoped_v1(&scoped,inputs.data(),2,route.data(),scoped_out.data(),4,error.data(),error.size(),readers.data())==0,error.data());
    require(std::memcmp(raw_out.data(),scoped_out.data(),2*in*sizeof(float))==0,"rows bytes differ");
    // Unit route weights still apply down.scale; they cannot implement raw rows.
    auto bare_raw=raw; auto bare_scoped=scoped;
    bare_raw.down.scale=bare_scoped.down.scale=0.37f;
    std::vector<float> bare(2*in), bound_bare(2*in), incorrectly_scaled(2*in);
    require(memra_cpu_expert_rows_raw_v2(&bare_raw,inputs.data(),2,bare.data(),4,error.data(),error.size())==0,error.data());
    require(memra_cpu_expert_rows_raw_scoped_v1(&bare_scoped,inputs.data(),2,bound_bare.data(),4,error.data(),error.size(),readers.data())==0,error.data());
    require(std::memcmp(bare.data(),bound_bare.data(),bare.size()*sizeof(float))==0,"scoped raw rows changed bare down projection");
    std::array<float,2> ones {1.0f,1.0f};
    require(memra_cpu_expert_rows_scoped_v1(&bare_scoped,inputs.data(),2,ones.data(),incorrectly_scaled.data(),4,error.data(),error.size(),readers.data())==0,error.data());
    require(std::memcmp(bare.data(),incorrectly_scaled.data(),bare.size()*sizeof(float))!=0,"weighted unit rows accidentally treated as raw");
    std::array<memra_cpu_expert_v2,2> ordered {bare_raw,bare_raw};
    ordered[0].route_weight=0.73f; ordered[1].route_weight=-0.29f; ordered[1].down.scale=0.61f;
    std::vector<float> token(in),fold(in,0.0f);
    require(memra_cpu_moe_token_v2(ordered.data(),2,input.data(),token.data(),4,error.data(),error.size())==0,error.data());
    for (const auto & ex:ordered) for (int i=0;i<in;++i) fold[i]=std::fma(bare[i],ex.route_weight*ex.down.scale,fold[i]);
    require(std::memcmp(token.data(),fold.data(),in*sizeof(float))==0,"raw rows lost ordered token FMA parity");
    auto * first=const_cast<ReaderContext *>(static_cast<const ReaderContext *>(handles[0].context));
    require(first->refs.load()==1,"completed call retained scoped reader");
    first->fail_info=true;
    require(memra_cpu_moe_token_scoped_v1(&scoped,1,input.data(),scoped_out.data(),4,error.data(),error.size(),readers.data())!=0,"metadata failure accepted");
    require(first->refs.load()==1,"metadata failure leaked retained reader");first->fail_info=false;
    for (auto & h:handles) h.release(h.context);
    require(stats->destroyed.load()==3,"token/rows scope did not release all readers");
    close(raw_fd);

    // Force reads to remain blocked after prefetch returns and every caller reference drops.
    const int pf=fixture(prefetch_path,bytes);auto detached=std::make_shared<ReaderStats>();
    detached->paused=true;ResumeReads unblock {detached};
    const FileKey pf_key=file_key(pf);
    std::array<memra_cpu_projection_v2,3> projections {scoped.gate,scoped.up,scoped.down};
    for (int i=0;i<3;++i) { handles[i]=make_reader(pf,4096+i*gate.size(),gate.size(),detached); readers[i]=&handles[i]; }
    require(memra_cpu_expert_prefetch_scoped_v1(projections.data(),3,error.data(),error.size(),readers.data())==3,error.data());
    for (auto & h:handles) h.release(h.context);
    close(pf);require(unlink(prefetch_path.c_str())==0,"prefetch unlink failed");
    const int replacement=open(prefetch_path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600);
    require(replacement>=0,"replacement create failed");write_all(replacement,"replacement",11);close(replacement);
    const bool started=wait_until([&] { return detached->reads.load()>0; });
    const bool live=detached->destroyed.load()==0 && prefetch_inflight().load()==3;
    detached->resume();
    require(started && live,"detached reads were not held alive after caller release");
    require(wait_until([&] { return prefetch_inflight().load()==0 && detached->destroyed.load()==3; }),"detached ownership did not drain");
    for (int i=0;i<3;++i) {
        auto landed=PrefetchAnnex::instance().take(CacheKey {pf_key,4096+i*gate.size(),gate.size()});
        require(bool(landed),"detached read failed to publish annex bytes");
        require(std::memcmp(landed->data,bytes[i].data(),bytes[i].size())==0,"detached bytes differ after path replacement");
    }
    unlink(prefetch_path.c_str());

    // Failure on the second reader must undo the first prepared claim without submitting I/O.
    const auto failed_path=root+"/failures.bin";const int failed_fd=fixture(failed_path,bytes);
    const FileKey failed_key=file_key(failed_fd);auto failed=std::make_shared<ReaderStats>();
    auto context = [&](int i) { return const_cast<ReaderContext *>(static_cast<const ReaderContext *>(handles[i].context)); };
    auto original_refs = [&] { return context(0)->refs.load()==1 && context(1)->refs.load()==1 && context(2)->refs.load()==1; };
    for (int i=0;i<3;++i) { handles[i]=make_reader(failed_fd,4096+i*gate.size(),gate.size(),failed); readers[i]=&handles[i]; }
    context(1)->fail_info=true;
    require(memra_cpu_expert_prefetch_scoped_v1(projections.data(),3,error.data(),error.size(),readers.data())==-1,"partial metadata failure accepted");
    require(original_refs() && failed->reads.load()==0 && prefetch_inflight().load()==0,"preparation failure leaked owner, I/O or counter");
    for (int i=0;i<3;++i) require(!PrefetchAnnex::instance().speculated(CacheKey {failed_key,4096+i*gate.size(),gate.size()}),"preparation failure leaked annex claim");
    context(1)->fail_info=false;

    // Demand and detached EIO must drain retained readers and never publish bad bytes.
    for (int i=0;i<3;++i) context(i)->fail_read=true;
    require(memra_cpu_moe_token_scoped_v1(&scoped,1,input.data(),scoped_out.data(),4,error.data(),error.size(),readers.data())!=0,"demand EIO accepted");
    require(original_refs(),"demand EIO returned before readers drained");
    for (int i=0;i<3;++i) require(!weight_cache().contains(CacheKey {failed_key,4096+i*gate.size(),gate.size()}),"demand EIO published cache bytes");
    require(memra_cpu_expert_prefetch_scoped_v1(projections.data(),3,error.data(),error.size(),readers.data())==3,error.data());
    require(wait_until([&] { return prefetch_inflight().load()==0 && original_refs(); }),"prefetch EIO did not drain");
    for (int i=0;i<3;++i) {
        const CacheKey key {failed_key,4096+i*gate.size(),gate.size()};
        require(!PrefetchAnnex::instance().speculated(key) && !weight_cache().contains(key),"prefetch EIO published or retained claim");
        context(i)->fail_read=false;
    }
    require(memra_cpu_expert_prefetch_scoped_v1(projections.data(),3,error.data(),error.size(),readers.data())==3,"failed source could not retry");
    require(wait_until([&] { return prefetch_inflight().load()==0 && original_refs(); }),"successful retry did not drain");
    for (int i=0;i<3;++i) {
        auto landed=PrefetchAnnex::instance().take(CacheKey {failed_key,4096+i*gate.size(),gate.size()});
        require(bool(landed) && std::memcmp(landed->data,bytes[i].data(),bytes[i].size())==0,"retry annex bytes differ");
        handles[i].release(handles[i].context);
    }
    require(failed->destroyed.load()==3,"failure/retry readers leaked");close(failed_fd);
    unlink(failed_path.c_str());unlink(raw_path.c_str());unlink(scoped_path.c_str());
    std::printf("SCOPED_CPU_PASS token_rows_bit_identity=1 cold_reads=%d warm_cache=1 detached_barrier=1 annex_bytes=1 partial_metadata_rollback=1 eio_drain_retry=1 mode=%s pipeline=%d\n",cold_reads,direct_io_enabled()?"direct":"buffered",int(pipeline_enabled()));

}
}
int main(int argc,char ** argv) try {
    require(argc==2,"expected isolated fixture directory");run(argv[1]);return 0;
} catch(const std::exception & e) { std::fprintf(stderr,"SCOPED_CPU_FAIL: %s\n",e.what());return 1; }
