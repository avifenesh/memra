// Test-only allocator and publication fault against the actual companion and scoped readers.
#include <memory>
#include <atomic>
#include <new>
static std::atomic<bool> queue_allocation_failure {false};
static std::atomic<bool> publication_failure {false};
template<class T> struct MemraCpuQueueTestAllocator {
    using value_type = T;
    MemraCpuQueueTestAllocator() = default;
    template<class U> MemraCpuQueueTestAllocator(const MemraCpuQueueTestAllocator<U> &) noexcept {}
    T * allocate(std::size_t n) {
        if (queue_allocation_failure.load()) throw std::bad_alloc();
        return std::allocator<T>{}.allocate(n);
    }
    void deallocate(T * p, std::size_t n) noexcept { std::allocator<T>{}.deallocate(p,n); }
    template<class U> struct rebind { using other = MemraCpuQueueTestAllocator<U>; };
};
template<class T,class U> bool operator==(const MemraCpuQueueTestAllocator<T>&,const MemraCpuQueueTestAllocator<U>&) { return true; }
template<class T,class U> bool operator!=(const MemraCpuQueueTestAllocator<T>&,const MemraCpuQueueTestAllocator<U>&) { return false; }
#define MEMRA_CPU_QUEUE_TEST
#define main memra_existing_scoped_control_main
#include "memra_cpu_scoped_reader_test.cpp"
#undef main
extern "C" void memra_cpu_test_before_queue_publish() {
    if (publication_failure.load()) throw std::bad_alloc();
}
namespace {
struct MemraCpuQueueTestAccess {
    static std::vector<std::vector<std::size_t>> snapshot(IoPool & p) {
        std::lock_guard<std::mutex> lock(p.mutex_);
        std::vector<std::vector<std::size_t>> result;
        for (const auto & b:p.queue_) {
            std::vector<std::size_t> row;
            for (const auto & job:b.jobs) row.push_back(job.length);
            result.push_back(std::move(row));
        }
        return result;
    }
    static void allocation_failure_preserves_prefix() {
        IoPool p; // no workers: inspect queued ownership without consuming non-owning sentinels
        bool failed=false;
        for (std::size_t i=0;i<1024;++i) {
            const auto before=snapshot(p);
            std::vector<IoJob> pair(2);
            pair[0].length=2*i+1; pair[1].length=2*i+2; pair[1].alternate=true;
            queue_allocation_failure.store(!before.empty());
            try { p.submit(std::move(pair)); }
            catch (const std::bad_alloc &) {
                queue_allocation_failure.store(false);
                require(!before.empty(),"failure must preserve a pre-existing queued prefix");
                require(snapshot(p)==before,"failed whole-batch allocation changed queued prefix");
                failed=true;break;
            }
            queue_allocation_failure.store(false);
        }
        require(failed,"deque allocator failure was not exercised");
        require(p.workers_.empty(),"pure queue control unexpectedly started workers");
    }
};
void scoped_publication_failure(const std::string & root) {
    require(pipeline_enabled(),"run publication control with MEMRA_CPU_EXPERT_PIPELINE=1");
    const int in=128,ff=512;
    const std::array<std::vector<std::uint8_t>,3> bytes {weights(in,ff),weights(in,ff),weights(ff,in)};
    const auto path=root+"/publication.bin";const int fd=fixture(path,bytes);const FileKey key=file_key(fd);
    auto stats=std::make_shared<ReaderStats>();
    std::array<memra_scoped_disk_reader_v1,3> handles;
    std::array<const memra_scoped_disk_reader_v1 *,3> readers;
    std::array<memra_cpu_projection_v2,3> projections;
    std::size_t off=4096;
    for(int i=0;i<3;++i) {
        projections[i]=projection(i==2?ff:in,i==2?in:ff,-1,0);
        handles[i]=make_reader(fd,off,bytes[i].size(),stats);readers[i]=&handles[i];off+=bytes[i].size();
    }
    const auto refs_balanced=[&] {
        for(auto & h:handles) if(static_cast<const ReaderContext *>(h.context)->refs.load()!=1) return false;
        return true;
    };
    std::array<char,1024> error{};std::vector<float> input(in,0.01f),output(in);
    memra_cpu_expert_v2 expert{projections[0],projections[1],projections[2],0.7f};
    publication_failure.store(true);
    require(memra_cpu_expert_prefetch_scoped_v1(projections.data(),3,error.data(),error.size(),readers.data())==-1,"failed prefetch publication was accepted");
    require(prefetch_inflight().load()==0 && refs_balanced(),"prefetch publication leaked charge/readers");
    require(stats->reads.load()==0,"failed prefetch published work to a worker");
    off=4096;
    for(int i=0;i<3;++i) {require(!PrefetchAnnex::instance().speculated(CacheKey{key,off,bytes[i].size()}),"failed publication leaked annex claim");off+=bytes[i].size();}
    require(memra_cpu_moe_token_scoped_v1(&expert,1,input.data(),output.data(),4,error.data(),error.size(),readers.data())!=0,"failed demand publication was accepted");
    require(stats->reads.load()==0 && refs_balanced(),"demand publication leaked or waited for unpublished work");
    publication_failure.store(false);
    require(memra_cpu_moe_token_scoped_v1(&expert,1,input.data(),output.data(),4,error.data(),error.size(),readers.data())==0,error.data());
    require(refs_balanced(),"successful retry did not drain readers");
    for(auto & h:handles) h.release(h.context);
    require(stats->destroyed.load()==3,"publication controls leaked source contexts");
    close(fd);unlink(path.c_str());
}
}
int main(int argc,char **argv) try {
    require(argc==2,"expected isolated fixture directory");
    MemraCpuQueueTestAccess::allocation_failure_preserves_prefix();
    scoped_publication_failure(argv[1]);
    std::puts("QUEUE_PUBLICATION_PASS actual_allocator_failure=1 prior_prefix_preserved=1 mirrored_pair_atomic=1 demand_no_unpublished_drain=1 prefetch_claim_charge_reader_balance=1");
    return 0;
} catch(const std::exception & e) {std::fprintf(stderr,"QUEUE_PUBLICATION_FAIL: %s\n",e.what());return 1;}
