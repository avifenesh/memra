#include "llama.h"
#include "common.h"
#include "nlohmann/json.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <filesystem>
#include <cstring>
#include "ggml-backend.h"
using json = nlohmann::json;
struct Trace {
    std::filesystem::path dir;
    int sample = -1;
    int chunk = 0;
    int count = 0;
};
static bool trace_eval(ggml_tensor * tensor, bool ask, void * data) {
    auto & trace = *static_cast<Trace *>(data);
    std::string name = ggml_get_name(tensor);
    bool wanted = name.rfind("l_out-", 0) == 0 || name == "attn_norm-0" ||
                  name == "Qcur-0" || name == "Kcur-0" || name == "Vcur-0" ||
                  name == "attn_post_norm-0";
    wanted = wanted && trace.sample >= 0 && tensor->type == GGML_TYPE_F32 &&
             tensor->ne[2] == 1 && tensor->ne[3] == 1 && tensor->nb[0] == sizeof(float);
    if (ask) return wanted;
    if (!wanted) return true;
    std::vector<float> values(tensor->ne[0]);
    ggml_backend_tensor_get(tensor, values.data(), (tensor->ne[1]-1)*tensor->nb[1], values.size()*sizeof(float));
    std::string file = std::to_string(trace.sample)+"-"+std::to_string(trace.chunk)+"-"+std::to_string(trace.count++)+"-"+name+".f32";
    std::ofstream raw(trace.dir/file, std::ios::binary);
    raw.write(reinterpret_cast<const char *>(values.data()), values.size()*sizeof(float));
    if (!raw) std::abort();
    std::ofstream meta(trace.dir/"index.jsonl", std::ios::app);
    meta << json({{"i",trace.sample},{"chunk",trace.chunk},{"name",name},{"rows",tensor->ne[1]},
                  {"width",tensor->ne[0]},{"file",file}}).dump() << "\n";
    return true;
}
int main(int argc, char ** argv) {
    if (argc != 3 && argc != 4) return 64;
    Trace trace;
    if (argc == 4) { trace.dir = argv[3]; std::filesystem::create_directories(trace.dir); }
    llama_backend_init();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = 99;
    auto cp = llama_context_default_params();
    std::string mode = argv[2];
    if (mode != "aligned" && mode != "split4" && mode != "output-limit" &&
        mode != "common-init" && mode != "common" && mode != "api-defaults") return 64;
    cp.n_ctx = 8192; cp.n_batch = 4096; cp.n_ubatch = 2048;
    cp.n_threads = 2; cp.n_threads_batch = 2; cp.swa_full = false;
    if (mode == "common") {
        common_params p; p.n_ctx = 8192; p.n_batch = 4096; p.n_ubatch = 2048;
        p.n_parallel = 1; p.cpuparams.n_threads = 2; p.cpuparams_batch.n_threads = 2;
        cp = common_context_params_to_llama(p);
    }
    if (mode == "api-defaults") { cp.n_batch = 8192; cp.swa_full = true; }
    if (argc == 4) { cp.cb_eval = trace_eval; cp.cb_eval_user_data = &trace; }
    common_init_result_ptr common_init;
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    if (mode == "common-init") {
        common_params p; p.model.path = argv[1]; p.n_gpu_layers = 99; p.n_ctx = 8192;
        p.n_batch = 4096; p.n_ubatch = 2048; p.n_parallel = 1;
        p.n_outputs_max = 1; p.n_outputs_max_per_seq = 1;
        p.cpuparams.n_threads = 2; p.cpuparams_batch.n_threads = 2; p.warmup = false;
        if (argc == 4) { p.cb_eval = trace_eval; p.cb_eval_user_data = &trace; }
        common_init = common_init_from_params(p);
        model = common_init->model(); ctx = common_init->context();
    } else {
        if (mode == "output-limit") cp.n_outputs_max = 1;
        model = llama_model_load_from_file(argv[1], mp); if (!model) return 1;
        ctx = llama_init_from_model(model, cp);
    }
    if (!ctx) return 2;
    const auto * vocab = llama_model_get_vocab(model);
    std::string line;
    while (std::getline(std::cin, line)) {
        auto x = json::parse(line); std::string prompt = x["request"]["prompt"];
        trace.sample = x["i"]; trace.chunk = 0;
        int n = -llama_tokenize(vocab, prompt.data(), prompt.size(), nullptr, 0, true, true);
        std::vector<llama_token> tokens(n);
        n = llama_tokenize(vocab, prompt.data(), prompt.size(), tokens.data(), tokens.size(), true, true);
        if (n < 0) return 3; tokens.resize(n);
        llama_memory_clear(llama_get_memory(ctx), true);
        if (mode == "split4") {
            if (n <= 4 || n > 2048) return 65;
            for (int part = 0; part < 2; ++part) {
                trace.chunk = part;
                int begin = part == 0 ? 0 : n - 4;
                int end = part == 0 ? n - 4 : n;
                auto batch = llama_batch_init(end - begin, 0, 1);
                batch.n_tokens = end - begin;
                for (int j = begin; j < end; ++j) {
                    int k = j - begin;
                    batch.token[k] = tokens[j]; batch.pos[k] = j;
                    batch.n_seq_id[k] = 1; batch.seq_id[k][0] = 0;
                    batch.logits[k] = j == n - 1;
                }
                int rc = llama_decode(ctx, batch); llama_batch_free(batch);
                if (rc) return 4;
            }
        } else {
            auto batch = llama_batch_get_one(tokens.data(), tokens.size());
            if (llama_decode(ctx, batch)) return 4;
        }
        const float * logits = llama_get_logits_ith(ctx, -1);
        if (const char * dump = std::getenv("ORACLE_LOGITS_DIR")) {
            std::filesystem::create_directories(dump);
            std::ofstream raw(std::filesystem::path(dump)/(std::to_string(trace.sample)+"-logits.f32"), std::ios::binary);
            raw.write(reinterpret_cast<const char *>(logits), llama_vocab_n_tokens(vocab)*sizeof(float));
            if (!raw) return 5;
        }
        std::vector<int> ids(llama_vocab_n_tokens(vocab)); std::iota(ids.begin(), ids.end(), 0);
        std::partial_sort(ids.begin(), ids.begin()+5, ids.end(), [&](int a, int b){return logits[a]>logits[b];});
        json top = json::array();
        for (int j=0;j<5;++j) top.push_back({{"id",ids[j]},{"logit",logits[ids[j]]}});
        std::cout << json({{"i",x["i"]},{"mode",mode},{"tokens",tokens},{"top5",top}}).dump() << std::endl;
    }
    if (common_init) common_init.reset();
    else { llama_free(ctx); llama_model_free(model); }
    llama_backend_free();
}
