// Diagnostic only: upstream `run_visits` is not graph-recorded. Show a selected *input* position's
// device bytes at each operation boundary, with no tolerance masking the first arithmetic change.
void recon_trace(const char* stage, int visit, int64_t pos, int n, const void* data, size_t bytes,
                 cudaStream_t st) {
    static const int64_t selected = []() -> int64_t {
        const char* value = std::getenv("RECON_TRACE_POS");
        return value ? std::stoll(value) : -1;
    }();
    if (selected < pos || selected >= pos + n) return;
    std::vector<uint8_t> host(bytes);
    const auto* row = static_cast<const uint8_t*>(data) + (size_t) (selected - pos) * bytes;
    ck(cudaMemcpyAsync(host.data(), row, bytes, cudaMemcpyDeviceToHost, st), "probe trace copy");
    ck(cudaStreamSynchronize(st), "probe trace sync");
    uint64_t hash = 14695981039346656037ull;
    for (uint8_t x : host) hash = (hash ^ x) * 1099511628211ull;
    std::fprintf(stderr, "RECON_TRACE %lld %d %s %zu %016llx\n", (long long) selected, visit, stage, bytes,
                 (unsigned long long) hash);
}
