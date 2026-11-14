#include <acl/acl.h>
#include <tiling/platform/platform_ascendc.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "shmem.h"

#include "ascend_deepep/kernels/launch.h"
#include "kernels/mega_moe/include/mega_moe_quant_constants.h"
#include "kernels/mega_moe/include/mega_moe_tiling.h"
#include "kernels/mega_moe/include/mega_moe_tiling_host.h"
#include "kernels/mega_moe_bf16/include/mega_moe_training_workspace.h"
#if defined(DEEPEP_MEGA_MOE_STUB_BF16_UDMA)
#include "kernels/mega_moe_bf16_udma/include/mega_moe_bf16_udma_tiling_host.h"
#endif

namespace {

constexpr uint64_t kDefaultLocalMemBytes = 1024ULL * 1024ULL * 1024ULL;
aclshmemx_uniqueid_t g_default_flag_uid{};

struct MegaMoeContext {
    uint32_t ep_rank_id;
    uint32_t rank_size_per_server;
    uint64_t kfc_context_addr;
    uint64_t ep_buffer[1024];
};
static_assert(sizeof(MegaMoeContext) == 8208, "MegaMoe context ABI changed");

struct TestCase {
    std::string case_id;
    uint32_t m = 0;
    uint32_t k = 0;
    uint32_t n = 0;
    uint32_t topk = 0;
    uint32_t expert_per_rank = 0;
    uint32_t ep = 0;
    uint32_t max_recv_tokens = 0;
    uint64_t local_mem_bytes = kDefaultLocalMemBytes;
    int warmup = 0;
    int loops = 1;
    std::string data_dir;
};

std::string Trim(std::string value)
{
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return {};
    }
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

std::vector<std::string> SplitCsvLine(const std::string &line)
{
    std::vector<std::string> fields;
    std::string current;
    bool quoted = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char ch = line[i];
        if (ch == '"') {
            if (quoted && i + 1 < line.size() && line[i + 1] == '"') {
                current.push_back('"');
                ++i;
            } else {
                quoted = !quoted;
            }
        } else if (ch == ',' && !quoted) {
            fields.push_back(Trim(current));
            current.clear();
        } else {
            current.push_back(ch);
        }
    }
    if (quoted) {
        throw std::runtime_error("unterminated quoted CSV field");
    }
    fields.push_back(Trim(current));
    return fields;
}

uint64_t ParseUnsigned(
    const std::map<std::string, std::string> &row,
    const std::string &name,
    uint64_t default_value = 0,
    bool required = true)
{
    const auto it = row.find(name);
    if (it == row.end() || it->second.empty()) {
        if (required) {
            throw std::runtime_error("CSV column '" + name + "' is required");
        }
        return default_value;
    }
    std::size_t consumed = 0;
    const uint64_t value = std::stoull(it->second, &consumed);
    if (consumed != it->second.size()) {
        throw std::runtime_error("invalid integer in CSV column '" + name + "'");
    }
    return value;
}

TestCase ReadCase(const std::string &csv_path, const std::string &selected_id)
{
    std::ifstream input(csv_path);
    if (!input) {
        throw std::runtime_error("cannot open CSV file: " + csv_path);
    }
    std::string line;
    if (!std::getline(input, line)) {
        throw std::runtime_error("CSV file is empty: " + csv_path);
    }
    const auto headers = SplitCsvLine(line);
    while (std::getline(input, line)) {
        if (Trim(line).empty() || Trim(line).front() == '#') {
            continue;
        }
        const auto fields = SplitCsvLine(line);
        if (fields.size() != headers.size()) {
            throw std::runtime_error("CSV row has a different number of fields than its header");
        }
        std::map<std::string, std::string> row;
        for (std::size_t i = 0; i < headers.size(); ++i) {
            row.emplace(Trim(headers[i]), fields[i]);
        }
        const std::string case_id =
            row.count("case_id") != 0 ? row.at("case_id") : std::string("0");
        if (!selected_id.empty() && selected_id != case_id) {
            continue;
        }

        TestCase result;
        result.case_id = case_id;
        result.m = static_cast<uint32_t>(ParseUnsigned(row, "M"));
        result.k = static_cast<uint32_t>(ParseUnsigned(row, "K"));
        result.n = static_cast<uint32_t>(ParseUnsigned(row, "N"));
        result.topk = static_cast<uint32_t>(ParseUnsigned(row, "TOPK"));
        result.expert_per_rank =
            static_cast<uint32_t>(ParseUnsigned(row, "expertPerRank"));
        result.ep = static_cast<uint32_t>(ParseUnsigned(row, "EP"));
        result.max_recv_tokens = static_cast<uint32_t>(ParseUnsigned(
            row,
            "maxRecvTokenNum",
            static_cast<uint64_t>(result.m) * result.ep *
                std::min(result.expert_per_rank, result.topk),
            false));
        result.local_mem_bytes = ParseUnsigned(
            row, "localMemBytes", kDefaultLocalMemBytes, false);
        result.warmup = static_cast<int>(ParseUnsigned(row, "warmup", 0, false));
        result.loops = static_cast<int>(ParseUnsigned(row, "loops", 1, false));
        if (row.count("dataDir") != 0) {
            result.data_dir = row.at("dataDir");
        }
        return result;
    }
    throw std::runtime_error(
        selected_id.empty() ? "CSV contains no data rows"
                            : "case_id '" + selected_id + "' was not found");
}

void CheckAcl(aclError status, const char *operation)
{
    if (status != ACL_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + " failed, ACL error " +
            std::to_string(status));
    }
}

void CheckShmem(int status, const char *operation)
{
    if (status != 0) {
        throw std::runtime_error(
            std::string(operation) + " failed, SHMEM error " +
            std::to_string(status));
    }
}

bool LogCleanupError(int status, const char *operation) noexcept
{
    if (status != 0) {
        std::cerr << "[ERROR] " << operation
                  << " failed during cleanup, error " << status << std::endl;
        return false;
    }
    return true;
}

void FillShmemAttributes(
    int rank,
    int world_size,
    uint64_t local_mem_bytes,
    const std::string &ip_port,
    aclshmemx_init_attr_t *attributes)
{
    std::memset(attributes, 0, sizeof(*attributes));
    if (ip_port.empty() || ip_port.size() >= ACLSHMEM_MAX_IP_PORT_LEN) {
        throw std::invalid_argument("invalid SHMEM ip_port");
    }
    std::copy(ip_port.begin(), ip_port.end(), attributes->ip_port);
    attributes->my_pe = rank;
    attributes->n_pes = world_size;
    attributes->local_mem_size = local_mem_bytes;
    attributes->option_attr = {
        (1 << 16) + static_cast<int>(sizeof(aclshmemx_init_attr_t)),
#if defined(DEEPEP_MEGA_MOE_STUB_BF16_UDMA)
        static_cast<data_op_engine_type_t>(
            ACLSHMEM_DATA_OP_UDMA | ACLSHMEM_DATA_OP_SDMA),
#else
        ACLSHMEM_DATA_OP_UDMA,
#endif
        DEFAULT_TIMEOUT,
        DEFAULT_TIMEOUT,
        DEFAULT_TIMEOUT};
    attributes->comm_args = reinterpret_cast<void *>(&g_default_flag_uid);
}

bool ReadBinary(const std::string &path, void *buffer, std::size_t bytes)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || static_cast<std::size_t>(input.tellg()) != bytes) {
        return false;
    }
    input.seekg(0);
    input.read(static_cast<char *>(buffer), bytes);
    return input.good();
}

void WriteBinary(const std::string &path, const void *buffer, std::size_t bytes)
{
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot create output file: " + path);
    }
    output.write(static_cast<const char *>(buffer), bytes);
    if (!output.good()) {
        throw std::runtime_error("failed to write output file: " + path);
    }
}

class Runner {
public:
    Runner(int rank, TestCase test_case, std::string ip_port)
        : rank_(rank), test_case_(std::move(test_case)), ip_port_(std::move(ip_port))
    {
    }

    ~Runner()
    {
        Cleanup();
    }

    void Run()
    {
        Validate();
        InitRuntime();
        MakeTiling();
        Allocate();
        InitializeInputs();
        CopyInputs();

        CheckAcl(aclrtSynchronizeStream(stream_), "aclrtSynchronizeStream");
        aclshmem_barrier_all();
        const uint64_t ffts_addr = shmemx_get_ffts_config();
        for (int iter = 0; iter < test_case_.warmup; ++iter) {
            LaunchKernel(ffts_addr);
        }

        const auto begin = std::chrono::steady_clock::now();
        for (int iter = 0; iter < test_case_.loops; ++iter) {
            LaunchKernel(ffts_addr);
        }
        const auto end = std::chrono::steady_clock::now();
        const double elapsed_us =
            std::chrono::duration<double, std::micro>(end - begin).count();

        CopyAndWriteOutputs();
        aclshmem_barrier_all();
        std::cout << "[PERF] MegaMoe BF16 case=" << test_case_.case_id
                  << " rank=" << rank_
                  << " loops=" << test_case_.loops
                  << " average_us=" << elapsed_us / test_case_.loops
                  << std::endl;
        if (!Cleanup()) {
            throw std::runtime_error(
                "MegaMoe BF16 resource cleanup failed; see cleanup errors above");
        }
        std::cout << "[SUCCESS] MegaMoe BF16 case=" << test_case_.case_id
                  << " rank=" << rank_ << "/" << test_case_.ep
                  << " workspace=" << workspace_bytes_
                  << " training_saved_state=" << training_saved_state_bytes_
                  << " peer=" << peer_bytes_ << std::endl;
    }

private:
    void LaunchKernel(uint64_t ffts_addr)
    {
#if defined(DEEPEP_MEGA_MOE_STUB_BF16_UDMA)
        mega_moe_bf16_udma_kernel_do(
            block_num_, stream_, Device(0), Device(1), Device(2), Device(3),
            Device(4), Device(5), nullptr, Device(6), Device(7),
            static_cast<uint8_t *>(training_saved_state_device_),
            static_cast<uint8_t *>(workspace_device_),
            reinterpret_cast<const uint8_t *>(&tiling_), ffts_addr);
#else
        mega_moe_bf16_kernel_do(
            block_num_, stream_, Device(0), Device(1), Device(2), Device(3),
            Device(4), Device(5), nullptr, Device(6), Device(7),
            static_cast<uint8_t *>(training_saved_state_device_),
            nullptr,
            static_cast<uint8_t *>(workspace_device_),
            reinterpret_cast<const uint8_t *>(&tiling_), ffts_addr);
#endif
        CheckAcl(
            aclrtSynchronizeStream(stream_),
            "aclrtSynchronizeStream(kernel)");
        aclshmem_barrier_all();
    }

    void CopyAndWriteOutputs()
    {
        if (test_case_.data_dir.empty()) {
            return;
        }
        for (std::size_t i = 6; i <= 7; ++i) {
            CheckAcl(
                aclrtMemcpy(
                    host_[i], sizes_[i], device_[i], sizes_[i],
                    ACL_MEMCPY_DEVICE_TO_HOST),
                "aclrtMemcpy(D2H output)");
        }
        const std::string suffix =
            std::to_string(test_case_.m) + "_" +
            std::to_string(test_case_.k) + "_" +
            std::to_string(test_case_.n) + "_" +
            std::to_string(test_case_.ep) + ".bin";
        WriteBinary(
            test_case_.data_dir + "/rank" + std::to_string(rank_) +
                "_output_y_bf16_" + suffix,
            host_[6], sizes_[6]);
        WriteBinary(
            test_case_.data_dir + "/rank" + std::to_string(rank_) +
                "_output_expert_token_nums_bf16_" + suffix,
            host_[7], sizes_[7]);
    }

    void Validate()
    {
        if (test_case_.ep < 2 || rank_ < 0 ||
            rank_ >= static_cast<int>(test_case_.ep)) {
            throw std::invalid_argument("invalid rank/EP in CSV");
        }
        if (test_case_.m == 0 || test_case_.k == 0 || test_case_.n == 0 ||
            test_case_.topk == 0 || test_case_.expert_per_rank == 0 ||
            test_case_.max_recv_tokens == 0 || test_case_.loops <= 0) {
            throw std::invalid_argument(
                "MegaMoe BF16 CSV dimensions and loops must be positive");
        }
        if (test_case_.k % 256 != 0 || test_case_.n % 512 != 0) {
            throw std::invalid_argument(
                "MegaMoe BF16 requires K divisible by 256 and N by 512");
        }
#if defined(DEEPEP_MEGA_MOE_STUB_BF16_UDMA)
        if (test_case_.ep > 768 || test_case_.expert_per_rank > 16 ||
            test_case_.ep * test_case_.expert_per_rank > 1024 ||
#else
        if (test_case_.ep > MegaMoeBF16Training::kMaxEpRanks ||
            test_case_.expert_per_rank > MegaMoeBF16Training::kMaxLocalExperts ||
            test_case_.ep * test_case_.expert_per_rank >
                MegaMoeBF16Training::kMaxTotalExperts ||
#endif
            test_case_.topk >
                test_case_.ep * test_case_.expert_per_rank) {
            throw std::invalid_argument("invalid expert count or TOPK");
        }
        if (test_case_.local_mem_bytes % (2ULL * 1024ULL * 1024ULL) != 0) {
            throw std::invalid_argument("localMemBytes must be 2MB aligned");
        }
    }

    void InitRuntime()
    {
        CheckAcl(aclInit(nullptr), "aclInit");
        acl_initialized_ = true;
        CheckAcl(aclrtSetDevice(rank_), "aclrtSetDevice");
        device_set_ = true;
        CheckAcl(aclrtCreateStream(&stream_), "aclrtCreateStream");

        aclshmemx_init_attr_t attributes{};
        FillShmemAttributes(
            rank_, test_case_.ep, test_case_.local_mem_bytes, ip_port_,
            &attributes);
        CheckShmem(
            aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes),
            "aclshmemx_init_attr");
        shmem_initialized_ = true;
    }

    void MakeTiling()
    {
        auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
        block_num_ = platform->GetCoreNumAic();
        const uint32_t aiv_num = platform->GetCoreNumAiv();
        uint64_t ub_size = 0;
        platform->GetCoreMemSize(
            platform_ascendc::CoreMemType::UB, ub_size);
        if (block_num_ == 0 || aiv_num == 0 || ub_size == 0) {
            throw std::runtime_error("failed to query AscendC platform resources");
        }
#if defined(DEEPEP_MEGA_MOE_STUB_BF16_UDMA)
        if (test_case_.ep > block_num_ * 10) {
            throw std::runtime_error(
                "EP exceeds the kernel dispatch bookkeeping capacity");
        }
#endif
        const uint32_t blocks_per_ep =
            std::max<uint32_t>(1, block_num_ / test_case_.ep);
        if (!optiling::SetInitTilingData<
                uint16_t,
                DISPATCH_QUANT_MODE_NO_QUANT>(
                tiling_, test_case_.m, test_case_.k, test_case_.n,
                test_case_.topk, test_case_.ep, test_case_.expert_per_rank,
                blocks_per_ep, test_case_.max_recv_tokens,
                static_cast<uint32_t>(rank_), 2, 1, aiv_num, ub_size)) {
            throw std::runtime_error("failed to generate MegaMoe BF16 tiling");
        }
        const uint64_t savedQuantUbBytes = MegaMoeBF16Training::SaveGateUpClipMasks(
            tiling_.swigluLimit, tiling_.savedGateUpMode)
            ? MegaMoeBF16Training::SavedGateUpQuantUbLayout::totalBytesWithClipMasks
            : MegaMoeBF16Training::SavedGateUpQuantUbLayout::totalBytes;
        if (MegaMoeBF16Training::SaveGateUpAsMxFp8(tiling_.swigluLimit, tiling_.savedGateUpMode) &&
            savedQuantUbBytes > std::min<uint64_t>(ub_size, 248ULL * 1024ULL)) {
            throw std::runtime_error("saved MXFP8 epilogue exceeds device UB capacity");
        }
        workspace_bytes_ = optiling::CalcWorkspaceSize(tiling_).totalSize;
        peer_bytes_ = optiling::CalcPeermemSize(tiling_).totalSize;
        const MegaMoeBF16Training::TrainingSavedStateLayout savedStateLayout(
            test_case_.m, test_case_.k, test_case_.topk, test_case_.ep,
            test_case_.expert_per_rank, test_case_.max_recv_tokens, test_case_.n, MegaMoeBF16Training::SaveGateUpAsMxFp8(tiling_.swigluLimit, tiling_.savedGateUpMode), tiling_.savedGateUpMode != MegaMoeBF16Training::kSaveNone, MegaMoeBF16Training::SaveGateUpClipMasks(tiling_.swigluLimit, tiling_.savedGateUpMode));
        training_saved_state_bytes_ = savedStateLayout.totalBytes;
        training_saved_state_init_bytes_ = savedStateLayout.legacyBytes;
#if !defined(DEEPEP_MEGA_MOE_STUB_BF16_UDMA)
        tiling_.compactTokenCountExchange =
            MegaMoeBF16Training::UseCompactTokenCountExchangeByDefault(
                test_case_.ep, test_case_.expert_per_rank) ? 1U : 0U;
        const MegaMoeBF16Training::ForwardDispatchUbLayout dispatchUb(
            test_case_.ep, test_case_.expert_per_rank, test_case_.k, 2);
        if (dispatchUb.totalBytes > ub_size) {
            throw std::runtime_error("MegaMoe BF16 Dispatch exceeds device UB capacity");
        }
        const MegaMoeBF16Training::ForwardDispatchTileReadyLayout
            transientWorkspaceLayout(
                test_case_.n, test_case_.expert_per_rank,
                test_case_.max_recv_tokens);
        workspace_bytes_ = transientWorkspaceLayout.totalBytes;
        const MegaMoeBF16Training::CompactTokenCountExchangeLayout countLayout(
            test_case_.ep, test_case_.expert_per_rank,
            tiling_.compactTokenCountExchange != 0);
        peer_bytes_ += countLayout.tokenCountsOffset -
            MegaMoeBF16Training::kPeerBaseDataOffset;
        peer_bytes_ +=
            optiling::CalcBf16EarlyTopkWeightPeermemSize(tiling_);
#endif
#if defined(DEEPEP_MEGA_MOE_STUB_BF16_UDMA)
        const auto udmaExtra = MegaMoeBF16UDMAHost::CalcExtraBufferSize(
            test_case_.m, test_case_.topk,
            test_case_.ep, test_case_.expert_per_rank,
            test_case_.max_recv_tokens, test_case_.k);
        const MegaMoeBF16Training::UDMAForwardTransientWorkspaceLayout
            transientWorkspaceLayout(
                test_case_.n, test_case_.expert_per_rank,
                test_case_.max_recv_tokens);
        workspace_bytes_ = transientWorkspaceLayout.totalBytes;
        peer_bytes_ += udmaExtra.peerBytes;
#endif
        if (peer_bytes_ > test_case_.local_mem_bytes) {
            throw std::runtime_error(
                "localMemBytes is smaller than required BF16 peer memory");
        }
    }

    void Allocate()
    {
        const std::size_t m = test_case_.m;
        const std::size_t k = test_case_.k;
        const std::size_t n = test_case_.n;
        const std::size_t topk = test_case_.topk;
        const std::size_t experts = test_case_.expert_per_rank;
        sizes_ = {
            sizeof(MegaMoeContext),
            m * k * sizeof(uint16_t),
            m * topk * sizeof(int32_t),
            m * topk * sizeof(uint16_t),
            experts * n * k * sizeof(uint16_t),
            experts * k * (n / 2) * sizeof(uint16_t),
            m * k * sizeof(uint16_t),
            experts * sizeof(int32_t)};
        host_.resize(sizes_.size(), nullptr);
        device_.resize(sizes_.size(), nullptr);
        for (std::size_t i = 0; i < sizes_.size(); ++i) {
            CheckAcl(aclrtMallocHost(&host_[i], sizes_[i]), "aclrtMallocHost");
            CheckAcl(
                aclrtMalloc(
                    &device_[i], sizes_[i], ACL_MEM_MALLOC_HUGE_FIRST),
                "aclrtMalloc");
        }
        CheckAcl(
            aclrtMalloc(
                &workspace_device_, workspace_bytes_,
                ACL_MEM_MALLOC_HUGE_FIRST),
            "aclrtMalloc(workspaceDevice)");
        CheckAcl(
            aclrtMalloc(
                &training_saved_state_device_, training_saved_state_bytes_,
                ACL_MEM_MALLOC_HUGE_FIRST),
            "aclrtMalloc(trainingSavedStateDevice)");
        peer_mem_device_ = aclshmem_malloc(peer_bytes_);
        if (peer_mem_device_ == nullptr) {
            throw std::runtime_error(
                "aclshmem_malloc(peerMemDevice) returned nullptr");
        }
    }

    void InitializeInputs()
    {
        for (std::size_t i = 0; i < host_.size(); ++i) {
            std::memset(host_[i], 0, sizes_[i]);
        }
        auto *context = static_cast<MegaMoeContext *>(host_[0]);
        context->ep_rank_id = rank_;
        context->rank_size_per_server = test_case_.ep;
        for (uint32_t peer = 0; peer < test_case_.ep; ++peer) {
            void *address = aclshmem_ptr(peer_mem_device_, peer);
            if (address == nullptr) {
                throw std::runtime_error(
                    "aclshmem_ptr failed for peer " + std::to_string(peer));
            }
            context->ep_buffer[peer] = reinterpret_cast<uint64_t>(address);
        }

        if (!test_case_.data_dir.empty()) {
            const std::vector<std::string> names = {
                "", "input_x_bf16", "input_expert_ids_bf16",
                "input_expert_weights_bf16", "weight1_bf16",
                "weight2_bf16"};
            const std::string suffix =
                std::to_string(test_case_.m) + "_" +
                std::to_string(test_case_.k) + "_" +
                std::to_string(test_case_.n) + "_" +
                std::to_string(test_case_.ep) + ".bin";
            for (std::size_t i = 1; i < names.size(); ++i) {
                const std::string path =
                    test_case_.data_dir + "/rank" + std::to_string(rank_) +
                    "_" + names[i] + "_" + suffix;
                if (!ReadBinary(path, host_[i], sizes_[i])) {
                    throw std::runtime_error(
                        "input file not found or size mismatch: " + path);
                }
            }
        }
    }

    void CopyInputs()
    {
        for (std::size_t i = 0; i <= 5; ++i) {
            CheckAcl(
                aclrtMemcpy(
                    device_[i], sizes_[i], host_[i], sizes_[i],
                    ACL_MEMCPY_HOST_TO_DEVICE),
                "aclrtMemcpy(H2D)");
        }
        for (std::size_t i = 6; i < device_.size(); ++i) {
            CheckAcl(
                aclrtMemset(device_[i], sizes_[i], 0, sizes_[i]),
                "aclrtMemset(device output)");
        }
        CheckAcl(
            aclrtMemset(
                workspace_device_, workspace_bytes_, 0, workspace_bytes_),
            "aclrtMemset(workspaceDevice)");
        CheckAcl(
            aclrtMemset(
                training_saved_state_device_, training_saved_state_bytes_, 0,
                training_saved_state_init_bytes_),
            "aclrtMemset(trainingSavedStateDevice)");
        CheckAcl(
            aclrtMemset(peer_mem_device_, peer_bytes_, 0, peer_bytes_),
            "aclrtMemset(peerMemDevice)");
    }

    uint8_t *Device(std::size_t index)
    {
        return static_cast<uint8_t *>(device_.at(index));
    }

    bool Cleanup() noexcept
    {
        bool succeeded = true;
        if (peer_mem_device_ != nullptr) {
            aclshmem_free(peer_mem_device_);
            peer_mem_device_ = nullptr;
        }
        if (workspace_device_ != nullptr) {
            succeeded = LogCleanupError(
                            aclrtFree(workspace_device_),
                            "aclrtFree(workspaceDevice)") &&
                        succeeded;
            workspace_device_ = nullptr;
        }
        if (training_saved_state_device_ != nullptr) {
            succeeded = LogCleanupError(
                            aclrtFree(training_saved_state_device_),
                            "aclrtFree(trainingSavedStateDevice)") &&
                        succeeded;
            training_saved_state_device_ = nullptr;
        }
        for (void *ptr : device_) {
            if (ptr != nullptr) {
                succeeded = LogCleanupError(
                                aclrtFree(ptr), "aclrtFree(device buffer)") &&
                            succeeded;
            }
        }
        for (void *ptr : host_) {
            if (ptr != nullptr) {
                succeeded = LogCleanupError(
                                aclrtFreeHost(ptr),
                                "aclrtFreeHost(host buffer)") &&
                            succeeded;
            }
        }
        device_.clear();
        host_.clear();
        if (stream_ != nullptr) {
            succeeded = LogCleanupError(
                            aclrtDestroyStream(stream_),
                            "aclrtDestroyStream") &&
                        succeeded;
            stream_ = nullptr;
        }
        if (shmem_initialized_) {
            succeeded = LogCleanupError(
                            aclshmem_finalize(), "aclshmem_finalize") &&
                        succeeded;
            shmem_initialized_ = false;
        }
        if (device_set_) {
            succeeded = LogCleanupError(
                            aclrtResetDevice(rank_), "aclrtResetDevice") &&
                        succeeded;
            device_set_ = false;
        }
        if (acl_initialized_) {
            succeeded = LogCleanupError(aclFinalize(), "aclFinalize") &&
                        succeeded;
            acl_initialized_ = false;
        }
        return succeeded;
    }

    int rank_;
    TestCase test_case_;
    std::string ip_port_;
    MegaMoeTilingData tiling_{};
    uint32_t block_num_ = 0;
    std::size_t workspace_bytes_ = 0;
    std::size_t training_saved_state_bytes_ = 0;
    std::size_t training_saved_state_init_bytes_ = 0;
    std::size_t peer_bytes_ = 0;
    aclrtStream stream_ = nullptr;
    void *workspace_device_ = nullptr;
    void *training_saved_state_device_ = nullptr;
    void *peer_mem_device_ = nullptr;
    std::vector<std::size_t> sizes_;
    std::vector<void *> host_;
    std::vector<void *> device_;
    bool acl_initialized_ = false;
    bool device_set_ = false;
    bool shmem_initialized_ = false;
};

}  // namespace

int main(int argc, char **argv)
{
    if (argc < 3 || argc > 6) {
        std::cerr
#if defined(DEEPEP_MEGA_MOE_STUB_BF16_UDMA)
            << "Usage: mega_moe_bf16_udma_stub <rank> <shmem_ip_port> "
#else
            << "Usage: mega_moe_bf16_stub <rank> <shmem_ip_port> "
#endif
               "[csv_path] [case_id] [data_dir]\n";
        return 1;
    }
    try {
        const int rank = std::stoi(argv[1]);
        const std::string csv_path =
            argc >= 4 ? argv[3]
                      : "test/megamoe_bf16_test/mega_moe_cases.csv";
        const std::string case_id = argc >= 5 ? argv[4] : "";
        TestCase test_case = ReadCase(csv_path, case_id);
        if (argc >= 6) {
            test_case.data_dir = argv[5];
        }
        Runner(rank, std::move(test_case), argv[2]).Run();
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "[ERROR] " << error.what() << std::endl;
        return 1;
    }
}
