#include <acl/acl.h>
#include <tiling/platform/platform_ascendc.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "shmem.h"

#include "ascend_deepep/kernels/launch.h"
#include "kernels/common/timer/AscendTimerV2.hpp"
#include "kernels/mega_moe_w4a8/include/mega_moe_tiling.h"
#include "kernels/mega_moe_w4a8/include/mega_moe_tiling_host.h"

namespace {

constexpr uint64_t kDefaultLocalMemBytes = 1024ULL * 1024ULL * 1024ULL;
constexpr uint16_t kBfloat16One = 0x3f80;
aclshmemx_uniqueid_t g_default_flag_uid{};

struct MegaMoeW4A8Context {
    uint32_t ep_rank_id;
    uint32_t ep_rank_size;
    uint64_t win_size;
    uint64_t ep_buffer[1024];
};
static_assert(sizeof(MegaMoeW4A8Context) == 8208, "MegaMoe context ABI changed");

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
    bool enable_timer = false;
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

std::vector<std::string> SplitCsvLine(const std::string& line)
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
    const std::map<std::string, std::string>& row,
    const std::string& name,
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

TestCase ReadCase(const std::string& csv_path, const std::string& selected_id)
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
            row, "maxRecvTokenNum",
            static_cast<uint64_t>(result.m) * result.topk * 4, false));
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

void CheckAcl(aclError status, const char* operation)
{
    if (status != ACL_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + " failed, ACL error " +
            std::to_string(status));
    }
}

void CheckShmem(int status, const char* operation)
{
    if (status != 0) {
        throw std::runtime_error(
            std::string(operation) + " failed, SHMEM error " +
            std::to_string(status));
    }
}

bool LogCleanupError(int status, const char* operation) noexcept
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
    const std::string& ip_port,
    aclshmemx_init_attr_t* attributes)
{
    std::memset(attributes, 0, sizeof(*attributes));
    if (ip_port.empty() || ip_port.size() >= ACLSHMEM_MAX_IP_PORT_LEN) {
        throw std::invalid_argument("invalid SHMEM ip_port");
    }
    std::copy(ip_port.begin(), ip_port.end(), attributes->ip_port);
    attributes->my_pe = rank;
    attributes->n_pes = world_size;
    attributes->local_mem_size = local_mem_bytes;
    attributes->option_attr.version = (1 << 16) + static_cast<int>(sizeof(aclshmemx_init_attr_t));
    attributes->option_attr.data_op_engine_type = static_cast<data_op_engine_type_t>(
        ACLSHMEM_DATA_OP_UDMA | ACLSHMEM_DATA_OP_SDMA);
    // attributes->option_attr.data_op_engine_type = static_cast<data_op_engine_type_t>(
    //     ACLSHMEM_DATA_OP_SDMA);
    attributes->option_attr.shm_init_timeout = DEFAULT_TIMEOUT;
    attributes->option_attr.shm_create_timeout = DEFAULT_TIMEOUT;
    attributes->option_attr.control_operation_timeout = DEFAULT_TIMEOUT;
    attributes->option_attr.sockFd = -1;
    attributes->comm_args = reinterpret_cast<void*>(&g_default_flag_uid);
}

bool ReadBinary(const std::string& path, void* buffer, std::size_t bytes)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || static_cast<std::size_t>(input.tellg()) != bytes) {
        return false;
    }
    input.seekg(0);
    input.read(static_cast<char*>(buffer), bytes);
    return input.good();
}

void WriteBinary(const std::string& path, const void* buffer, std::size_t bytes)
{
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot create output file: " + path);
    }
    output.write(static_cast<const char*>(buffer), bytes);
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
        // Keep every PE alive until all ranks have finished copying and writing
        // their outputs. Otherwise one rank may tear down its symmetric heap or
        // bootstrap while another rank is still completing the test.
        aclshmem_barrier_all();
        std::cout << "[PERF] MegaMoe W4A8 case=" << test_case_.case_id
                  << " rank=" << rank_
                  << " loops=" << test_case_.loops
                  << " average_us=" << elapsed_us / test_case_.loops
                  << std::endl;
        if (!Cleanup()) {
            throw std::runtime_error(
                "MegaMoe resource cleanup failed; see cleanup errors above");
        }
        std::cout << "[SUCCESS] MegaMoe W4A8 case=" << test_case_.case_id
                  << " rank=" << rank_ << "/" << test_case_.ep
                  << " workspace=" << workspace_bytes_
                  << " peer=" << peer_bytes_ << std::endl;
    }

private:
    void LaunchKernel(uint64_t ffts_addr)
    {
        if (test_case_.enable_timer) {
            CheckAcl(
                aclrtMemset(timer_device_, timer_bytes_, 0, timer_bytes_),
                "aclrtMemset(timer)");
        }
        mega_moe_w4a8_kernel_do(
            block_num_, stream_, Device(0), Device(1), Device(2), Device(3),
            Device(4), Device(5), Device(6), Device(7), Device(8), Device(9),
            test_case_.enable_timer
                ? static_cast<uint8_t*>(timer_device_)
                : nullptr,
            Device(10), Device(11),
            static_cast<uint8_t*>(workspace_device_),
            reinterpret_cast<const uint8_t*>(&tiling_), ffts_addr);
        CheckAcl(aclrtSynchronizeStream(stream_), "aclrtSynchronizeStream(kernel)");
        aclshmem_barrier_all();
    }

    void CopyAndWriteOutputs()
    {
        if (test_case_.data_dir.empty()) {
            return;
        }
        for (std::size_t i = 10; i <= 11; ++i) {
            CheckAcl(
                aclrtMemcpy(
                    host_[i], sizes_[i], device_[i], sizes_[i],
                    ACL_MEMCPY_DEVICE_TO_HOST),
                "aclrtMemcpy(D2H output)");
        }
        if (test_case_.enable_timer) {
            CheckAcl(
                aclrtMemcpy(
                    timer_host_, timer_bytes_, timer_device_, timer_bytes_,
                    ACL_MEMCPY_DEVICE_TO_HOST),
                "aclrtMemcpy(D2H timer)");
        }
        const std::string suffix =
            std::to_string(test_case_.m) + "_" +
            std::to_string(test_case_.k) + "_" +
            std::to_string(test_case_.n) + "_" +
            std::to_string(test_case_.ep) + ".bin";
        WriteBinary(
            test_case_.data_dir + "/rank" + std::to_string(rank_) +
                "_output_y_w4a8_" + suffix,
            host_[10], sizes_[10]);
        WriteBinary(
            test_case_.data_dir + "/rank" + std::to_string(rank_) +
                "_output_expert_token_nums_w4a8_" + suffix,
            host_[11], sizes_[11]);
        if (test_case_.enable_timer) {
            std::ofstream timer_file(
                test_case_.data_dir + "/rank_" + std::to_string(rank_) +
                "_npu_time.csv");
            if (!timer_file.is_open()) {
                throw std::runtime_error("failed to open timer CSV output");
            }
            const auto* timer_values =
                static_cast<const int64_t*>(timer_host_);
            for (int index = 0;
                 index < AscendTimer::TOTAL_BUFFER_SIZE;
                 ++index) {
                timer_file << timer_values[index] << '\n';
            }
        }
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
            throw std::invalid_argument("MegaMoe CSV dimensions and loops must be positive");
        }
        if (test_case_.k % 64 != 0 || test_case_.n % 128 != 0) {
            throw std::invalid_argument("K must be divisible by 64 and N by 128");
        }
        if (test_case_.expert_per_rank > 64 ||
            test_case_.ep * test_case_.expert_per_rank > 1024 ||
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
            rank_, test_case_.ep, test_case_.local_mem_bytes, ip_port_, &attributes);
        CheckShmem(
            aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes),
            "aclshmemx_init_attr");
        shmem_initialized_ = true;
    }

    void MakeTiling()
    {
        auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
        block_num_ = platform->GetCoreNumAic();
        const uint32_t aiv_num = platform->GetCoreNumAiv();
        uint64_t ub_size = 0;
        platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        if (block_num_ == 0 || aiv_num == 0 || ub_size == 0) {
            throw std::runtime_error("failed to query AscendC platform resources");
        }
        // Keep the original standalone W4A8 tiling configuration. Dispatch
        // still emits exactly one completion per source rank.
        constexpr uint32_t blocks_per_ep = 2;
        if (!optiling_w4a8::SetInitTilingData<uint16_t, DISPATCH_QUANT_OUT_TYPE_E4M3FN>(
                tiling_, test_case_.m, test_case_.k, test_case_.n,
                test_case_.topk, test_case_.ep, test_case_.expert_per_rank,
                blocks_per_ep, test_case_.max_recv_tokens,
                static_cast<uint32_t>(rank_), 16, 0, aiv_num, ub_size)) {
            throw std::runtime_error("failed to generate MegaMoe tiling");
        }
        workspace_bytes_ = optiling_w4a8::CalcWorkspaceSize(tiling_).totalSize;
        peer_bytes_ = optiling_w4a8::CalcPeermemSize(tiling_).totalSize;
        if (peer_bytes_ > test_case_.local_mem_bytes) {
            throw std::runtime_error("localMemBytes is smaller than required peer memory");
        }
    }

    void Allocate()
    {
        const std::size_t m = test_case_.m;
        const std::size_t k = test_case_.k;
        const std::size_t n = test_case_.n;
        const std::size_t topk = test_case_.topk;
        const std::size_t experts = test_case_.expert_per_rank;
        const std::size_t max_recv = test_case_.max_recv_tokens;
        sizes_ = {
            sizeof(MegaMoeW4A8Context),
            m * k * sizeof(uint16_t),
            m * topk * sizeof(int32_t),
            m * topk * sizeof(uint16_t),
            experts * n * k / 2,
            experts * k * n / 4,
            experts * n * ((k + 63) / 64) * 2,
            experts * k * (((n / 2) + 63) / 64) * 2,
            m,
            max_recv * sizeof(float),
            m * k * sizeof(uint16_t),
            experts * sizeof(int32_t)};
        host_.resize(sizes_.size(), nullptr);
        device_.resize(sizes_.size(), nullptr);
        for (std::size_t i = 0; i < sizes_.size(); ++i) {
            CheckAcl(aclrtMallocHost(&host_[i], sizes_[i]), "aclrtMallocHost");
            CheckAcl(aclrtMalloc(&device_[i], sizes_[i], ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
        }
        CheckAcl(
            aclrtMalloc(&workspace_device_, workspace_bytes_, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(workspaceDevice)");
        if (test_case_.enable_timer) {
            CheckAcl(
                aclrtMallocHost(&timer_host_, timer_bytes_),
                "aclrtMallocHost(timer)");
            CheckAcl(
                aclrtMalloc(
                    &timer_device_, timer_bytes_, ACL_MEM_MALLOC_HUGE_FIRST),
                "aclrtMalloc(timer)");
        }
        peer_mem_device_ = aclshmem_malloc(peer_bytes_);
        if (peer_mem_device_ == nullptr) {
            throw std::runtime_error("aclshmem_malloc(peerMemDevice) returned nullptr");
        }
    }

    void InitializeInputs()
    {
        for (std::size_t i = 0; i < host_.size(); ++i) {
            if (host_[i] != nullptr) {
                std::memset(host_[i], 0, sizes_[i]);
            }
        }
        auto* context = static_cast<MegaMoeW4A8Context*>(host_[0]);
        context->ep_rank_id = rank_;
        context->ep_rank_size = test_case_.ep;
        context->win_size = peer_bytes_;
        for (uint32_t peer = 0; peer < test_case_.ep; ++peer) {
            void* address = aclshmem_ptr(peer_mem_device_, peer);
            if (address == nullptr) {
                throw std::runtime_error("aclshmem_ptr failed for peer " + std::to_string(peer));
            }
            context->ep_buffer[peer] = reinterpret_cast<uint64_t>(address);
        }

        auto* ids = static_cast<int32_t*>(host_[2]);
        auto* weights = static_cast<uint16_t*>(host_[3]);
        const std::size_t routes =
            static_cast<std::size_t>(test_case_.m) * test_case_.topk;
        for (std::size_t i = 0; i < routes; ++i) {
            ids[i] = static_cast<int32_t>(
                i % (test_case_.ep * test_case_.expert_per_rank));
            weights[i] = kBfloat16One;
        }

        if (!test_case_.data_dir.empty()) {
            const std::vector<std::string> names = {
                "", "input_x_w4a8", "input_expert_ids_w4a8",
                "input_expert_scales_w4a8", "weight1_w4_npu_w4a8",
                "weight2_w4_npu_w4a8", "weight1_w4_npu_scale_w4a8",
                "weight2_w4_npu_scale_w4a8", "input_x_active_mask_w4a8",
                "scales"};
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
                    // Match the original W4A8 standalone harness: active mask
                    // and auxiliary scales are optional and default to zero.
                    if (i <= 7) {
                        throw std::runtime_error(
                            "input file not found or size mismatch: " + path);
                    }
                    std::cout << "[INFO] Optional input unavailable, using "
                              << "zero initialization: " << path << std::endl;
                }
            }
        }
    }

    void CopyInputs()
    {
        for (std::size_t i = 0; i <= 9; ++i) {
            CheckAcl(
                aclrtMemcpy(
                    device_[i], sizes_[i], host_[i], sizes_[i],
                    ACL_MEMCPY_HOST_TO_DEVICE),
                "aclrtMemcpy(H2D)");
        }
        for (std::size_t i = 10; i < device_.size(); ++i) {
            CheckAcl(
                aclrtMemset(device_[i], sizes_[i], 0, sizes_[i]),
                "aclrtMemset(device output)");
        }
        CheckAcl(
            aclrtMemset(
                workspace_device_, workspace_bytes_, 0, workspace_bytes_),
            "aclrtMemset(workspaceDevice)");
        constexpr std::array<uint8_t, 16> fp4_to_fp8 = {
            0, 48, 56, 60, 64, 68, 72, 76,
            128, 176, 184, 188, 192, 196, 200, 204};
        std::array<uint8_t, 512> gamma{};
        for (std::size_t value = 0; value < 256; ++value) {
            gamma[value] = fp4_to_fp8[value & 0x0f];
            gamma[value + 256] = fp4_to_fp8[(value >> 4) & 0x0f];
        }
        CheckAcl(
            aclrtMemcpy(
                workspace_device_, workspace_bytes_, gamma.data(), gamma.size(),
                ACL_MEMCPY_HOST_TO_DEVICE),
            "aclrtMemcpy(H2D gamma)");
        CheckAcl(
            aclrtMemset(
                peer_mem_device_, peer_bytes_, 0, peer_bytes_),
            "aclrtMemset(peerMemDevice)");
    }

    uint8_t* Device(std::size_t index)
    {
        return static_cast<uint8_t*>(device_.at(index));
    }

    bool Cleanup() noexcept
    {
        bool succeeded = true;
        if (timer_device_ != nullptr) {
            succeeded =
                LogCleanupError(aclrtFree(timer_device_), "aclrtFree(timer)") &&
                succeeded;
            timer_device_ = nullptr;
        }
        if (timer_host_ != nullptr) {
            succeeded =
                LogCleanupError(
                    aclrtFreeHost(timer_host_), "aclrtFreeHost(timer)") &&
                succeeded;
            timer_host_ = nullptr;
        }
        if (peer_mem_device_ != nullptr) {
            aclshmem_free(peer_mem_device_);
            peer_mem_device_ = nullptr;
        }
        if (workspace_device_ != nullptr) {
            succeeded =
                LogCleanupError(
                    aclrtFree(workspace_device_),
                    "aclrtFree(workspaceDevice)") &&
                succeeded;
            workspace_device_ = nullptr;
        }
        for (void* ptr : device_) {
            if (ptr != nullptr) {
                succeeded =
                    LogCleanupError(
                        aclrtFree(ptr), "aclrtFree(device buffer)") &&
                    succeeded;
            }
        }
        for (void* ptr : host_) {
            if (ptr != nullptr) {
                succeeded =
                    LogCleanupError(
                        aclrtFreeHost(ptr), "aclrtFreeHost(host buffer)") &&
                    succeeded;
            }
        }
        device_.clear();
        host_.clear();

        // Match the W4A8 runner teardown order: destroy the user stream before
        // finalizing SHMEM, then release the device and ACL runtime.
        if (stream_ != nullptr) {
            succeeded =
                LogCleanupError(
                    aclrtDestroyStream(stream_), "aclrtDestroyStream") &&
                succeeded;
            stream_ = nullptr;
        }
        if (shmem_initialized_) {
            succeeded =
                LogCleanupError(
                    aclshmem_finalize(), "aclshmem_finalize") &&
                succeeded;
            shmem_initialized_ = false;
        }
        if (device_set_) {
            succeeded =
                LogCleanupError(
                    aclrtResetDevice(rank_), "aclrtResetDevice") &&
                succeeded;
            device_set_ = false;
        }
        if (acl_initialized_) {
            succeeded =
                LogCleanupError(aclFinalize(), "aclFinalize") &&
                succeeded;
            acl_initialized_ = false;
        }
        return succeeded;
    }

    int rank_;
    TestCase test_case_;
    std::string ip_port_;
    MegaMoeW4A8TilingData tiling_{};
    uint32_t block_num_ = 0;
    std::size_t workspace_bytes_ = 0;
    std::size_t peer_bytes_ = 0;
    std::size_t timer_bytes_ =
        static_cast<std::size_t>(AscendTimer::TOTAL_BUFFER_SIZE) *
        sizeof(int64_t);
    aclrtStream stream_ = nullptr;
    void* workspace_device_ = nullptr;
    void* timer_device_ = nullptr;
    void* timer_host_ = nullptr;
    void* peer_mem_device_ = nullptr;
    std::vector<std::size_t> sizes_;
    std::vector<void*> host_;
    std::vector<void*> device_;
    bool acl_initialized_ = false;
    bool device_set_ = false;
    bool shmem_initialized_ = false;
};

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 3 || argc > 7) {
        std::cerr
            << "Usage: mega_moe_w4a8_stub <rank> <shmem_ip_port> "
               "[csv_path] [case_id] [data_dir] [enable_timer]\n";
        return 1;
    }
    try {
        const int rank = std::stoi(argv[1]);
        const std::string csv_path =
            argc >= 4 ? argv[3] :
                        "test/megamoe_w4a8_test/mega_moe_cases.csv";
        const std::string case_id = argc >= 5 ? argv[4] : "";
        TestCase test_case = ReadCase(csv_path, case_id);
        if (argc >= 6) {
            test_case.data_dir = argv[5];
        }
        if (argc >= 7) {
            test_case.enable_timer = std::stoi(argv[6]) != 0;
        }
        Runner(rank, std::move(test_case), argv[2]).Run();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[ERROR] " << error.what() << std::endl;
        return 1;
    }
}
