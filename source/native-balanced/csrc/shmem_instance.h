#pragma once

#include <acl/acl.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <torch/custom_class.h>

// Distinct from npu_patch's global ShmemInstance. Both are CustomClassHolder
// with the same class name; a shared type_info makes torch cast one into the
// other when MoonEP is loaded first.
namespace deepep {

class ShmemInstance : public torch::jit::CustomClassHolder {
public:
    ShmemInstance(
        int64_t rank,
        int64_t world_size,
        std::string ip_port);
    ShmemInstance(
        int64_t rank,
        int64_t world_size,
        std::string ip_port,
        int64_t data_op_engine_type);

    ~ShmemInstance();

    ShmemInstance(const ShmemInstance&) = delete;
    ShmemInstance& operator=(const ShmemInstance&) = delete;

    void Init();
    void Init(int64_t local_mem_size);
    // Use a SHMEM runtime that another library already initialized. Does not
    // call aclshmemx_init_attr and does not finalize that runtime on destroy.
    void Attach();
    void SetDataOpEngineType(int64_t data_op_engine_type);
    void Barrier();
    void WaitHandle(aclrtStream stream) const;
    void Synchronize(aclrtStream stream) const;
    void Destroy();

    uint64_t FftsAddr() const;
    bool Initialized() const;
    bool IsAvailable() const;
    int64_t Rank() const;
    int64_t WorldSize() const;

private:
    void InitRuntime();
    void DestroyNoThrow() noexcept;

    int rank_;
    int world_size_;
    uint64_t local_mem_size_;
    std::string ip_port_;
    int data_op_engine_type_;
    bool initialized_;
    bool acquired_global_runtime_;
    bool borrowed_;

    uint64_t ffts_addr_;
};

}  // namespace deepep

using ShmemInstance = deepep::ShmemInstance;
