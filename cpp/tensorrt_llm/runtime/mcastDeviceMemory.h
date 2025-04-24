/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "tensorrt_llm/common/mcastDevMemUtils.h"
#include "tensorrt_llm/runtime/ipcNvlsMemory.h"
#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <memory>
#include <vector>

namespace tensorrt_llm::runtime
{
//! Add Doxygen comment for the class
class McastDeviceMemory
{
public:
    // Disallow copy construction
    McastDeviceMemory(McastDeviceMemory const&) = delete;
    McastDeviceMemory& operator=(McastDeviceMemory const&) = delete;

    // // Move construction
    // McastDeviceMemory(McastDeviceMemory&&) noexcept;
    // McastDeviceMemory& operator=(McastDeviceMemory&&) noexcept;

    //! Add Doxygen comment
    McastDeviceMemory(size_t bufSize, uint32_t groupSize, uint32_t groupRank, int deviceIdx, bool mnNvlink);

    //! Add Doxygen comment
    void** getSignalPadPtrsDev()
    {
        return reinterpret_cast<void**>(mSignalPadsDev.data());
    }

    //! Add Doxygen comment
    void** getBufferPtrsDev()
    {
        return reinterpret_cast<void**>(mUcPtrs.data());
    }

    //! Add Doxygen comment
    void* getUnicastPtr(uint32_t rank)
    {
        auto* data_ptr = reinterpret_cast<void*>(mUcPtrs[rank]);
        tensorrt_llm::common::registerMcastDevMemBuffer(data_ptr, this);
        return data_ptr;
    }

    //! Add Doxygen comment
    void* getMulticastPtr()
    {
        auto* data_ptr = reinterpret_cast<void*>(mMcPtr);
        tensorrt_llm::common::registerMcastDevMemBuffer(data_ptr, this);
        return data_ptr;
    }

    //! Add Doxygen comment
    [[nodiscard]] size_t getRank() const
    {
        return mGroupRank;
    }

    //! Add Doxygen comment
    [[nodiscard]] size_t getWorldSize() const
    {
        return mGroupSize;
    }

    ~McastDeviceMemory();

private:
    bool mIsMNNvlink;
    int mDeviceIdx;
    uint32_t mGroupSize, mGroupRank;
    size_t mBufSize;
    size_t mSignalPadOffset;
    size_t mAllocationSize;

    CUdeviceptr mMcPtr;
    std::vector<CUdeviceptr> mUcPtrs;
    std::vector<CUdeviceptr> mSignalPadsDev;
    CUmemGenericAllocationHandle mMcHandle;
    std::vector<CUmemGenericAllocationHandle> mUcHandles;

    // For intra-node mcast
    tensorrt_llm::runtime::IpcNvlsHandle* mNvlsHandle;

    void allocMnMcastMem(size_t bufSize);
    void allocNvlsMcastMem(size_t bufSize);
};

constexpr size_t kSIGNAL_PAD_SIZE = 2048;

} // namespace tensorrt_llm::runtime
