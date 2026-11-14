#include <cassert>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include "kernels/mega_moe_bf16/include/mega_moe_training_workspace.h"

using namespace MegaMoeBF16Training;
int main()
{
    static_assert(SaveGateUpAsMxFp8(FLT_MAX));
    static_assert(!SaveGateUpAsMxFp8(1.0F));
    static_assert(!SaveGateUpAsMxFp8(10.0F));
    // This selector accepts normalized tiling values, not public limit=0.
    static_assert(!SaveGateUpAsMxFp8(0.0F));
    constexpr TrainingSavedStateLayout compressed(4096,7168,16,8,4,65536,8192,true);
    constexpr TrainingSavedStateLayout plain(4096,7168,16,8,4,65536,8192,false);
    constexpr TrainingSavedStateLayout disabled(4096,7168,16,8,4,65536,8192,false,false);
    constexpr TrainingSavedStateLayout clipped(4096,7168,16,8,4,65536,8192,true,true,true);
    static_assert(clipped.totalBytes == 1594497024ULL);
    static_assert(clipped.gateHighOffset == compressed.totalBytes);
    static_assert(clipped.upLowOffset == 1527388160ULL);
    static_assert(clipped.upHighOffset == 1560942592ULL);
    static_assert(clipped.clipMaskPlaneBytes == 32ULL*1024*1024);
    static_assert(clipped.layoutVersion == 2 && clipped.clipMaskKind == kClipMaskThreeBitplanesV1);
    static_assert(compressed.layoutVersion == 1 && compressed.clipMaskKind == kClipMaskNone);
    static_assert(SaveGateUpClipMasks(10.0F, kSaveMxFp8));
    static_assert(!SaveGateUpClipMasks(10.0F, kSaveAuto));
    static_assert(!SaveGateUpClipMasks(FLT_MAX, kSaveMxFp8));
    static_assert(SavedGateUpQuantUbLayout::totalBytesWithClipMasks == 218*1024);
    static_assert(SavedGateUpQuantUbLayout::totalBytesWithClipMasks <= 248*1024);
    static_assert(ResolveSavedGateUpMode(kSaveAuto, FLT_MAX) == kSaveMxFp8);
    static_assert(ResolveSavedGateUpMode(kSaveAuto, 10.0F) == kSaveBf16);
    static_assert(ResolveSavedGateUpMode(kSaveNone, FLT_MAX) == kSaveNone);
    static_assert(ResolveSavedGateUpMode(kSaveBf16, FLT_MAX) == kSaveBf16);
    static_assert(disabled.totalBytes == 940185600ULL);
    static_assert(disabled.totalBytes == disabled.legacyBytes);
    static_assert(!disabled.gateUpSaved && !disabled.gateUpMxFp8);
    static_assert(disabled.gateUpPlaneBytes == 0 && disabled.gateUpScaleBytes == 0);
    static_assert(compressed.totalBytes == 1493833728ULL);
    static_assert(plain.totalBytes == 2013927424ULL);
    static_assert(compressed.gateUpScaleBytes == 8ULL*1024*1024);
    static_assert(SavedGateUpQuantUbLayout::totalBytes == 206*1024);
    static_assert(SavedGateUpQuantUbLayout::totalBytes <= 248*1024);
    static_assert(SavedGateUpQuantUbLayout::gateData == 128*1024);

    using M192QuantLayout = SavedGateUpQuantUbLayoutFor<192, 64>;
    static_assert(M192QuantLayout::gateData == 192 * 1024);
    static_assert(M192QuantLayout::totalBytesWithClipMasks == 237 * 1024);
    static_assert(M192QuantLayout::totalBytesWithClipMasks <= 248 * 1024);
    static_assert(M192QuantLayout::gateData >= 2 * 192 * 256 * sizeof(uint16_t));
    static_assert(M192QuantLayout::upData >= M192QuantLayout::gateData + 64 * 256);
    int checked=0;
    for(uint64_t ep: {2ULL,8ULL,512ULL})
    for(uint64_t rows: {1ULL,127ULL,128ULL,129ULL,65536ULL})
    for(uint64_t n: {512ULL,8192ULL}) {
        TrainingSavedStateLayout a(4096,7168,16,ep,4,rows,n,true);
        TrainingSavedStateLayout b(4096,7168,16,ep,4,rows,n,false);
        TrainingSavedStateLayout off(4096,7168,16,ep,4,rows,n,false,false);
        TrainingSavedStateLayout clip(4096,7168,16,ep,4,rows,n,true,true,true);
        TrainingSavedStateLayout noMasks(4096,7168,16,ep,4,rows,n,false,true,true);
        assert(clip.gateOffset == a.gateOffset && clip.upOffset == a.upOffset);
        assert(clip.gateScaleOffset == a.gateScaleOffset && clip.upScaleOffset == a.upScaleOffset);
        assert(clip.gateHighOffset == a.totalBytes);
        assert(clip.clipMaskPlaneBytes == AlignUp(rows * (n/64) * 4, 512));
        assert(clip.totalBytes == a.totalBytes + 3*clip.clipMaskPlaneBytes);
        assert(clip.upHighOffset + clip.clipMaskPlaneBytes == clip.totalBytes);
        assert(noMasks.totalBytes == b.totalBytes && noMasks.clipMaskKind == kClipMaskNone);
        assert(off.totalBytes == a.legacyBytes && off.totalBytes == b.legacyBytes);
        assert(off.backwardProgressOffset == a.backwardProgressOffset);
        assert(off.dispatchedTopkWeightsOffset == a.dispatchedTopkWeightsOffset);
        assert(off.gateUpPlaneBytes == 0 && off.gateUpScaleBytes == 0);
        assert(a.ptrAOffset==b.ptrAOffset && a.cumsumOffset==b.cumsumOffset);
        assert(a.expandedRowIdxOffset==b.expandedRowIdxOffset);
        assert(a.expandedRowIdxGatherOffset==b.expandedRowIdxGatherOffset);
        assert(a.sumBeforeRankOffset==b.sumBeforeRankOffset);
        assert(a.savedTokenCountsOffset==b.savedTokenCountsOffset);
        assert(a.dispatchedTopkWeightsOffset==b.dispatchedTopkWeightsOffset);
        assert(a.backwardProgressOffset==b.backwardProgressOffset && a.legacyBytes==b.legacyBytes);
        assert(a.gateOffset==b.gateOffset);
        assert(a.upOffset>=a.gateOffset+rows*(n/2));
        assert(a.gateScaleOffset>=a.upOffset+rows*(n/2));
        assert(a.upScaleOffset>=a.gateScaleOffset+rows*(n/64));
        assert(a.totalBytes>=a.upScaleOffset+rows*(n/64));
        assert(a.totalBytes-a.upScaleOffset-rows*(n/64)<512);
        assert(b.gateUpScaleBytes==0 && b.totalBytes==b.upOffset+b.gateUpPlaneBytes);
        // Tiny capacities can cost more due to separately aligned scale planes.
        if(rows*(n/2)>=4096) assert(a.totalBytes<b.totalBytes);
        ++checked;
    }
    printf("PASS saved MXFP8/BF16: %d layouts, fixed legacy offsets, scale capacity, M128 206 KiB / M192 237 KiB UB\n",checked);
}
