/*
 * rga_cvt.c — 薄封装 librga im2d API，供 Python ctypes 调用
 * 编译：见 build_rga.sh
 */
#include <stdint.h>
#include "im2d.h"

/*
 * NV12 (YCbCr 4:2:0 semi-planar) → BGR packed
 * 成功返回 0，失败返回负数
 */
int rga_nv12_to_bgr(uintptr_t src_addr, uintptr_t dst_addr, int w, int h)
{
    rga_buffer_t src = wrapbuffer_virtualaddr((void *)src_addr, w, h,
                                              RK_FORMAT_YCbCr_420_SP);
    rga_buffer_t dst = wrapbuffer_virtualaddr((void *)dst_addr, w, h,
                                              RK_FORMAT_BGR_888);
    IM_STATUS st = imcvtcolor(src, dst,
                              RK_FORMAT_YCbCr_420_SP, RK_FORMAT_BGR_888,
                              IM_COLOR_SPACE_DEFAULT, 1);
    return (st >= IM_STATUS_SUCCESS) ? 0 : (int)st;
}

/*
 * BGR packed → NV12 (YCbCr 4:2:0 semi-planar)
 * 成功返回 0，失败返回负数
 */
int rga_bgr_to_nv12(uintptr_t src_addr, uintptr_t dst_addr, int w, int h)
{
    rga_buffer_t src = wrapbuffer_virtualaddr((void *)src_addr, w, h,
                                              RK_FORMAT_BGR_888);
    rga_buffer_t dst = wrapbuffer_virtualaddr((void *)dst_addr, w, h,
                                              RK_FORMAT_YCbCr_420_SP);
    IM_STATUS st = imcvtcolor(src, dst,
                              RK_FORMAT_BGR_888, RK_FORMAT_YCbCr_420_SP,
                              IM_COLOR_SPACE_DEFAULT, 1);
    return (st >= IM_STATUS_SUCCESS) ? 0 : (int)st;
}
