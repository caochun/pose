#!/bin/bash
# 编译 rga_cvt.so — 依赖 librga-dev
set -e

# 查找 librga 头文件
INC=""
for d in /usr/include/rga /usr/local/include/rga /usr/include; do
    if [ -f "$d/im2d.h" ]; then
        INC="-I$d"
        break
    fi
done

if [ -z "$INC" ]; then
    echo "错误：找不到 im2d.h，请先安装 librga-dev："
    echo "  sudo apt install librga2 librga-dev"
    echo "  或从源码编译：https://github.com/airockchip/librga"
    exit 1
fi

echo "使用头文件：$INC"
gcc -O2 -shared -fPIC $INC -o rga_cvt.so rga_cvt.c -lrga
echo "编译成功：rga_cvt.so"
