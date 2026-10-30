"""
crop_goci2.py - GOCI-II 250m 东海原生网格多时序追加与纯切片引擎 (12波段穿透修复·单行看板加速版)
"""
import os
import re
import sys
import time
import shutil
import argparse
import subprocess
import requests
import urllib3
import h5py
import netCDF4
import numpy as np
from datetime import datetime

urllib3.disable_warnings()

BASE_OPENDAP = "https://www.nosc.go.kr/opendap/GOCI-II"

# 针对 AHI 目标区 [117.0°E ~ 126.0°E, 25.0°N ~ 35.0°N] 的严格原生像元切片 (保持 250m 原生分辨率)
ROW_START, ROW_END = 3776, 7335  # 3559 行 (纬度向)
COL_START, COL_END = 103, 3723   # 3620 列 (经度向)


def get_available_slots(date_str):
    """查询指定日期 OpenDAP 所有可用时次"""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    url = f"{BASE_OPENDAP}/{dt.strftime('%Y/%m/%d')}/L2/contents.html"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, verify=False, timeout=30)
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        pattern = rf"GK2_GC2_L2_{dt.strftime('%Y%m%d')}_\d{{6}}"
        return sorted(list(set(re.findall(pattern, resp.text))))
    except Exception as e:
        print(f"  [查询异常] {date_str}: {e}", flush=True)
        return []


def download_file_accelerated(url, out_path):
    """
    高性能多线程下载：
    优先检测并使用 aria2c (16 连接分块并发，秒破服务器端单连接流控)；
    若未安装则自动回退至静默 curl (-k -s -S，杜绝几万行进度条刷屏)。
    返回: (成功标志 bool, 耗时秒 float, 平均速度 MB/s float)
    """
    t0 = time.time()
    out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    out_fn = os.path.basename(out_path)

    has_aria2 = shutil.which("aria2c") is not None

    if has_aria2:
        cmd = [
            "aria2c",
            "-x", "16",
            "-s", "16",
            "-k", "1M",
            "--check-certificate=false",
            "--file-allocation=none",
            "--summary-interval=0",
            "--console-log-level=warn",
            "-d", out_dir,
            "-o", out_fn,
            url
        ]
    else:
        cmd = [
            "curl", "-k", "-s", "-S",
            "--retry", "3",
            "--connect-timeout", "30",
            url,
            "-o", out_path
        ]

    try:
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        elapsed = max(0.01, time.time() - t0)
        if res.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 1024 * 1024:
            sz_mb = os.path.getsize(out_path) / (1024 * 1024)
            speed = sz_mb / elapsed
            return True, elapsed, speed
        return False, elapsed, 0.0
    except Exception:
        return False, max(0.01, time.time() - t0), 0.0


def init_multitime_nc(nc_path, sample_raw_nc, product):
    """初始化带 time 无限维度的多时次 NetCDF4 文件，静态经纬度只存 1 次"""
    with h5py.File(sample_raw_nc, "r") as src:
        lats = src["navigation_data/latitude"][ROW_START:ROW_END, COL_START:COL_END]
        lons = src["navigation_data/longitude"][ROW_START:ROW_END, COL_START:COL_END]
        
        var_names = []
        if product.upper() == "TSS":
            var_names = ["TSS", "flag"]
        elif product.upper() == "AC":
            # 穿透 Rrs 独立子组提取全部 12 个可见/近红外波段
            if "geophysical_data/Rrs" in src:
                for k in sorted(src["geophysical_data/Rrs"].keys()):
                    var_names.append(k)
            elif "geophysical_data" in src:
                for k in sorted(src["geophysical_data"].keys()):
                    if k.startswith("Rrs_"):
                        var_names.append(k)
            # 质检标志位
            if "geophysical_data/flag" in src or ("geophysical_data" in src and "flag" in src["geophysical_data"]):
                var_names.append("flag")

    print(f"  [变量骨架构建] 检测到产品变量数: {len(var_names)} 个 -> {var_names}", flush=True)

    with netCDF4.Dataset(nc_path, "w", format="NETCDF4") as dst:
        dst.createDimension("time", None)  # 无限时间维
        dst.createDimension("rows", ROW_END - ROW_START)
        dst.createDimension("cols", COL_END - COL_START)
        dst.spatial_extent = "117.0E-126.0E, 25.0N-35.0N (Preserved 250m Native Resolution)"
        dst.source_sensor = "GEO-KOMPSAT-2B GOCI-II"

        # 静态经纬度
        v_lat = dst.createVariable("latitude", "f4", ("rows", "cols"), zlib=True, complevel=4)
        v_lon = dst.createVariable("longitude", "f4", ("rows", "cols"), zlib=True, complevel=4)
        v_lat[:] = lats
        v_lon[:] = lons

        # 时间轴 (Unix 时间戳，单位秒)
        v_time = dst.createVariable("time", "i8", ("time",))
        v_time.units = "seconds since 1970-01-01 00:00:00"
        v_time.calendar = "standard"

        # 12 波段与标志位 (开启多维分块压缩 chunksizes=(1, 512, 512))
        for vname in var_names:
            dtype = "u4" if vname == "flag" else "f4"
            fill = None if vname == "flag" else np.nan
            dst.createVariable(
                vname, dtype, ("time", "rows", "cols"),
                zlib=True, complevel=4, chunksizes=(1, 512, 512), fill_value=fill
            )


def append_slot_to_nc(nc_path, raw_nc, dt_slot, product):
    """追加一个时次切片到 NetCDF4 文件末尾"""
    with h5py.File(raw_nc, "r") as src:
        var_data = {}
        if product.upper() == "TSS":
            var_data["TSS"] = src["geophysical_data/TSS"][ROW_START:ROW_END, COL_START:COL_END]
            if "geophysical_data/flag" in src:
                var_data["flag"] = src["geophysical_data/flag"][ROW_START:ROW_END, COL_START:COL_END]
        elif product.upper() == "AC":
            # 穿透提取 12 个 Rrs 波段
            if "geophysical_data/Rrs" in src:
                rrs_grp = src["geophysical_data/Rrs"]
                for k in rrs_grp.keys():
                    var_data[k] = rrs_grp[k][ROW_START:ROW_END, COL_START:COL_END]
            elif "geophysical_data" in src:
                for k in src["geophysical_data"].keys():
                    if k.startswith("Rrs_"):
                        var_data[k] = src[f"geophysical_data/{k}"][ROW_START:ROW_END, COL_START:COL_END]
            # 提取 flag
            if "geophysical_data/flag" in src:
                var_data["flag"] = src["geophysical_data/flag"][ROW_START:ROW_END, COL_START:COL_END]

    with netCDF4.Dataset(nc_path, "a", format="NETCDF4") as dst:
        cur_idx = len(dst.variables["time"])
        dst.variables["time"][cur_idx] = int(dt_slot.timestamp())
        for vname, arr in var_data.items():
            if vname in dst.variables:
                dst.variables[vname][cur_idx, :, :] = arr


def make_bar(pct, width=15):
    filled = int(width * pct / 100)
    return "█" * filled + "░" * (width - filled)


def main():
    parser = argparse.ArgumentParser(description="GOCI-II 250m 清爽看板切片流水线 (12波段修复版)")
    parser.add_argument("--year_month", type=str, required=True, help="年月 YYYY-MM")
    parser.add_argument("--start_day", type=int, required=True, help="起始日 (1-31)")
    parser.add_argument("--end_day", type=int, required=True, help="结束日 (1-31)")
    parser.add_argument("--product", type=str, default="AC", choices=["AC", "TSS"])
    parser.add_argument("--hours", type=str, nargs="+", default=["23", "00", "01", "02", "03", "04", "05", "06", "07", "08"], help="UTC小时列表 (全覆盖10帧)")
    parser.add_argument("--output", type=str, required=True, help="输出分块 nc 路径")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    year, month = int(args.year_month.split("-")[0]), int(args.year_month.split("-")[1])
    num_days = args.end_day - args.start_day + 1
    total_expected = num_days * len(args.hours)

    print("=" * 96, flush=True)
    print(f"  GOCI-II 250m 原生切片与时序组装流水线 (12波段完整穿透提取版)", flush=True)
    print(f"  目标区间: {args.year_month}-{args.start_day:02d} 至 {args.year_month}-{args.end_day:02d} (共 {num_days} 天) | 产品: {args.product}", flush=True)
    print(f"  每日时次: 10 帧全覆盖 ({', '.join(args.hours)} UTC) | 理论总时次: {total_expected} 帧", flush=True)
    print(f"  下载引擎: {'aria2c (16多线程极速版)' if shutil.which('aria2c') else 'curl (静默精简版)'}", flush=True)
    print(f"  目标输出: {args.output}", flush=True)
    print("=" * 96, flush=True)

    nc_initialized = False
    total_processed = 0
    t_start_all = time.time()

    for d_idx, day in enumerate(range(args.start_day, args.end_day + 1), 1):
        day_str = f"{year:04d}-{month:02d}-{day:02d}"
        slots = get_available_slots(day_str)
        target_slots = [s for s in slots if s.split("_")[-1][:2] in args.hours]

        print(f"\n[第 {d_idx:02d}/{num_days:02d} 天] >>> {day_str} | 韩国服务器检索到 {len(target_slots)}/{len(args.hours)} 个有效时次", flush=True)

        for s_idx, s_name in enumerate(target_slots, 1):
            time_tag = s_name.split("_")[-1]  # HHMMSS
            slot_dt = datetime.strptime(f"{day_str} {time_tag}", "%Y-%m-%d %H%M%S")
            raw_fn = f"raw_tmp_{day_str}_{time_tag}.nc"
            url = f"{BASE_OPENDAP}/{year:04d}/{month:02d}/{day:02d}/L2/{s_name}/GK2B_GOCI2_L2_{year:04d}{month:02d}{day:02d}_{time_tag}_LA_{args.product}.nc"

            success, dt, speed = download_file_accelerated(url, raw_fn)
            if success:
                if not nc_initialized:
                    init_multitime_nc(args.output, raw_fn, args.product)
                    nc_initialized = True

                append_slot_to_nc(args.output, raw_fn, slot_dt, args.product)
                try:
                    os.remove(raw_fn)
                except Exception:
                    pass

                total_processed += 1
                pct = (total_processed / total_expected) * 100
                bar = make_bar(pct, width=15)
                cur_sz_mb = os.path.getsize(args.output) / (1024 * 1024)
                
                # 规整单行输出看板
                print(
                    f"  [{bar}] {pct:5.1f}% ({total_processed:02d}/{total_expected}帧) "
                    f"| {time_tag[:2]}:{time_tag[2:4]} UTC "
                    f"| 拉取: {dt:4.1f}s ({speed:5.1f} MB/s) "
                    f"| 12波段切片入库 (产物体积: {cur_sz_mb:5.1f} MB)",
                    flush=True
                )
            else:
                print(f"  [ ✗ 下载失败/不存在 ] {time_tag[:2]}:{time_tag[2:4]} UTC | 耗时 {dt:.1f}s", flush=True)

    t_all_elapsed = time.time() - t_start_all
    print("\n" + "=" * 96, flush=True)
    print(f"  ✓ 分块切片任务执行完毕！累计处理: {total_processed} 帧 | 总耗时: {t_all_elapsed/60:.1f} 分钟", flush=True)
    if os.path.exists(args.output):
        print(f"  ✓ 产物立方体: {args.output} ({os.path.getsize(args.output)/(1024*1024):.2f} MB)", flush=True)
    print("=" * 96, flush=True)


if __name__ == "__main__":
    main()
