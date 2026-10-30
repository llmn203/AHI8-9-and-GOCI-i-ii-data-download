"""
crop_goci2.py - GOCI-II 250m 东海原生网格按天单文件切片引擎 (单日单文件 ~600MB·12波段完整修复版)
"""
import os
import re
import sys
import time
import shutil
import argparse
import subprocess
import calendar
import requests
import urllib3
import h5py
import netCDF4
import numpy as np
from datetime import datetime

urllib3.disable_warnings()

BASE_OPENDAP = "https://www.nosc.go.kr/opendap/GOCI-II"
DEFAULT_REPO = "llmn203/AHI8-9-and-GOCI-i-ii-data-download"

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
    优先使用 aria2c (16 连接分块并发，秒破服务器端单连接流控)；
    若未安装则自动回退至静默 curl (-k -s -S，杜绝几万行进度条刷屏)。
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


def init_daily_nc(nc_path, sample_raw_nc, product):
    """初始化单日 NetCDF4 文件 (包含全天多时次，静态经纬度只存 1 次)"""
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
            if "geophysical_data/flag" in src or ("geophysical_data" in src and "flag" in src["geophysical_data"]):
                var_names.append("flag")

    with netCDF4.Dataset(nc_path, "w", format="NETCDF4") as dst:
        dst.createDimension("time", None)
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

        # 12 波段与标志位
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
            if "geophysical_data/flag" in src:
                var_data["flag"] = src["geophysical_data/flag"][ROW_START:ROW_END, COL_START:COL_END]

    with netCDF4.Dataset(nc_path, "a", format="NETCDF4") as dst:
        cur_idx = len(dst.variables["time"])
        dst.variables["time"][cur_idx] = int(dt_slot.timestamp())
        for vname, arr in var_data.items():
            if vname in dst.variables:
                dst.variables[vname][cur_idx, :, :] = arr


def check_asset_exists_in_release(repo, tag, filename):
    """探测 GitHub Release 中是否已存在切片资产，避免重跑时重复下载切片"""
    url = f"https://github.com/{repo}/releases/download/{tag}/{filename}"
    try:
        r = requests.head(url, timeout=5, allow_redirects=True)
        if r.status_code == 200:
            return True
    except Exception:
        pass
    return False


def make_bar(pct, width=15):
    filled = int(width * pct / 100)
    return "█" * filled + "░" * (width - filled)


def process_single_day(year, month, day, product, hours, output_dir, tag=None):
    """处理单一完整自然日 (10 帧切片合并为单日独立单文件 ~500-600MB)"""
    day_str = f"{year:04d}-{month:02d}-{day:02d}"
    date_tag = f"{year:04d}{month:02d}{day:02d}"
    out_nc_name = f"GOCI2_250m_ECS_{date_tag}_{product.upper()}.nc"
    out_nc_path = os.path.join(output_dir, out_nc_name)

    # 1. 检查本地是否已有完好的切片产物
    if os.path.exists(out_nc_path) and os.path.getsize(out_nc_path) > 10 * 1024 * 1024:
        print(f"\n  [√ 本地已存在] {out_nc_name} 已就绪，秒级跳过！", flush=True)
        return True

    # 2. 检查云端 Release 是否已发布过
    if tag and check_asset_exists_in_release(DEFAULT_REPO, tag, out_nc_name):
        print(f"\n  [√ 云端已发布] Release [{tag}] 已包含 {out_nc_name}，秒级跳过！", flush=True)
        return True

    slots = get_available_slots(day_str)
    target_slots = [s for s in slots if s.split("_")[-1][:2] in hours]

    print(f"\n" + "=" * 88, flush=True)
    print(f"  >>> 正在切片自然日: {day_str} ({date_tag})", flush=True)
    print(f"  韩国服务器命中时次: {len(target_slots)}/{len(hours)} 个 | 产物目标: {out_nc_name}", flush=True)
    print("=" * 88, flush=True)

    if len(target_slots) == 0:
        print(f"  [跳过] 该日无有效观测时次数据", flush=True)
        return False

    temp_nc = out_nc_path + ".tmp"
    if os.path.exists(temp_nc):
        try:
            os.remove(temp_nc)
        except Exception:
            pass

    nc_initialized = False
    processed_count = 0
    t0_day = time.time()

    for s_idx, s_name in enumerate(target_slots, 1):
        time_tag = s_name.split("_")[-1]  # HHMMSS
        slot_dt = datetime.strptime(f"{day_str} {time_tag}", "%Y-%m-%d %H%M%S")
        raw_fn = f"raw_tmp_{day_str}_{time_tag}.nc"
        url = f"{BASE_OPENDAP}/{year:04d}/{month:02d}/{day:02d}/L2/{s_name}/GK2B_GOCI2_L2_{year:04d}{month:02d}{day:02d}_{time_tag}_LA_{product}.nc"

        success, dt, speed = download_file_accelerated(url, raw_fn)
        if success:
            if not nc_initialized:
                init_daily_nc(temp_nc, raw_fn, product)
                nc_initialized = True

            append_slot_to_nc(temp_nc, raw_fn, slot_dt, product)
            try:
                os.remove(raw_fn)
            except Exception:
                pass

            processed_count += 1
            pct = (processed_count / len(target_slots)) * 100
            bar = make_bar(pct, width=12)
            cur_sz_mb = os.path.getsize(temp_nc) / (1024 * 1024)
            print(
                f"  [{bar}] {pct:5.1f}% ({processed_count:02d}/{len(target_slots)}帧) "
                f"| {time_tag[:2]}:{time_tag[2:4]} UTC "
                f"| 拉取: {dt:4.1f}s ({speed:5.1f} MB/s) "
                f"| 12波段切片入库 (单日大小: {cur_sz_mb:5.1f} MB)",
                flush=True
            )
        else:
            print(f"  [ ✗ 时次下载失败 ] {time_tag[:2]}:{time_tag[2:4]} UTC | 耗时 {dt:.1f}s", flush=True)

    if nc_initialized and os.path.exists(temp_nc) and os.path.getsize(temp_nc) > 10 * 1024 * 1024:
        if os.path.exists(out_nc_path):
            os.remove(out_nc_path)
        os.rename(temp_nc, out_nc_path)
        final_mb = os.path.getsize(out_nc_path) / (1024 * 1024)
        print(f"\n  ✓ [{day_str}] 单日切片完成！总计: {processed_count} 帧 | 体积: {final_mb:.1f} MB | 耗时: {(time.time()-t0_day)/60:.1f} 分钟", flush=True)
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="GOCI-II 250m 按天独立单文件切片流水线 (12波段完整版)")
    parser.add_argument("--year_month", type=str, required=True, help="年月 YYYY-MM")
    parser.add_argument("--start_day", type=int, required=True, help="起始日 (1-31)")
    parser.add_argument("--end_day", type=int, required=True, help="结束日 (1-31)")
    parser.add_argument("--product", type=str, default="AC", choices=["AC", "TSS"])
    parser.add_argument("--hours", type=str, nargs="+", default=["23", "00", "01", "02", "03", "04", "05", "06", "07", "08"], help="UTC小时列表 (全覆盖10帧)")
    parser.add_argument("--output_dir", type=str, default="output_ecs", help="输出目录")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    year, month = int(args.year_month.split("-")[0]), int(args.year_month.split("-")[1])
    tag = f"{args.year_month}-{args.product.upper()}"

    # 动态获取当月实际最大天数 (如 2024年2月为 29 天，4月为 30 天)
    # 彻底杜绝 2024-02-30 引发 ValueError: day is out of range for month 导致发布中断
    _, max_day = calendar.monthrange(year, month)
    effective_start = args.start_day
    effective_end = min(args.end_day, max_day)

    if effective_start > max_day:
        print(f"\n[跳过] 起始日 {effective_start} 超过该月自然总天数 ({max_day} 天)，无须切片任务。", flush=True)
        return

    num_days = effective_end - effective_start + 1

    print("=" * 96, flush=True)
    print(f"  GOCI-II 250m 东海原生网格按天单文件流水线 (单日单文件 ~600MB·12波段完整修复版)", flush=True)
    print(f"  区间: {args.year_month}-{effective_start:02d} 至 {args.year_month}-{effective_end:02d} (共 {num_days} 天) | 产品: {args.product}", flush=True)
    print(f"  每日时次: 10 帧全覆盖 ({', '.join(args.hours)} UTC)", flush=True)
    print(f"  单日产物格式: {args.output_dir}/GOCI2_250m_ECS_YYYYMMDD_{args.product}.nc", flush=True)
    print("=" * 96, flush=True)

    for day in range(effective_start, effective_end + 1):
        process_single_day(year, month, day, args.product, args.hours, args.output_dir, tag=tag)


if __name__ == "__main__":
    main()

