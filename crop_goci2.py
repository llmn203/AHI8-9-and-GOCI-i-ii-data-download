"""
crop_goci2.py - GOCI-II 250m 东海原生网格多时序追加与纯切片引擎 (时空立方体版)
"""
import os
import re
import sys
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


def download_file(url, out_path):
    """利用 curl 极速下载"""
    cmd = ["curl", "-k", "-S", "--retry", "3", "--connect-timeout", "30", url, "-o", out_path]
    res = subprocess.run(cmd)
    return res.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 1024 * 1024


def init_multitime_nc(nc_path, sample_raw_nc, product):
    """初始化带 time 无限维度的多时次 NetCDF4 文件，静态经纬度只存 1 次"""
    print(f"正在初始化多时序 NetCDF4 数据立方体: {nc_path}", flush=True)
    with h5py.File(sample_raw_nc, "r") as src:
        lats = src["navigation_data/latitude"][ROW_START:ROW_END, COL_START:COL_END]
        lons = src["navigation_data/longitude"][ROW_START:ROW_END, COL_START:COL_END]
        
        # 探测变量清单
        var_names = []
        if product.upper() == "TSS":
            var_names = ["TSS", "flag"]
        elif product.upper() == "AC":
            for k in src["geophysical_data"].keys():
                if k.startswith("Rrs_") or k == "flag":
                    var_names.append(k)

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
            dtype = "u1" if vname == "flag" else "f4"
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
            for k in src["geophysical_data"].keys():
                if k.startswith("Rrs_") or k == "flag":
                    var_data[k] = src[f"geophysical_data/{k}"][ROW_START:ROW_END, COL_START:COL_END]

    with netCDF4.Dataset(nc_path, "a", format="NETCDF4") as dst:
        cur_idx = len(dst.variables["time"])
        dst.variables["time"][cur_idx] = int(dt_slot.timestamp())
        for vname, arr in var_data.items():
            if vname in dst.variables:
                dst.variables[vname][cur_idx, :, :] = arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year_month", type=str, required=True, help="年月 YYYY-MM")
    parser.add_argument("--start_day", type=int, required=True, help="起始日 (1-31)")
    parser.add_argument("--end_day", type=int, required=True, help="结束日 (1-31)")
    parser.add_argument("--product", type=str, default="AC", choices=["AC", "TSS"])
    parser.add_argument("--hours", type=str, nargs="+", default=["23", "00", "01", "02", "03", "04", "05", "06", "07", "08"], help="UTC小时列表 (全覆盖10帧)")
    parser.add_argument("--output", type=str, required=True, help="输出分块 nc 路径")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    year, month = int(args.year_month.split("-")[0]), int(args.year_month.split("-")[1])

    print("=" * 80, flush=True)
    print(f" GOCI-II 250m 多日多时序切片流水线启动", flush=True)
    print(f" 目标区间: {args.year_month}-{args.start_day:02d} 至 {args.year_month}-{args.end_day:02d} | 产品: {args.product}", flush=True)
    print(f" 每日目标时次 (10 帧全覆盖): {args.hours}", flush=True)
    print(f" 输出立方体目标: {args.output}", flush=True)
    print("=" * 80, flush=True)

    nc_initialized = False
    total_slots_processed = 0

    for day in range(args.start_day, args.end_day + 1):
        day_str = f"{year:04d}-{month:02d}-{day:02d}"
        slots = get_available_slots(day_str)
        target_slots = [s for s in slots if s.split("_")[-1][:2] in args.hours]
        print(f"\n>>> 处理日期 {day_str} (命中有效时次 {len(target_slots)}/{len(slots)} 个) ...", flush=True)

        for s_idx, s_name in enumerate(target_slots, 1):
            time_tag = s_name.split("_")[-1]  # HHMMSS
            slot_dt = datetime.strptime(f"{day_str} {time_tag}", "%Y-%m-%d %H%M%S")
            raw_fn = f"raw_tmp_{day_str}_{time_tag}.nc"
            url = f"{BASE_OPENDAP}/{year:04d}/{month:02d}/{day:02d}/L2/{s_name}/GK2B_GOCI2_L2_{year:04d}{month:02d}{day:02d}_{time_tag}_LA_{args.product}.nc"

            print(f"  [{s_idx}/{len(target_slots)}] 拉取 {time_tag[:2]}:{time_tag[2:4]} UTC ...", end="", flush=True)
            if download_file(url, raw_fn):
                if not nc_initialized:
                    init_multitime_nc(args.output, raw_fn, args.product)
                    nc_initialized = True

                append_slot_to_nc(args.output, raw_fn, slot_dt, args.product)
                try:
                    os.remove(raw_fn)  # 彻底删除 400MB 原始文件
                except Exception:
                    pass
                total_slots_processed += 1
                cur_sz_mb = os.path.getsize(args.output) / (1024 * 1024)
                print(f" ✓ 已写入立方体 (累计: {total_slots_processed} 帧 | 立方体大小: {cur_sz_mb:.1f} MB)", flush=True)
            else:
                print(f" ✗ 下载失败/不存在", flush=True)

    print("\n" + "=" * 80, flush=True)
    print(f" 阶段性矩阵任务完成！总计追加切片: {total_slots_processed} 帧", flush=True)
    if os.path.exists(args.output):
        print(f" 产生成品立方体: {args.output} ({os.path.getsize(args.output)/(1024*1024):.2f} MB)", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
    
