"""
crop_goci2.py - GitHub Actions 云端专属 (带实时进度显示版)
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
ROW_START, ROW_END = 3776, 7335
COL_START, COL_END = 103, 3723


def get_available_slots(date_str):
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
        print(f"查询失败: {e}", flush=True)
        return []


def download_file(url, out_path):
    """利用 curl 显示进度拉取"""
    print(f"  -> 正在从韩国 NOSC 传输大文件 (可能需要 1~2 分钟，请稍候)...", flush=True)
    cmd = ["curl", "-k", "-S", "--retry", "3", "--connect-timeout", "30", url, "-o", out_path]
    res = subprocess.run(cmd)
    return res.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 1024*1024


def crop_and_compress(src_nc, out_nc, product):
    print(f"  -> 正在执行东海 250m 原生纯切片 (零插值)...", flush=True)
    with h5py.File(src_nc, "r") as src:
        lats = src["navigation_data/latitude"][ROW_START:ROW_END, COL_START:COL_END]
        lons = src["navigation_data/longitude"][ROW_START:ROW_END, COL_START:COL_END]
        
        var_dict = {}
        if product.upper() == "TSS":
            var_dict["TSS"] = src["geophysical_data/TSS"][ROW_START:ROW_END, COL_START:COL_END]
            if "geophysical_data/flag" in src:
                var_dict["flag"] = src["geophysical_data/flag"][ROW_START:ROW_END, COL_START:COL_END]
        elif product.upper() == "AC":
            for k in src["geophysical_data"].keys():
                if k.startswith("Rrs_") or k == "flag":
                    var_dict[k] = src[f"geophysical_data/{k}"][ROW_START:ROW_END, COL_START:COL_END]

    with netCDF4.Dataset(out_nc, "w", format="NETCDF4") as dst:
        dst.createDimension("rows", ROW_END - ROW_START)
        dst.createDimension("cols", COL_END - COL_START)
        dst.spatial_extent = "117.0E-126.0E, 25.0N-35.0N (Preserved 250m Native Resolution)"
        dst.source_sensor = "GEO-KOMPSAT-2B GOCI-II"

        v_lat = dst.createVariable("latitude", "f4", ("rows", "cols"), zlib=True, complevel=4)
        v_lon = dst.createVariable("longitude", "f4", ("rows", "cols"), zlib=True, complevel=4)
        v_lat[:] = lats
        v_lon[:] = lons

        for vname, arr in var_dict.items():
            dtype = "u1" if vname == "flag" else "f4"
            fill = None if vname == "flag" else np.nan
            v = dst.createVariable(vname, dtype, ("rows", "cols"), zlib=True, complevel=4, fill_value=fill)
            v[:] = arr

    orig_mb = os.path.getsize(src_nc) / (1024 * 1024)
    crop_mb = os.path.getsize(out_nc) / (1024 * 1024)
    print(f"  ✓ 切片完成! 原体积: {orig_mb:.2f} MB -> 原生250m成品: {crop_mb:.2f} MB (节省 {(1 - crop_mb/orig_mb)*100:.1f}%)\n", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, required=True, help="日期 YYYY-MM-DD")
    parser.add_argument("--product", type=str, default="TSS", choices=["TSS", "AC"])
    parser.add_argument("--hours", type=str, nargs="+", default=["03", "04", "05"], help="UTC小时列表")
    args = parser.parse_args()

    os.makedirs("output_ecs", exist_ok=True)
    dt = datetime.strptime(args.date, "%Y-%m-%d")
    
    print(f"==================================================", flush=True)
    print(f" GOCI-II 云端流水线启动: {args.date} | 产品: {args.product}", flush=True)
    print(f" 目标时次窗口: {args.hours} UTC", flush=True)
    print(f"==================================================", flush=True)

    slots = get_available_slots(args.date)
    target_slots = [s for s in slots if s.split("_")[-1][:2] in args.hours]
    print(f"匹配到待处理时次数: {len(target_slots)} 个: {[s.split('_')[-1][:4] for s in target_slots]}\n", flush=True)

    for idx, s_name in enumerate(target_slots, 1):
        time_tag = s_name.split("_")[-1]
        raw_fn = f"GK2B_GOCI2_L2_{dt.strftime('%Y%m%d')}_{time_tag}_LA_{args.product}.nc"
        url = f"{BASE_OPENDAP}/{dt.strftime('%Y/%m/%d')}/L2/{s_name}/{raw_fn}"
        
        print(f"[{idx}/{len(target_slots)}] 时次 {time_tag[:2]}:{time_tag[2:4]} UTC: 拉取 {raw_fn}", flush=True)
        if download_file(url, raw_fn):
            out_fn = os.path.join("output_ecs", f"GK2B_GOCI2_250m_ECS_{dt.strftime('%Y%m%d')}_{time_tag}_{args.product}.nc")
            crop_and_compress(raw_fn, out_fn, args.product)
            os.remove(raw_fn)
        else:
            print(f"  ✗ 下载失败: {url}\n", flush=True)

    print("全部时次切片处理完毕！准备打包发布...", flush=True)


if __name__ == "__main__":
    main()
