"""
crop_goci2.py - GitHub Actions 云端专属: GOCI-II 免密极速拉取与东海原生 250m 纯切片
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

# 针对 AHI [117.0°E ~ 126.0°E, 25.0°N ~ 35.0°N] 的严格原生像元切片索引 (实测解算)
# 严格保留 250m 原生分辨率，绝对不执行插值平滑
ROW_START, ROW_END = 3776, 7335  # 3559 行 (纬度向)
COL_START, COL_END = 103, 3723   # 3620 列 (经度向)


def get_available_slots(date_str):
    """查询当天所有可用时次"""
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
        print(f"查询失败: {e}")
        return []


def download_file(url, out_path):
    """利用 curl 极速下载"""
    cmd = ["curl", "-k", "-s", "-S", "--retry", "3", url, "-o", out_path]
    res = subprocess.run(cmd)
    return res.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 1024*1024


def crop_and_compress(src_nc, out_nc, product):
    """按 AHI BBOX 纯索引裁切并启用 zlib 高压缩，保留 250m 原生分辨率"""
    print(f"正在裁切原生 250m 网格: {src_nc} -> {out_nc}")
    
    with h5py.File(src_nc, "r") as src:
        lats = src["navigation_data/latitude"][ROW_START:ROW_END, COL_START:COL_END]
        lons = src["navigation_data/longitude"][ROW_START:ROW_END, COL_START:COL_END]
        
        # 提取目标波段/变量
        var_dict = {}
        if product.upper() == "TSS":
            var_dict["TSS"] = src["geophysical_data/TSS"][ROW_START:ROW_END, COL_START:COL_END]
            if "geophysical_data/flag" in src:
                var_dict["flag"] = src["geophysical_data/flag"][ROW_START:ROW_END, COL_START:COL_END]
        elif product.upper() == "AC":
            # 提取 12 波段大气校正 Rrs
            for k in src["geophysical_data"].keys():
                if k.startswith("Rrs_") or k == "flag":
                    var_dict[k] = src[f"geophysical_data/{k}"][ROW_START:ROW_END, COL_START:COL_END]

    # 导出轻量化 NetCDF4 (开启 zlib 压缩)
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
    print(f"裁切完成! 原体积: {orig_mb:.2f} MB -> 裁切后原生250m体积: {crop_mb:.2f} MB (节省 {(1 - crop_mb/orig_mb)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, required=True, help="日期 YYYY-MM-DD")
    parser.add_argument("--product", type=str, default="TSS", choices=["TSS", "AC"])
    parser.add_argument("--hours", type=str, nargs="+", default=["03", "04", "05"], help="UTC小时列表 (默认 03 04 05)")
    args = parser.parse_args()

    os.makedirs("output_ecs", exist_ok=True)
    dt = datetime.strptime(args.date, "%Y-%m-%d")
    
    print(f"=== GOCI-II 云端切片流水线启动: {args.date} | 产品: {args.product} ===")
    slots = get_available_slots(args.date)
    print(f"OpenDAP 探测到当天时次数: {len(slots)}")

    # 仅下载指定小时 (默认正午黄金窗口)
    target_slots = [s for s in slots if s.split("_")[-1][:2] in args.hours]
    print(f"计划切片时次: {[s.split('_')[-1][:4] for s in target_slots]}")

    for s_name in target_slots:
        time_tag = s_name.split("_")[-1]
        raw_fn = f"GK2B_GOCI2_L2_{dt.strftime('%Y%m%d')}_{time_tag}_LA_{args.product}.nc"
        url = f"{BASE_OPENDAP}/{dt.strftime('%Y/%m/%d')}/L2/{s_name}/{raw_fn}"
        
        print(f"\n[云端直连下载] {raw_fn} ...")
        if download_file(url, raw_fn):
            out_fn = os.path.join("output_ecs", f"GK2B_GOCI2_250m_ECS_{dt.strftime('%Y%m%d')}_{time_tag}_{args.product}.nc")
            crop_and_compress(raw_fn, out_fn, args.product)
            os.remove(raw_fn)  # 即刻粉碎原始 160MB 全幅大文件，不占云端存储
        else:
            print(f"下载失败: {url}")


if __name__ == "__main__":
    main()
