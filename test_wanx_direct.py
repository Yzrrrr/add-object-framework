#!/usr/bin/env python3
"""直接测试 wanx API 的局部重绘功能"""

import os
import sys
import base64
import io
import time
import requests
from pathlib import Path
from PIL import Image
import numpy as np

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not API_KEY:
    print("错误: DASHSCOPE_API_KEY 未设置")
    sys.exit(1)

SUBMIT_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/image2image/image-synthesis"
POLL_URL = "https://dashscope.aliyuncs.com/api/v1/tasks"

def test_wanx_inpaint():
    """测试 wanx 局部重绘"""
    
    # 加载测试图片
    image_path = "demo/demo2.jpg"
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    print(f"原图尺寸: {orig_w}x{orig_h}")
    
    # 创建一个简单的 mask（白色椭圆，表示编辑区域）
    # demo2 添加猫的位置是 (14%, 84%)，大小 (20%x18%)
    mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    
    cx = int(14 * orig_w / 100)
    cy = int(84 * orig_h / 100)
    rx = int(20 * orig_w / 100 / 2)
    ry = int(18 * orig_h / 100 / 2)
    
    import cv2
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
    
    mask_pil = Image.fromarray(mask)
    print(f"Mask 尺寸: {mask_pil.size}")
    print(f"Mask 白色像素: {np.sum(mask > 0)} ({np.sum(mask > 0) / mask.size * 100:.1f}%)")
    
    # 保存 mask 查看效果
    mask_pil.save("test_mask_debug.png")
    print("Mask 已保存到 test_mask_debug.png")
    
    # 转换为 base64
    def to_base64(pil_img, fmt="PNG"):
        buf = io.BytesIO()
        pil_img.save(buf, format=fmt)
        return f"data:image/{fmt.lower()};base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
    
    image_b64 = to_base64(img, "JPEG")
    mask_b64 = to_base64(mask_pil, "PNG")
    
    print(f"\nImage base64 长度: {len(image_b64)}")
    print(f"Mask base64 长度: {len(mask_b64)}")
    
    # 构建请求
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "X-DashScope-Async": "enable",
        "Content-Type": "application/json",
    }
    
    body = {
        "model": "wanx2.1-imageedit",
        "input": {
            "function": "description_edit_with_mask",
            "prompt": "一只真实的流浪猫坐在石板地上，清晰的猫轮廓，有头有尾有四条腿",
            "base_image_url": image_b64,
            "mask_image_url": mask_b64,
        },
        "parameters": {"n": 1}
    }
    
    print("\n提交任务到 wanx...")
    
    # 提交任务
    resp = requests.post(SUBMIT_URL, headers=headers, json=body, timeout=60)
    print(f"响应状态: {resp.status_code}")
    
    if resp.status_code != 200:
        print(f"错误响应: {resp.text[:500]}")
        return
    
    data = resp.json()
    print(f"响应: {data}")
    
    task_id = data.get("output", {}).get("task_id")
    if not task_id:
        print("错误: 没有 task_id")
        return
    
    print(f"\n任务 ID: {task_id}")
    
    # 轮询结果
    print("等待结果...")
    deadline = time.time() + 180
    
    while time.time() < deadline:
        time.sleep(3)
        
        poll_resp = requests.get(f"{POLL_URL}/{task_id}", 
                                  headers={"Authorization": f"Bearer {API_KEY}"}, 
                                  timeout=30)
        
        if poll_resp.status_code != 200:
            print(f"轮询错误: {poll_resp.text[:200]}")
            continue
        
        result = poll_resp.json()
        status = result.get("output", {}).get("task_status")
        print(f"  状态: {status}")
        
        if status == "SUCCEEDED":
            results = result.get("output", {}).get("results", [])
            if results and "url" in results[0]:
                url = results[0]["url"]
                print(f"\n结果 URL: {url}")
                
                # 下载结果
                img_resp = requests.get(url, timeout=60)
                result_img = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
                result_img.save("test_wanx_result.png")
                print("结果已保存到 test_wanx_result.png")
            break
        elif status in ("FAILED", "CANCELED", "UNKNOWN"):
            print(f"任务失败: {result}")
            break
    
    print("\n完成!")

if __name__ == "__main__":
    test_wanx_inpaint()
