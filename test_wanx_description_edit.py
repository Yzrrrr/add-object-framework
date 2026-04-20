#!/usr/bin/env python3
"""测试 wanx 的 description_edit（指令编辑）功能 - 无需 mask"""

import os
import sys
import base64
import io
import time
import requests
from pathlib import Path
from PIL import Image

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("DASHSCOPE_API_KEY")
SUBMIT_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/image2image/image-synthesis"
POLL_URL = "https://dashscope.aliyuncs.com/api/v1/tasks"

def test_description_edit():
    """测试 wanx 指令编辑 - 不需要 mask，直接描述要添加的内容"""
    
    # 加载测试图片
    image_path = "demo/demo2.jpg"
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    print(f"原图尺寸: {orig_w}x{orig_h}")
    
    # 转换为 base64
    def to_base64(pil_img, fmt="JPEG"):
        buf = io.BytesIO()
        pil_img.save(buf, format=fmt)
        return f"data:image/{fmt.lower()};base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
    
    image_b64 = to_base64(img, "JPEG")
    
    # 构建请求 - 使用 description_edit 功能
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "X-DashScope-Async": "enable",
        "Content-Type": "application/json",
    }
    
    # 测试不同的 prompt
    prompts = [
        # 简单直接
        "在画面左下角石板地上添加一只流浪猫",
        # 更详细
        "在画面左下角添加一只橙色的流浪猫，坐姿，看向镜头",
        # 英文版本
        "Add a stray cat sitting on the stone floor in the bottom left corner",
    ]
    
    for i, prompt in enumerate(prompts):
        print(f"\n{'='*50}")
        print(f"测试 {i+1}: {prompt}")
        print(f"{'='*50}")
        
        body = {
            "model": "wanx2.1-imageedit",
            "input": {
                "function": "description_edit",
                "prompt": prompt,
                "base_image_url": image_b64,
            },
            "parameters": {
                "n": 1,
                "strength": 0.7  # 控制修改幅度
            }
        }
        
        print("提交任务...")
        
        # 提交任务
        resp = requests.post(SUBMIT_URL, headers=headers, json=body, timeout=60)
        
        if resp.status_code != 200:
            print(f"错误: {resp.status_code} - {resp.text[:300]}")
            continue
        
        data = resp.json()
        task_id = data.get("output", {}).get("task_id")
        print(f"任务 ID: {task_id}")
        
        # 轮询结果
        deadline = time.time() + 120
        
        while time.time() < deadline:
            time.sleep(3)
            
            poll_resp = requests.get(f"{POLL_URL}/{task_id}", 
                                      headers={"Authorization": f"Bearer {API_KEY}"}, 
                                      timeout=30)
            
            result = poll_resp.json()
            status = result.get("output", {}).get("task_status")
            
            if status == "SUCCEEDED":
                results = result.get("output", {}).get("results", [])
                if results and "url" in results[0]:
                    url = results[0]["url"]
                    print(f"成功! URL: {url[:80]}...")
                    
                    # 下载结果
                    img_resp = requests.get(url, timeout=60)
                    result_img = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
                    result_img.save(f"test_description_edit_{i+1}.png")
                    print(f"结果已保存到 test_description_edit_{i+1}.png")
                break
            elif status in ("FAILED", "CANCELED", "UNKNOWN"):
                print(f"任务失败: {result}")
                break

if __name__ == "__main__":
    test_description_edit()
