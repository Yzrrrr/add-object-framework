# Add Object Framework

全链路 AIGC 图像生产框架 - 实现 Add Object 功能，核心保证 **PixelAlignment**（像素对齐）

> 作者：易泽仁 | 时间：2024年4月

---

## 目录

- [问题定义](#问题定义)
- [解决思路](#解决思路)
- [PixelAlignment 核心](#pixelalignment-核心加分项)
- [框架设计](#框架设计)
- [模块详解](#模块详解)
- [技术选型](#技术选型)
- [使用方法](#使用方法)
- [评估指标](#评估指标)
- [扩展性设计](#扩展性设计)

---

## 问题定义

### 核心任务

实现 **Add Object**（添加物体）的图像编辑指令

### 关键要求

|| 要求 | 描述 | 难度 |
||------|------|------|
|| **PixelAlignment** | 编辑前后原图物体不发生像素偏移 | ⭐⭐⭐⭐⭐ |
|| **自然度** | 添加物体自然，符合原图语义，无AI感 | ⭐⭐⭐⭐ |
|| **位置合理** | 新物体位置符合场景语义 | ⭐⭐⭐ |

### 技术挑战

```
传统 Inpainting 模型的问题：
┌─────────────────────────────────────────────────────┐
│  1. VAE 编码/解码误差 → 非编辑区域像素偏移          │
│  2. Latent Space 操作 → 全局语义改变                │
│  3. 缺乏位置理解 → 新物体位置可能不合理             │
│  4. 边界融合问题 → 新旧内容过渡不自然               │
└─────────────────────────────────────────────────────┘
```

---

## 解决思路

### 整体策略

```
问题分解：
├── 位置决策 → 使用多模态LLM理解场景，推荐合理位置
├── 编辑执行 → 调用 Wanx2.1-ImageEdit API 生成新物体
├── 像素对齐 → 信任 API 返回结果（已保证非 mask 区域不变）
└── 质量评估 → 多维度评估，验证效果
```

### 关键创新

**Wanx2.1-ImageEdit API 的优势**

阿里云万象图像编辑 API 提供了 `description_edit_with_mask` 功能，原生支持：
- Mask 区域内的精准生成
- 非 Mask 区域的像素保持不变
- 高质量的物体融合效果

```python
# API 调用示例
{
    "model": "wanx2.1-imageedit",
    "input": {
        "function": "description_edit_with_mask",
        "prompt": "一只真实的流浪猫坐在石板地上",
        "base_image_url": image_base64,
        "mask_image_url": mask_base64,
    }
}
```

---

## PixelAlignment 核心加分项

### Wanx API 的 PixelAlignment 保证

Wanx `description_edit_with_mask` API 原生保证非编辑区域不变：

```
Wanx API 流程：

原图 + Mask + Prompt → Wanx API → 结果
                                    │
                                    ▼
                         非 Mask 区域 = 原图像素（API 保证）
                         Mask 区域 = 新生成的物体
```

### 验证机制

虽然 API 承诺像素对齐，但我们仍需验证：

```python
def verify_alignment(original, result, mask, threshold=0.995):
    """
    验证 PixelAlignment 是否达标
    
    标准：非编辑区域像素相似度 >= 99.5%
    """
    non_edit = mask < 127
    diff = abs(original[non_edit] - result[non_edit])
    alignment = 1 - mean(diff) / 255
    
    return alignment >= threshold
```

### 实际测试结果

| Demo | 物体 | PixelAlignment | 效果 |
|------|------|----------------|------|
| demo1 | 海鸥 | 0.9976 | ✅ 物体成功保留 |
| demo2 | 流浪猫 | 0.9992 | ✅ 物体成功保留 |
| demo3 | 小花朵 | 0.9989 | ✅ 物体成功保留 |
| demo4 | 白色花朵 | 0.9964 | ✅ 物体成功保留 |
| demo5 | 老鹰 | 0.9984 | ✅ 物体成功保留 |

---

## 框架设计

### 架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Add Object Pipeline                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │   Input      │    │  Analysis    │    │   Planning   │              │
│  │   Module     │───►│   Module     │───►│    Module    │              │
│  │              │    │              │    │              │              │
│  │  图像 + 指令  │    │ Qwen2.5-VL   │    │ 掩码 + 提示词 │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│                                                  │                       │
│                                                  ▼                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │   Output     │    │    Blend     │    │    Edit      │              │
│  │   Module     │◄───│   Module     │◄───│    Module    │              │
│  │              │    │   (验证)      │    │              │              │
│  │ 结果 + 评估   │    │              │    │ Wanx2.1 API  │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 数据流

```
输入: image.jpg + "add a cat"
        │
        ▼
    [Analysis Module]
        │ scene_description: "A street scene..."
        │ suggested_position: {x: 50, y: 60, width: 20, height: 25}
        │ positive_prompt: "a realistic stray cat..."
        │ negative_prompt: "blurry, low quality..."
        ▼
    [Planning Module]  
        │ mask: [[0,0,...,255,255,...]] (椭圆掩码)
        │ dilation: 扩大掩码确保覆盖边界
        ▼
    [Edit Module]
        │ backend: "wanx"
        │ API: description_edit_with_mask
        │ result: edited_image
        ▼
    [Blend Module] ← 仅做验证，不执行混合
        │ verify_alignment(): 检查像素对齐
        │ result: 直接返回 Wanx 输出
        ▼
    [Output Module]
        │ metrics: {pixel_alignment: 0.997, boundary_quality: 0.0}
        │ files: {result.png, mask.png, comparison.png, metrics.json}
        ▼
输出: results/demo1_*
```

---

## 模块详解

### 1. Analysis Module（分析模块）

**职责**：理解场景，推荐位置

**技术方案**：
- `predefined` 模式：预定义的 demo 图分析（快速测试）
- `vlm_local` 模式：本地 Qwen2.5-VL 模型（隐私保护）

**输入输出**：
```python
输入:
  - image_path: 图像路径
  - instruction: 用户指令 (可选)

输出:
  - object_to_add: 要添加的物体
  - scene_description: 场景描述
  - suggested_position: {x, y, width, height} (百分比坐标)
  - positive_prompt: 正向提示词
  - negative_prompt: 负向提示词
  - edit_instruction: 编辑指令
```

### 2. Planning Module（规划模块）

**职责**：生成掩码，准备提示词

**关键技术**：
- 百分比坐标 → 像素坐标转换
- 椭圆掩码：比矩形更自然
- 掩码膨胀：确保覆盖物体边界

```python
# 掩码生成示例
center = (position.x * width / 100, position.y * height / 100)
size = (position.width * width / 200, position.height * height / 200)
cv2.ellipse(mask, center, size, 0, 0, 360, 255, -1)
# 膨胀确保边界覆盖
cv2.dilate(mask, kernel, iterations=1)
```

### 3. Edit Module（编辑模块）

**职责**：调用 Wanx API 执行编辑

**API 流程**：
```
1. 调整图像尺寸（512-1536 边长范围）
2. 图像和掩码编码为 base64
3. POST 异步任务 → 获取 task_id
4. 轮询任务状态直到 SUCCEEDED
5. 下载结果图像
6. 调整回原始尺寸
```

### 4. Blend Module（混合模块）

**职责**：验证 PixelAlignment

**当前实现**：
- 直接信任 Wanx API 输出
- 仅执行像素对齐验证
- 不执行额外的混合操作

### 5. Output Module（输出模块）

**职责**：质量评估，结果导出

**输出文件**：
```
results/
├── demo1_result.png         # 最终结果
├── demo1_mask.png           # 编辑掩码
├── demo1_comparison.png     # 对比图（原图|编辑|结果）
├── demo2_result.png
├── ...
└── report.json              # 汇总报告
```

---

## 技术选型

### 当前技术栈

| 功能 | 技术方案 | 理由 |
|------|----------|------|
| 场景分析 | Qwen2.5-VL (本地) | 隐私保护，免费 |
| 图像编辑 | Wanx2.1-ImageEdit | 原生支持 mask-based inpainting |
| 像素对齐 | API 原生保证 | Wanx 承诺非 mask 区域不变 |
| 质量评估 | 自研指标 | 针对 PixelAlignment 定制 |

### Wanx2.1-ImageEdit 优势

- ✅ **原生 PixelAlignment**：API 层面保证非编辑区域不变
- ✅ **高质量生成**：物体自然，边界融合好
- ✅ **简单易用**：只需图像、掩码、提示词
- ✅ **成本可控**：按调用计费

---

## 使用方法

### 安装依赖

```bash
cd add-object-framework
pip install -r requirements.txt
```

### 配置 API Key

```bash
cp .env.example .env
# 编辑 .env 填入你的 DASHSCOPE_API_KEY
```

获取 API Key: https://bailian.console.aliyun.com/

### 运行 Demo

```bash
# 运行所有 demo 图片
python3 scripts/run_demo.py

# 运行单张图片
python3 scripts/run_demo.py --image demo/demo1.jpg

# 使用本地 VLM 分析
python3 scripts/run_demo.py --analysis vlm_local
```

### 代码调用

```python
from src.pipeline import AddObjectPipeline, PipelineConfig

# 初始化 pipeline
config = PipelineConfig(
    analysis_mode="predefined",  # 或 "vlm_local"
    output_dir="results"
)
pipeline = AddObjectPipeline(config)

# 处理单张图片
result = pipeline.run(
    image_path="demo/demo1.jpg",
    output_name="demo1"
)

print(f"PixelAlignment: {result.metrics.pixel_alignment:.4f}")
print(f"Output: {result.files}")
```

---

## 评估指标

### PixelAlignment（核心指标）

**定义**：非编辑区域像素相似度

**计算公式**：
```
alignment = 1 - mean(|original - result|) / 255

范围: [0, 1]
目标: >= 0.995 (99.5% 相同)
```

**意义**：
- `1.0` = 完美对齐
- `0.995+` = 优秀，肉眼无法察觉差异
- `< 0.99` = 可能存在轻微偏移

### Boundary Quality（边界质量）

**定义**：编辑区域边界的自然程度

**当前状态**：暂未实现（值为 0.0）

### Overall Quality（综合评分）

**计算方式**：
```
overall = pixel_alignment * 0.7 + 0.3 * base_score
```

---

## 扩展性设计

### 添加新的编辑后端

```python
# 在 src/api/ 下新建文件
class MyCustomBackend:
    def inpaint(self, image, mask, prompt, negative_prompt=""):
        # 实现编辑逻辑
        return EditResult(image=edited, backend="custom")

# 在 src/modules/editor.py 中注册
from ..api.custom import MyCustomBackend

class EditModule:
    def __init__(self, backend="wanx"):
        if backend == "custom":
            self.backend = MyCustomBackend()
```

### 添加新的分析后端

```python
# 在 src/modules/analysis.py 中扩展
class AnalysisModule:
    def analyze(self, image_path, instruction=None):
        if self.mode == "my_analyzer":
            return self._my_custom_analysis(image_path)
```

---

## 项目结构

```
add-object-framework/
├── README.md                    # 本文档
├── requirements.txt             # Python 依赖
├── .env.example                 # 环境变量模板
├── src/
│   ├── __init__.py
│   ├── pipeline.py              # 主流程
│   ├── api/
│   │   ├── wanx.py              # Wanx2.1-ImageEdit API
│   │   └── qwen_vl_local.py     # 本地 Qwen2.5-VL
│   ├── modules/
│   │   ├── analysis.py          # 场景分析
│   │   ├── planning.py          # 编辑规划
│   │   ├── editor.py            # 编辑执行
│   │   ├── blender.py           # 像素验证
│   │   └── output.py            # 结果输出
│   └── utils/
│       ├── image_utils.py       # 图像工具
│       └── logger.py            # 日志工具
├── scripts/
│   └── run_demo.py              # Demo 脚本
├── demo/                        # 测试图片
│   ├── demo1.jpg
│   ├── demo2.jpg
│   ├── demo3.jpg
│   ├── demo4.jpg
│   └── demo5.jpg
└── results/                     # 输出结果
    ├── demo1_result.png
    ├── demo1_mask.png
    ├── demo1_comparison.png
    └── report.json
```

---

## 参考文献

1. Wanx Image Edit API - https://help.aliyun.com/zh/model-studio/developer-reference/wanx-image-edit-api-reference
2. Add-it: Training-Free Object Insertion in Images (NVIDIA 2024) - https://arxiv.org/abs/2411.07232
3. Qwen2.5-VL Technical Report - https://arxiv.org/abs/2412.15115

---

## 总结

本框架使用阿里云 Wanx2.1-ImageEdit API 实现 Add Object 功能：

**关键优势**：
- ✅ API 原生保证 PixelAlignment
- ✅ 模块化设计，易于扩展
- ✅ 支持本地 VLM 分析
- ✅ 完善的质量验证体系

**适用场景**：
- 📷 图像内容创作
- 🛒 电商产品图编辑
- 🎨 设计辅助工具
- 🤖 自动化图像生产
