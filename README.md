
# 🛠️ 多模态语义分割任务：数据标注工具

本项目基于 Segment Anything Model (SAM) 和 Gradio 构建，提供了一款多模态数据标注工具，旨在 可视化呈现 JSON 中的 "问题" 和 "答案" 字段，用户可以根据这些字段直观地进行分割掩码标注，从而提升标注的 效率 和 精准度。用户可以通过点击图像快速完成正负样本的标注，同时支持对语义问题与答案的编辑。
## ✨ 功能特点

- ✅ **支持交互式点选分割**
- ✅ **分类保存示例图的Overlay和Binary与无法标注图像**
- ✅ **支持编辑并自动保存 Question / Answer 到原 JSON**

![演示](./example.gif)

---



## 🔧 路径配置说明

你可以根据自己的数据集位置修改以下两个路径变量：

```python
JSON_PATH = "D:/dataset/test_images/grasp_test_annotation.json"  # JSON 标注文件路径
IMAGE_ROOT = "D:/dataset/test_images"                             # 原始图像所在目录
```
test_images 里可以有多个子文件夹。
只要 json 中 `"image"` 字段的路径和 `IMAGE_ROOT` 对齐，就没问题。

---

## 📁 数据格式说明

`grasp_test_annotation.json` 文件应遵循**每行一个 JSON 对象**的格式，字段说明如下：
- `"image"` 图片的相对路径
- `”question"` 对图片的描述或提问 
- `Answer` 对应问题的文本答案 

注：`”question"`和`Answer` **可以为 None**，如果字段不存在，系统会自动填充为 None。在标注工具界面填写后，内容会自动保存到原始的 JSON 文件中。
示例：

```json
{"image": "grasp/0001.jpg", "question": "图中是什么目标？", "answer": "左边的车"}
{"image": "grasp/0002.jpg", "question": "图中是什么目标？", "answer": "右边的车"}
```

---
## 🚀 启动方式

### 启动标注工具

```bash
python sam_annotation_tool.py
```

默认会启动本地服务在 `http://127.0.0.1:7861`，打开浏览器访问即可开始标注。

---

## 🖱️ 使用方式

### ✅ 添加标注点

- 左侧点击图像添加正（绿色圆点）或负样本点（红色叉）
- 可撤销 / 重做（支持多步）

### 📷 导航图像

- 使用左右按钮或进度条跳转图片
- 当前图像编号会显示在下方

### 📝 编辑 Q&A

- 右侧支持直接编辑该图片对应的 **Question** 和 **Answer**
- 每次修改会自动保存到原始 JSON 文件中

### 💾 保存方式

| 功能                       | 说明                                      | 保存路径         |
|--------------------------|-------------------------------------------|------------------|
| 保存数据标注                   | 保存 mask.png 和 overlay.png             | `results/`       |
| 保存你希望留下来在论文、汇报等中用作案例的好样本 | 保存原图 + 掩码 + 叠加图 + JSON           | `good_examples/` |
| 保存无法标注的坏样本               | 保存原图 + JSON + 备注文本（可选）        | `bad_examples/`  |



---

## 🧠 模型说明

本工具使用的是 Meta AI 发布的 [SAM](https://github.com/facebookresearch/segment-anything) 模型，确保你下载了权重并放置在配置位置：

```python
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"
```


