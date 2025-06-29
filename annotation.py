import gradio as gr
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image, ImageDraw
import json
import os


SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
JSON_PATH = "D:/dataset/test_images/grasp_test_annotation.json"
IMAGE_ROOT = "D:/dataset/test_images"
RESULTS_DIR = "results"
EXAMPLES_DIR = "good_examples"
BAD_DIR = "bad_examples"


print(f"Loading SAM model ({MODEL_TYPE}) to {DEVICE}...")
sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
sam.to(device=DEVICE)
predictor = SamPredictor(sam)


print(f"Loading annotations from {JSON_PATH} ...")
with open(JSON_PATH, "r", encoding="utf-8") as f:
    entries = [json.loads(line) for line in f]
total_images = len(entries)


def go_to_image_by_index(user_input):
    global index
    try:
        idx = int(user_input) - 1
        if 0 <= idx < total_images:
            index = idx
            return load_image_and_prompt(index)
        else:
            print(f"⚠️ 编号超出范围：应为 1 ~ {total_images}")
    except ValueError:
        print("⚠️ 请输入有效的整数编号")

    return display_image, "", "", "**无效编号**", index + 1



index = 0
pos_points = []
neg_points = []
latest_image = None
display_image = None


point_history = []
redo_stack = []

def save_state():
    point_history.append((pos_points.copy(), neg_points.copy()))
    if len(point_history) > 100:
        point_history.pop(0)
    redo_stack.clear()


def draw_points(image_pil, pos_pts, neg_pts):
    img = image_pil.copy()
    draw = ImageDraw.Draw(img)
    radius = 3
    for (x, y) in pos_pts:
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=(0,255,0,180))
    for (x, y) in neg_pts:
        draw.line([x-radius, y-radius, x+radius, y+radius], fill=(255,0,0,180), width=2)
        draw.line([x-radius, y+radius, x+radius, y-radius], fill=(255,0,0,180), width=2)
    return img

def redraw():
    global display_image
    display_image = draw_points(latest_image, pos_points, neg_points)
    if pos_points:
        input_points = np.array(pos_points + neg_points)
        input_labels = np.array([1]*len(pos_points) + [0]*len(neg_points))
        masks, _, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False,
        )
        mask = masks[0].astype(np.uint8) * 255
        base = np.array(display_image)
        red_mask = np.stack([mask, np.zeros_like(mask), np.zeros_like(mask)], axis=-1)
        overlay = cv2.addWeighted(base, 1.0, red_mask, 1, 0)
        display_image = Image.fromarray(overlay)
    return display_image


def load_image_and_prompt(idx):
    global predictor, pos_points, neg_points, latest_image, display_image, point_history, redo_stack
    pos_points.clear()
    neg_points.clear()
    point_history.clear()
    redo_stack.clear()
    entry = entries[idx]
    img_path = os.path.join(IMAGE_ROOT, entry["image"])
    image_bgr = cv2.imread(img_path)
    if image_bgr is None:
        raise FileNotFoundError(f"图片路径找不到: {img_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    latest_image = Image.fromarray(image_rgb)
    display_image = latest_image.copy()

    if "question" not in entry:
        entry["question"] = None
    if "answer" not in entry:
        entry["answer"] = None

    try:
        with open(JSON_PATH, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"❌ 保存失败: {e}")

    index_str = f"**🖼 当前图像：第 {idx + 1} 张 / 共 {total_images} 张**"
    progress_val = idx + 1

    return display_image, entry["question"],  entry["answer"], index_str, progress_val


def on_click(image, evt: gr.SelectData, point_type):
    global pos_points, neg_points, display_image
    save_state()
    x, y = evt.index
    if point_type == "positive":
        pos_points.append((x, y))
        print(f"🟢 添加正样本点: {(x, y)}")
    else:
        neg_points.append((x, y))
        print(f"🔴 添加负样本点: {(x, y)}")
    return redraw()


def undo():
    global pos_points, neg_points, display_image
    if not point_history:
        print("⚠️ 无可撤销记录")
        return display_image
    redo_stack.append((pos_points.copy(), neg_points.copy()))
    prev_pos, prev_neg = point_history.pop()
    pos_points[:] = prev_pos
    neg_points[:] = prev_neg
    return redraw(),"↩️ 成功"

def redo():
    global pos_points, neg_points, display_image
    if not redo_stack:
        print("⚠️ 无可恢复记录")
        return display_image
    point_history.append((pos_points.copy(), neg_points.copy()))
    next_pos, next_neg = redo_stack.pop()
    pos_points[:] = next_pos
    neg_points[:] = next_neg
    return redraw(),"↪️ 成功"

def save_segmentation():
    global index, pos_points, neg_points, latest_image, display_image
    if not pos_points:
        print("⚠️ 没有正样本点，无法分割。")
        return display_image
    input_points = np.array(pos_points + neg_points)
    input_labels = np.array([1]*len(pos_points) + [0]*len(neg_points))
    masks, _, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=False,
    )
    mask = masks[0].astype(np.uint8) * 255
    rel_path = entries[index]["image"]
    base_path, _ = os.path.splitext(rel_path)
    mask_save_path = os.path.join(RESULTS_DIR, base_path + "_mask.png")
    overlay_save_path = os.path.join(RESULTS_DIR, base_path + "_overlay.png")
    os.makedirs(os.path.dirname(mask_save_path), exist_ok=True)
    Image.fromarray(mask).save(mask_save_path)
    base = np.array(latest_image)
    red_mask = np.stack([mask, np.zeros_like(mask), np.zeros_like(mask)], axis=-1)
    overlay = cv2.addWeighted(base, 1.0, red_mask, 1, 0)
    Image.fromarray(overlay).save(overlay_save_path)
    print(f"✅ 保存成功: {mask_save_path}, {overlay_save_path}")
    pos_points.clear()
    neg_points.clear()
    point_history.clear()
    redo_stack.clear()
    display_image = latest_image.copy()
    return display_image, f"✅ 保存成功: {mask_save_path}, {overlay_save_path}"

def save_example():
    global index, pos_points, neg_points, latest_image
    if not pos_points:
        print("⚠️ 没有正样本点，无法保存示例。")
        return display_image

    input_points = np.array(pos_points + neg_points)
    input_labels = np.array([1]*len(pos_points) + [0]*len(neg_points))
    masks, _, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=False,
    )
    mask = masks[0].astype(np.uint8) * 255

    rel_path = entries[index]["image"]
    base_path, _ = os.path.splitext(rel_path)


    example_base_dir = os.path.join(EXAMPLES_DIR, os.path.dirname(base_path))
    os.makedirs(example_base_dir, exist_ok=True)


    orig_save_path = os.path.join(EXAMPLES_DIR, rel_path)
    os.makedirs(os.path.dirname(orig_save_path), exist_ok=True)
    latest_image.save(orig_save_path)

    mask_save_path = os.path.join(EXAMPLES_DIR, base_path + "_mask.png")
    Image.fromarray(mask).save(mask_save_path)

    base = np.array(latest_image)
    red_mask = np.stack([mask, np.zeros_like(mask), np.zeros_like(mask)], axis=-1)
    overlay = cv2.addWeighted(base, 1.0, red_mask, 1, 0)
    overlay_save_path = os.path.join(EXAMPLES_DIR, base_path + "_overlay.png")
    Image.fromarray(overlay).save(overlay_save_path)

    json_save_path = os.path.join(EXAMPLES_DIR, base_path + ".json")
    with open(json_save_path, "w", encoding="utf-8") as f:
        json.dump(entries[index], f, ensure_ascii=False, indent=2)

    print(f"✅ 示例保存成功:\n- 原图: {orig_save_path}\n- 掩码: {mask_save_path}\n- 叠加: {overlay_save_path}\n- JSON: {json_save_path}")

    return display_image, f"✅ 示例保存成功:\n- 原图: {orig_save_path}"

def save_unlabeled_example(remark):
    global index, latest_image
    rel_path = entries[index]["image"]
    base_path, _ = os.path.splitext(rel_path)

    example_base_dir = os.path.join(BAD_DIR, os.path.dirname(base_path))
    os.makedirs(example_base_dir, exist_ok=True)

    orig_save_path = os.path.join(BAD_DIR, rel_path)
    os.makedirs(os.path.dirname(orig_save_path), exist_ok=True)
    latest_image.save(orig_save_path)

    json_save_path = os.path.join(BAD_DIR, base_path + ".json")
    with open(json_save_path, "w", encoding="utf-8") as f:
        json.dump(entries[index], f, ensure_ascii=False, indent=2)

    if remark.strip():
        remark_save_path = os.path.join(BAD_DIR, base_path + "_remark.txt")
        with open(remark_save_path, "w", encoding="utf-8") as f:
            f.write(remark.strip())
    else:
        remark_save_path = None

    msg = f"✅ 无法标注图片信息保存成功:\n- 原图: {orig_save_path}\n- JSON: {json_save_path}"
    if remark_save_path:
        msg += f"\n- 备注: {remark_save_path}"

    print(msg)
    return latest_image, msg

def update_question_answer(q_text, a_text):
    global index, entries

    entries[index]["question"] = q_text
    entries[index]["answer"] = a_text

    try:
        with open(JSON_PATH, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return "✅ 保存成功"
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        return f"❌ 保存失败: {e}"

def next_image():
    global index
    index = (index + 1) % len(entries)
    return load_image_and_prompt(index)

def prev_image():
    global index
    index = (index - 1 + len(entries)) % len(entries)
    return load_image_and_prompt(index)


with gr.Blocks() as demo:
    gr.HTML("""
    <h2 style="text-align: center;">🛠️多模态语义分割任务 数据标注工具</h2>
    <p style="text-align: center;">点击图像添加点（绿色：正样本，红叉：负样本）</p>
    <hr>
    """)


    gr.Markdown("""
       ### ✏️ 问题与答案编辑功能说明

       - 当前图片对应的 **Question** 和 **Answer** 字段是可编辑的。
       - 你可以直接在输入框中修改内容。
       - 修改后会**自动保存**到原始 JSON 文件中。

       > ⚠️ 请注意：保存会覆盖原始 JSON 文件中的对应项，请谨慎修改。
       """)

    gr.HTML("<hr>")

    gr.Markdown("### 📦 保存功能说明")

    with gr.Row():
        gr.HTML("""
           <div style="background-color: rgba(128, 128, 128, 0.6); padding: 16px; border-radius: 12px; box-shadow: 2px 2px 8px rgba(0,0,0,0.1);">
               <h4>💾 保存分割结果</h4>
               <ul style="padding-left: 1em;">
                   <li>保存掩码和叠加图</li>
                   <li>路径：<code>results/</code></li>
                   <li>用于保存标注好的图片</li>
               </ul>
           </div>
           """)
        gr.HTML("""
           <div style="background-color: rgba(128, 128, 128, 0.6); padding: 16px; border-radius: 12px; box-shadow: 2px 2px 8px rgba(0,0,0,0.1);">
               <h4>📁 保存示例图</h4>
               <ul style="padding-left: 1em;">
                   <li>保存原图、掩码、叠加图、JSON</li>
                   <li>路径：<code>good_examples/</code></li>
                   <li>用于保存你希望留下来在论文、汇报等中用作案例的好样本</li>
               </ul>
           </div>
           """)
        gr.HTML("""
           <div style="background-color: rgba(128, 128, 128, 0.6); padding: 16px; border-radius: 12px; box-shadow: 2px 2px 8px rgba(0,0,0,0.1);">
               <h4>📁 保存无法标注图片信息</h4>
               <ul style="padding-left: 1em;">
                   <li>保存原图和 JSON</li>
                   <li>路径：<code>bad_examples/</code></li>
                   <li>用于保存无法标注，不知道如何标注的坏样本，可备注原因</li>
               </ul>
           </div>
           """)


    with gr.Row():
        with gr.Column(scale=2):
            image_display = gr.Image(type="pil", label="点击添加点")
            image_index_md = gr.Markdown()
            image_progress = gr.Slider(minimum=1, maximum=total_images, step=1, value=1,
                                       interactive=True, label="图像编号（拖动跳转）")
        with gr.Column(scale=1):
            point_type = gr.Radio(choices=["positive", "negative"], value="positive", label="点的类型",
                                  interactive=True)
            status_text = gr.Textbox(label="状态提示", interactive=False)


            with gr.Group():
                question_text = gr.Textbox(label="Question", interactive=True)
                answer_text = gr.Textbox(label="Answer", interactive=True)
            with gr.Row():
                prev_btn = gr.Button("⬅️ 上一张图")
                next_btn = gr.Button("➡️ 下一张图")


            with gr.Group():
                save_btn = gr.Button("💾 保存分割结果")
                save_example_btn = gr.Button("📁 保存示例图")
                save_unlabeled_btn = gr.Button("📁 保存无法标注图片信息")
                unlabeled_reason = gr.Textbox(label="备注原因（选填）", placeholder="填写无法标注的原因，比如图像质量差等",
                                              lines=2)

                with gr.Row():
                    undo_btn = gr.Button("↩️ 撤销")
                    redo_btn = gr.Button("↪️ 重做")




    image_display.select(fn=on_click, inputs=[image_display, point_type], outputs=image_display)
    undo_btn.click(fn=undo, outputs=[image_display, status_text])
    redo_btn.click(fn=redo, outputs=[image_display, status_text])
    save_btn.click(fn=save_segmentation, outputs=[image_display,status_text])
    save_example_btn.click(fn=save_example, outputs=[image_display,status_text])
    save_unlabeled_btn.click(fn=save_unlabeled_example,
                             inputs=[unlabeled_reason],
                             outputs=[image_display, status_text])
    question_text.change(fn=update_question_answer,
                         inputs=[question_text, answer_text],
                         outputs=status_text)

    answer_text.change(fn=update_question_answer,
                       inputs=[question_text, answer_text],
                       outputs=status_text)

    prev_btn.click(fn=prev_image,
                   outputs=[image_display, question_text, answer_text, image_index_md, image_progress])
    next_btn.click(fn=next_image,
                   outputs=[image_display, question_text, answer_text, image_index_md, image_progress])

    image_progress.release(fn=go_to_image_by_index,
                          inputs=image_progress,
                          outputs=[image_display, question_text, answer_text, image_index_md, image_progress])

    demo.load(fn=lambda: load_image_and_prompt(index),
              outputs=[image_display, question_text, answer_text, image_index_md, image_progress])


demo.launch(server_name="127.0.0.1", server_port=7861,show_api=False, show_error=False,share=False)
