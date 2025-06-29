import gradio as gr
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image, ImageDraw
import json
import os
import logging

from config import (
    SAM_CHECKPOINT, MODEL_TYPE, JSON_PATH, IMAGE_ROOT, RESULTS_DIR,
    GOOD_EXAMPLES_DIR, BAD_EXAMPLES_DIR, POINT_HISTORY_LIMIT,
    SERVER_NAME, SERVER_PORT, SHOW_API, SHOW_ERROR, SHARE
)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AnnotationTool:
    def __init__(self):
        self.index = 0
        self.pos_points = []
        self.neg_points = []
        self.latest_image = None
        self.display_image = None
        self.point_history = []
        self.redo_stack = []
        self.entries = self._load_annotations()
        self.total_images = len(self.entries)

        logging.info(f"Loading SAM model ({MODEL_TYPE}) to {self._get_device()}...")
        self.sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
        self.sam.to(device=self._get_device())
        self.predictor = SamPredictor(self.sam)

    def _get_device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load_annotations(self):
        logging.info(f"Loading annotations from {JSON_PATH} ...")
        try:
            with open(JSON_PATH, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f]
        except FileNotFoundError:
            logging.error(f"Annotation file not found: {JSON_PATH}")
            return []
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from {JSON_PATH}: {e}")
            return []

    def _save_annotations(self):
        try:
            with open(JSON_PATH, "w", encoding="utf-8") as f:
                for entry in self.entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            return "✅ 保存成功"
        except Exception as e:
            logging.error(f"❌ 保存失败: {e}")
            return f"❌ 保存失败: {e}"

    def _save_image_assets(self, mask, base_path, target_dir):
        os.makedirs(os.path.join(target_dir, os.path.dirname(base_path)), exist_ok=True)

        orig_save_path = os.path.join(target_dir, base_path + os.path.splitext(self.entries[self.index]["image"])[1])
        self.latest_image.save(orig_save_path)

        mask_save_path = os.path.join(target_dir, base_path + "_mask.png")
        Image.fromarray(mask).save(mask_save_path)

        base = np.array(self.latest_image)
        red_mask = np.stack([mask, np.zeros_like(mask), np.zeros_like(mask)], axis=-1)
        overlay = cv2.addWeighted(base, 1.0, red_mask, 1, 0)
        overlay_save_path = os.path.join(target_dir, base_path + "_overlay.png")
        Image.fromarray(overlay).save(overlay_save_path)

        return orig_save_path, mask_save_path, overlay_save_path

    def go_to_image_by_index(self, user_input):
        try:
            idx = int(user_input) - 1
            if 0 <= idx < self.total_images:
                self.index = idx
                return self.load_image_and_prompt(self.index)
            else:
                msg = f"⚠️ 编号超出范围：应为 1 ~ {self.total_images}"
                logging.warning(msg)
                return self.display_image, self.entries[self.index].get("question", ""), self.entries[self.index].get("answer", ""), f"**{msg}**", self.index + 1
        except ValueError:
            msg = "⚠️ 请输入有效的整数编号"
            logging.warning(msg)
            return self.display_image, self.entries[self.index].get("question", ""), self.entries[self.index].get("answer", ""), f"**{msg}**", self.index + 1

    def save_state(self):
        self.point_history.append((self.pos_points.copy(), self.neg_points.copy()))
        if len(self.point_history) > POINT_HISTORY_LIMIT:
            self.point_history.pop(0)
        self.redo_stack.clear()

    def draw_points(self, image_pil, pos_pts, neg_pts):
        img = image_pil.copy()
        draw = ImageDraw.Draw(img)
        radius = 3
        for (x, y) in pos_pts:
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=(0,255,0,180))
        for (x, y) in neg_pts:
            draw.line([x-radius, y-radius, x+radius, y+radius], fill=(255,0,0,180), width=2)
            draw.line([x-radius, y+radius, x+radius, y-radius], fill=(255,0,0,180), width=2)
        return img

    def redraw(self):
        self.display_image = self.draw_points(self.latest_image, self.pos_points, self.neg_points)
        if self.pos_points:
            input_points = np.array(self.pos_points + self.neg_points)
            input_labels = np.array([1]*len(self.pos_points) + [0]*len(self.neg_points))
            masks, _, _ = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=False,
            )
            mask = masks[0].astype(np.uint8) * 255
            base = np.array(self.display_image)
            red_mask = np.stack([mask, np.zeros_like(mask), np.zeros_like(mask)], axis=-1)
            overlay = cv2.addWeighted(base, 1.0, red_mask, 1, 0)
            self.display_image = Image.fromarray(overlay)
        return self.display_image

    def load_image_and_prompt(self, idx):
        self.pos_points.clear()
        self.neg_points.clear()
        self.point_history.clear()
        self.redo_stack.clear()
        entry = self.entries[idx]
        img_path = os.path.join(IMAGE_ROOT, entry["image"])
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            msg = f"图片路径找不到: {img_path}"
            logging.error(msg)
            raise FileNotFoundError(msg)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)
        self.latest_image = Image.fromarray(image_rgb)
        self.display_image = self.latest_image.copy()

        index_str = f"**🖼 当前图像：第 {idx + 1} 张 / 共 {self.total_images} 张**"
        progress_val = idx + 1

        return self.display_image, entry.get("question", ""),  entry.get("answer", ""), index_str, progress_val

    def on_click(self, image, evt: gr.SelectData, point_type):
        self.save_state()
        x, y = evt.index
        if point_type == "positive":
            self.pos_points.append((x, y))
            logging.info(f"🟢 添加正样本点: {(x, y)}")
        else:
            self.neg_points.append((x, y))
            logging.info(f"🔴 添加负样本点: {(x, y)}")
        return self.redraw()

    def undo(self):
        if not self.point_history:
            msg = "⚠️ 无可撤销记录"
            logging.warning(msg)
            return self.display_image, msg
        self.redo_stack.append((self.pos_points.copy(), self.neg_points.copy()))
        prev_pos, prev_neg = self.point_history.pop()
        self.pos_points[:] = prev_pos
        self.neg_points[:] = prev_neg
        return self.redraw(),"↩️ 成功"

    def redo(self):
        if not self.redo_stack:
            msg = "⚠️ 无可恢复记录"
            logging.warning(msg)
            return self.display_image, msg
        self.point_history.append((self.pos_points.copy(), self.neg_points.copy()))
        next_pos, next_neg = self.redo_stack.pop()
        self.pos_points[:] = next_pos
        self.neg_points[:] = next_neg
        return self.redraw(),"↪️ 成功"

    def save_segmentation(self):
        if not self.pos_points:
            msg = "⚠️ 没有正样本点，无法分割。"
            logging.warning(msg)
            return self.display_image, msg
        input_points = np.array(self.pos_points + self.neg_points)
        input_labels = np.array([1]*len(self.pos_points) + [0]*len(self.neg_points))
        masks, _, _ = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False,
        )
        mask = masks[0].astype(np.uint8) * 255
        rel_path = self.entries[self.index]["image"]
        base_path, _ = os.path.splitext(rel_path)

        mask_save_path = os.path.join(RESULTS_DIR, base_path + "_mask.png")
        overlay_save_path = os.path.join(RESULTS_DIR, base_path + "_overlay.png")

        os.makedirs(os.path.dirname(mask_save_path), exist_ok=True)
        Image.fromarray(mask).save(mask_save_path)

        base = np.array(self.latest_image)
        red_mask = np.stack([mask, np.zeros_like(mask), np.zeros_like(mask)], axis=-1)
        overlay = cv2.addWeighted(base, 1.0, red_mask, 1, 0)
        Image.fromarray(overlay).save(overlay_save_path)

        msg = f"✅ 保存成功: {mask_save_path}, {overlay_save_path}"
        logging.info(msg)
        self.pos_points.clear()
        self.neg_points.clear()
        self.point_history.clear()
        self.redo_stack.clear()
        self.display_image = self.latest_image.copy()
        return self.display_image, msg

    def save_example(self):
        if not self.pos_points:
            msg = "⚠️ 没有正样本点，无法保存示例。"
            logging.warning(msg)
            return self.display_image, msg

        input_points = np.array(self.pos_points + self.neg_points)
        input_labels = np.array([1]*len(self.pos_points) + [0]*len(self.neg_points))
        masks, _, _ = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False,
        )
        mask = masks[0].astype(np.uint8) * 255

        rel_path = self.entries[self.index]["image"]
        base_path, _ = os.path.splitext(rel_path)

        orig_save_path, mask_save_path, overlay_save_path = self._save_image_assets(mask, base_path, GOOD_EXAMPLES_DIR)

        json_save_path = os.path.join(GOOD_EXAMPLES_DIR, base_path + ".json")
        with open(json_save_path, "w", encoding="utf-8") as f:
            json.dump(self.entries[self.index], f, ensure_ascii=False, indent=2)

        msg = f"✅ 示例保存成功:\n- 原图: {orig_save_path}\n- 掩码: {mask_save_path}\n- 叠加: {overlay_save_path}\n- JSON: {json_save_path}"
        logging.info(msg)

        return self.display_image, msg

    def save_unlabeled_example(self, remark):
        rel_path = self.entries[self.index]["image"]
        base_path, _ = os.path.splitext(rel_path)

        os.makedirs(os.path.join(BAD_EXAMPLES_DIR, os.path.dirname(base_path)), exist_ok=True)

        orig_save_path = os.path.join(BAD_EXAMPLES_DIR, rel_path)
        self.latest_image.save(orig_save_path)

        json_save_path = os.path.join(BAD_EXAMPLES_DIR, base_path + ".json")
        with open(json_save_path, "w", encoding="utf-8") as f:
            json.dump(self.entries[self.index], f, ensure_ascii=False, indent=2)

        remark_save_path = None
        if remark.strip():
            remark_save_path = os.path.join(BAD_EXAMPLES_DIR, base_path + "_remark.txt")
            with open(remark_save_path, "w", encoding="utf-8") as f:
                f.write(remark.strip())

        msg = f"✅ 无法标注图片信息保存成功:\n- 原图: {orig_save_path}\n- JSON: {json_save_path}"
        if remark_save_path:
            msg += f"\n- 备注: {remark_save_path}"

        logging.info(msg)
        return self.latest_image, msg

    def update_question_answer(self, q_text, a_text):
        self.entries[self.index]["question"] = q_text
        self.entries[self.index]["answer"] = a_text
        return self._save_annotations()

    def next_image(self):
        self.index = (self.index + 1) % len(self.entries)
        return self.load_image_and_prompt(self.index)

    def prev_image(self):
        self.index = (self.index - 1 + len(self.entries)) % len(self.entries)
        return self.load_image_and_prompt(self.index)


annotation_tool = AnnotationTool()

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
            image_progress = gr.Slider(minimum=1, maximum=annotation_tool.total_images, step=1, value=1,
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




    image_display.select(fn=annotation_tool.on_click, inputs=[image_display, point_type], outputs=image_display)
    undo_btn.click(fn=annotation_tool.undo, outputs=[image_display, status_text])
    redo_btn.click(fn=annotation_tool.redo, outputs=[image_display, status_text])
    save_btn.click(fn=annotation_tool.save_segmentation, outputs=[image_display,status_text])
    save_example_btn.click(fn=annotation_tool.save_example, outputs=[image_display,status_text])
    save_unlabeled_btn.click(fn=annotation_tool.save_unlabeled_example,
                             inputs=[unlabeled_reason],
                             outputs=[image_display, status_text])
    question_text.change(fn=annotation_tool.update_question_answer,
                         inputs=[question_text, answer_text],
                         outputs=status_text)

    answer_text.change(fn=annotation_tool.update_question_answer,
                       inputs=[question_text, answer_text],
                       outputs=status_text)

    prev_btn.click(fn=annotation_tool.prev_image,
                   outputs=[image_display, question_text, answer_text, image_index_md, image_progress])
    next_btn.click(fn=annotation_tool.next_image,
                   outputs=[image_display, question_text, answer_text, image_index_md, image_progress])

    image_progress.release(fn=annotation_tool.go_to_image_by_index,
                          inputs=image_progress,
                          outputs=[image_display, question_text, answer_text, image_index_md, image_progress])

    demo.load(fn=lambda: annotation_tool.load_image_and_prompt(annotation_tool.index),
              outputs=[image_display, question_text, answer_text, image_index_md, image_progress])


demo.launch(server_name=SERVER_NAME, server_port=SERVER_PORT, show_api=SHOW_API, show_error=SHOW_ERROR, share=SHARE)


