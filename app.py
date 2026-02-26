"""
糖尿病视网膜病变诊断 Demo 后端
运行: python app.py
访问: http://localhost:5000
"""

import io
import os
import torch
import torch.nn as nn
import torchvision.transforms as T
import timm
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ─────────────────────────────────────────
#  配置（与 train.py 保持一致）
# ─────────────────────────────────────────
MODEL_NAME   = "tf_efficientnetv2_s"
NUM_CLASSES  = 5
IMG_SIZE     = 384
CHECKPOINT   = "./checkpoints/best_model.pth"   # 训练好的权重
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_INFO = [
    {"level": 0, "name": "No DR",           "cn": "无病变",   "color": "#22c55e"},
    {"level": 1, "name": "Mild DR",          "cn": "轻度",     "color": "#84cc16"},
    {"level": 2, "name": "Moderate DR",      "cn": "中度",     "color": "#f59e0b"},
    {"level": 3, "name": "Severe DR",        "cn": "重度",     "color": "#f97316"},
    {"level": 4, "name": "Proliferative DR", "cn": "增殖性",   "color": "#ef4444"},
]

# ─────────────────────────────────────────
#  模型定义（与 train.py 一致）
# ─────────────────────────────────────────
class EfficientNetV2Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            MODEL_NAME, pretrained=False, num_classes=0, global_pool="avg"
        )
        in_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, NUM_CLASSES),
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))


# ─────────────────────────────────────────
#  加载模型
# ─────────────────────────────────────────
def load_model():
    model = EfficientNetV2Classifier().to(DEVICE)
    if os.path.exists(CHECKPOINT):
        ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"✓ 模型加载成功: {CHECKPOINT}")
    else:
        print(f"⚠ 未找到 checkpoint: {CHECKPOINT}，使用随机权重（仅供 UI 演示）")
    model.eval()
    return model


transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])

model = load_model()

# ─────────────────────────────────────────
#  Flask App
# ─────────────────────────────────────────
app = Flask(__name__, static_folder=".")
CORS(app)


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1)[0].cpu().tolist()

        pred_idx = int(torch.tensor(probs).argmax())
        result = {
            "prediction": pred_idx,
            "class_name": CLASS_INFO[pred_idx]["name"],
            "class_cn":   CLASS_INFO[pred_idx]["cn"],
            "color":      CLASS_INFO[pred_idx]["color"],
            "probabilities": [
                {
                    "level":  info["level"],
                    "name":   info["name"],
                    "cn":     info["cn"],
                    "color":  info["color"],
                    "prob":   round(probs[i] * 100, 2),
                }
                for i, info in enumerate(CLASS_INFO)
            ]
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print("启动服务: http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
