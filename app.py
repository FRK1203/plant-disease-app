import streamlit as st
import onnxruntime as ort
from PIL import Image
import numpy as np

# 类别名（顺序与训练时一致）
CLASS_NAMES = [
    'rice_brown_spot', 'rice_healthy', 'rice_leaf_blast',
    'tomato_bacterial_spot', 'tomato_early_blight', 'tomato_healthy',
    'tomato_late_blight', 'tomato_leaf_mold', 'tomato_septoria_leaf_spot',
    'tomato_spider_mites_two-spotted_spider_mite', 'tomato_target_spot',
    'tomato_tomato_mosaic_virus', 'tomato_tomato_yellow_leaf_curl_virus',
    'wheat_brown_rust', 'wheat_healthy', 'wheat_yellow_rust'
]

st.set_page_config(page_title="植物叶片病害识别", page_icon="🌿")
st.title("🌿 植物叶片疾病识别系统")
st.write("上传一张叶片照片，自动诊断病害。")


@st.cache_resource
def load_model():
    return ort.InferenceSession("model.onnx")


session = load_model()


def preprocess(image):
    # 手动实现 Resize(256) + CenterCrop(224) + ToTensor + Normalize
    image = image.resize((256, 256), Image.BILINEAR)
    # Center crop to 224x224
    left = (256 - 224) // 2
    top = (256 - 224) // 2
    image = image.crop((left, top, left + 224, top + 224))
    # To numpy array and normalize to [0, 1]
    img_array = np.array(image).astype(np.float32) / 255.0
    # Normalize with ImageNet mean/std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    # Transpose from HWC to CHW and add batch dimension
    img_array = img_array.transpose(2, 0, 1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array.astype(np.float32)


uploaded_file = st.file_uploader("选择一张叶片图片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="上传的叶片图片", width=300)

    input_array = preprocess(image)
    outputs = session.run(["output"], {"input": input_array})
    # Softmax
    logits = outputs[0]
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    prob = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    conf = float(np.max(prob))
    pred_idx = int(np.argmax(prob))
    pred_class = CLASS_NAMES[pred_idx]

    st.success(f"**诊断结果：{pred_class}**")
    st.metric("置信度", f"{conf:.2%}")

    if conf > 0.8:
        st.info("💊 请参考 PlantVillage 官方防治建议或咨询当地农技站。")

st.markdown("---")
st.caption("植物叶片疾病识别系统 | MobileNetV3 | 准确率 95%+")