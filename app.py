import streamlit as st
import onnxruntime as ort
from PIL import Image
import numpy as np
from torchvision import transforms

# 类别名（保持与本地训练顺序一致）
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
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).numpy()


uploaded_file = st.file_uploader("选择一张叶片图片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="上传的叶片图片", width=300)

    input_array = preprocess(image)
    outputs = session.run(["output"], {"input": input_array})
    prob = np.exp(outputs[0]) / np.sum(np.exp(outputs[0]), axis=1, keepdims=True)
    conf = float(np.max(prob))
    pred_idx = int(np.argmax(prob))
    pred_class = CLASS_NAMES[pred_idx]

    st.success(f"**诊断结果：{pred_class}**")
    st.metric("置信度", f"{conf:.2%}")

    if conf > 0.8:
        st.info("💊 请参考 PlantVillage 官方防治建议或咨询当地农技站。")

st.markdown("---")
st.caption("植物叶片疾病识别系统 | MobileNetV3 | 准确率 95%+")