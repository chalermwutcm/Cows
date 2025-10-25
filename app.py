import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path

# --- ส่วนการตั้งค่าและโหลดโมเดล ---
MODEL_PATH = Path('models/DenseNet121_cow_weight.h5') # <<--- แก้ชื่อไฟล์โมเดลตรงนี้ ถ้าใช้ตัวอื่น
IMG_SIZE = (224, 224)

# Cache การโหลดโมเดลเพื่อความเร็ว
@st.cache_resource
def load_keras_model(model_path):
    """Loads the compiled Keras model."""
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
        print(f"Error loading model: {e}")
        return None

# ฟังก์ชัน Preprocess ภาพ (ยกมาจาก w1.py)
def preprocess_image(image_bytes):
    """
    ประมวลผลภาพที่อัปโหลด (จาก Bytes)
    - Decode
    - Resize
    - Normalize
    """
    try:
        # Decode image using OpenCV
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
             raise ValueError("ไม่สามารถ decode ภาพได้")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Streamlit ใช้ RGB
        img_resized = cv2.resize(img, IMG_SIZE)
        img_normalized = img_resized.astype(np.float32) / 255.0
        return img_normalized
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการประมวลผลภาพ: {e}")
        print(f"Error preprocessing image: {e}")
        return None

# --- ส่วนหน้าเว็บ Streamlit ---
st.set_page_config(layout="wide")
st.title("🐮 ระบบประมาณน้ำหนักวัวจากภาพถ่าย")
st.write("อัปโหลดภาพถ่าย 4 มุม (ซ้าย, หน้า, หลัง, ขวา) เพื่อประมาณน้ำหนัก")

# โหลดโมเดล
model = load_keras_model(MODEL_PATH)

if model is not None:
    # สร้าง Columns สำหรับอัปโหลดไฟล์
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        uploaded_file_0 = st.file_uploader("1. ภาพด้านซ้าย (_0)", type=["png", "jpg", "jpeg"])
        if uploaded_file_0 is not None:
            image_0 = Image.open(uploaded_file_0)
            st.image(image_0, caption='ภาพด้านซ้าย', use_column_width=True)

    with col2:
        uploaded_file_1 = st.file_uploader("2. ภาพด้านหน้า (_1)", type=["png", "jpg", "jpeg"])
        if uploaded_file_1 is not None:
            image_1 = Image.open(uploaded_file_1)
            st.image(image_1, caption='ภาพด้านหน้า', use_column_width=True)

    with col3:
        uploaded_file_2 = st.file_uploader("3. ภาพด้านหลัง (_2)", type=["png", "jpg", "jpeg"])
        if uploaded_file_2 is not None:
            image_2 = Image.open(uploaded_file_2)
            st.image(image_2, caption='ภาพด้านหลัง', use_column_width=True)

    with col4:
        uploaded_file_3 = st.file_uploader("4. ภาพด้านขวา (_3)", type=["png", "jpg", "jpeg"])
        if uploaded_file_3 is not None:
            image_3 = Image.open(uploaded_file_3)
            st.image(image_3, caption='ภาพด้านขวา', use_column_width=True)

    # ปุ่มทำนาย
    if st.button("ประเมินน้ำหนัก"):
        # ตรวจสอบว่าอัปโหลดครบ 4 ภาพหรือไม่
        if uploaded_file_0 and uploaded_file_1 and uploaded_file_2 and uploaded_file_3:
            try:
                # อ่าน Bytes ของไฟล์
                img_bytes_0 = uploaded_file_0.getvalue()
                img_bytes_1 = uploaded_file_1.getvalue()
                img_bytes_2 = uploaded_file_2.getvalue()
                img_bytes_3 = uploaded_file_3.getvalue()

                # Preprocess ภาพ
                img_proc_0 = preprocess_image(img_bytes_0) # ซ้าย
                img_proc_1 = preprocess_image(img_bytes_1) # หน้า
                img_proc_2 = preprocess_image(img_bytes_2) # หลัง
                img_proc_3 = preprocess_image(img_bytes_3) # ขวา

                if all(img is not None for img in [img_proc_0, img_proc_1, img_proc_2, img_proc_3]):

                    # สร้าง Input list ให้โมเดล (ต้องเรียงตามลำดับที่โมเดลคาดหวัง: [หน้า, หลัง, ซ้าย, ขวา])
                    model_input = [
                        np.expand_dims(img_proc_1, axis=0), # หน้า
                        np.expand_dims(img_proc_2, axis=0), # หลัง
                        np.expand_dims(img_proc_0, axis=0), # ซ้าย
                        np.expand_dims(img_proc_3, axis=0)  # ขวา
                    ]

                    # ทำนายผล
                    with st.spinner('กำลังประเมินผล...'):
                        prediction = model.predict(model_input)
                        estimated_weight = prediction[0][0]

                    st.success(f"น้ำหนักวัวโดยประมาณ: **{estimated_weight:.2f} กิโลกรัม**")
                else:
                    st.warning("มีข้อผิดพลาดในการประมวลผลภาพบางส่วน กรุณาลองใหม่อีกครั้ง")

            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดระหว่างการประเมินผล: {e}")
                print(f"Prediction Error: {e}")
        else:
            st.warning("กรุณาอัปโหลดภาพให้ครบทั้ง 4 มุม")
else:
    st.error("ไม่สามารถโหลดโมเดลได้ กรุณาตรวจสอบไฟล์โมเดล")

# แสดงข้อความเล็กๆ ด้านล่าง (Optional)
st.caption("พัฒนาโดย [ชื่อของคุณ หรือ ชื่อกลุ่ม]")