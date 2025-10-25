import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path

# --- ส่วนการตั้งค่าและโหลดโมเดล ---
MODEL_PATH = Path('models/MobileNetV2_cow_weight.h5') # <<--- Updated this line
IMG_SIZE = (224, 224)

# Cache การโหลดโมเดลเพื่อความเร็ว
@st.cache_resource
def load_keras_model(model_path):
    """Loads the compiled Keras model."""
    try:
        # Check if model file exists
        if not model_path.is_file():
            st.error(f"Error: Model file not found at {model_path}")
            print(f"Error: Model file not found at {model_path}")
            # Try to list files in the models directory for debugging
            models_dir = Path('models')
            if models_dir.is_dir():
                 print(f"Files in models directory: {list(models_dir.glob('*'))}")
            else:
                 print("Models directory not found.")
            return None
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
             raise ValueError("ไม่สามารถ decode ภาพได้ (cv2.imdecode returned None)")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert BGR (OpenCV default) to RGB
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
st.write("---") # Add a horizontal rule

# โหลดโมเดล
model = load_keras_model(MODEL_PATH)

if model is not None:
    # สร้าง Columns สำหรับอัปโหลดไฟล์
    col1, col2, col3, col4 = st.columns(4)
    uploaded_images = {} # Dictionary to store uploaded image data

    with col1:
        uploaded_file_0 = st.file_uploader("1. ภาพด้านซ้าย (_0)", type=["png", "jpg", "jpeg"], key="left")
        if uploaded_file_0 is not None:
            uploaded_images['left'] = uploaded_file_0
            image_0 = Image.open(uploaded_file_0)
            st.image(image_0, caption='ภาพด้านซ้าย', use_column_width=True)

    with col2:
        uploaded_file_1 = st.file_uploader("2. ภาพด้านหน้า (_1)", type=["png", "jpg", "jpeg"], key="front")
        if uploaded_file_1 is not None:
            uploaded_images['front'] = uploaded_file_1
            image_1 = Image.open(uploaded_file_1)
            st.image(image_1, caption='ภาพด้านหน้า', use_column_width=True)

    with col3:
        uploaded_file_2 = st.file_uploader("3. ภาพด้านหลัง (_2)", type=["png", "jpg", "jpeg"], key="back")
        if uploaded_file_2 is not None:
            uploaded_images['back'] = uploaded_file_2
            image_2 = Image.open(uploaded_file_2)
            st.image(image_2, caption='ภาพด้านหลัง', use_column_width=True)

    with col4:
        uploaded_file_3 = st.file_uploader("4. ภาพด้านขวา (_3)", type=["png", "jpg", "jpeg"], key="right")
        if uploaded_file_3 is not None:
            uploaded_images['right'] = uploaded_file_3
            image_3 = Image.open(uploaded_file_3)
            st.image(image_3, caption='ภาพด้านขวา', use_column_width=True)

    st.write("---") # Add a horizontal rule

    # ปุ่มทำนาย
    if st.button("ประเมินน้ำหนัก", key="predict_button"):
        # ตรวจสอบว่าอัปโหลดครบ 4 ภาพหรือไม่
        required_keys = ['left', 'front', 'back', 'right']
        if all(key in uploaded_images for key in required_keys):
            try:
                # อ่าน Bytes ของไฟล์
                img_bytes_0 = uploaded_images['left'].getvalue()  # ซ้าย
                img_bytes_1 = uploaded_images['front'].getvalue() # หน้า
                img_bytes_2 = uploaded_images['back'].getvalue()  # หลัง
                img_bytes_3 = uploaded_images['right'].getvalue() # ขวา

                # Preprocess ภาพ
                with st.spinner('กำลังประมวลผลภาพ...'):
                    img_proc_0 = preprocess_image(img_bytes_0) # ซ้าย
                    img_proc_1 = preprocess_image(img_bytes_1) # หน้า
                    img_proc_2 = preprocess_image(img_bytes_2) # หลัง
                    img_proc_3 = preprocess_image(img_bytes_3) # ขวา

                processed_images = [img_proc_1, img_proc_2, img_proc_0, img_proc_3] # [หน้า, หลัง, ซ้าย, ขวา]

                # Check if preprocessing was successful for all images
                if all(img is not None for img in processed_images):

                    # สร้าง Input list ให้โมเดล (ต้องเรียงตามลำดับที่โมเดลคาดหวัง: [หน้า, หลัง, ซ้าย, ขวา])
                    model_input = [np.expand_dims(img, axis=0) for img in processed_images]


                    # ทำนายผล
                    with st.spinner('กำลังประเมินผลด้วยโมเดล...'):
                        prediction = model.predict(model_input)
                        estimated_weight = prediction[0][0] # Get the single value

                    # Display result with larger font
                    st.metric(label="น้ำหนักวัวโดยประมาณ (กก.)", value=f"{estimated_weight:.2f}")
                    # st.success(f"น้ำหนักวัวโดยประมาณ: **{estimated_weight:.2f} กิโลกรัม**")

                else:
                    st.warning("มีข้อผิดพลาดในการประมวลผลภาพบางส่วน กรุณาลองอัปโหลดใหม่อีกครั้ง")

            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดระหว่างการประเมินผล: {e}")
                print(f"Prediction Error: {e}")
                # Print traceback for more details in the console
                import traceback
                traceback.print_exc()
        else:
            st.warning("กรุณาอัปโหลดภาพให้ครบทั้ง 4 มุม")
else:
    st.error("ไม่สามารถโหลดโมเดลได้ กรุณาตรวจสอบว่าไฟล์โมเดลอยู่ในโฟลเดอร์ 'models' และชื่อไฟล์ถูกต้อง")


