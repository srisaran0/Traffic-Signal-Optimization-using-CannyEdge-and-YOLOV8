# Python In-built packages
from pathlib import Path
import PIL
import cv2
import numpy as np

# External packages
import streamlit as st

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Traffic Congestion Detection",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Traffic Congestion Detection")

# Sidebar
st.sidebar.header("ML Model Config")

# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Detection'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 10, 100, 40)) / 100

# Selecting Detection Or Segmentation
if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Segmentation':
    model_path = Path(settings.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
    if st.sidebar.button('Detect and Process'):
        col1, col2 = st.columns(2)

        with col1:
            try:
                if source_img is None:
                    default_image_path = str(settings.DEFAULT_IMAGE)
                    default_image = PIL.Image.open(default_image_path)
                    st.image(default_image_path, caption="Default Image", use_container_width=True)
                    image_to_process = default_image
                else:
                    uploaded_image = PIL.Image.open(source_img)
                    st.image(source_img, caption="Uploaded Image", use_container_width=True)
                    image_to_process = uploaded_image
                
                img_np = np.array(image_to_process)
                gray_image = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
                edges = cv2.Canny(blurred_image, threshold1=100, threshold2=200)
                edges_brightened = cv2.convertScaleAbs(edges, alpha=1.5, beta=50)  
                edges_pil = PIL.Image.fromarray(edges_brightened)
                st.image(edges_pil, caption="Canny Edge Detection", use_container_width=True)

            except Exception as ex:
                st.error("Error occurred while opening the image.")
                st.error(ex)

        with col2:
            if source_img is None:
                default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
                default_detected_image = PIL.Image.open(
                    default_detected_image_path)
                st.image(default_detected_image_path, caption='Detected Image',
                        use_container_width=True)
            else:
                res = model.predict(uploaded_image,
                                    conf=confidence
                                    )
                names = model.names
                #st.text(names)
                # Motorcycle
                moto_id = list(names)[list(names.values()).index('Motorcyecle')]
                moto = res[0].boxes.cls.tolist().count(moto_id)
                # st.text("Motorcycle: " + str(moto))

                # Bus
                bus_id = list(names)[list(names.values()).index('Bus')]
                bus = res[0].boxes.cls.tolist().count(bus_id)
                # st.text("Bus: " + str(bus))

                # Pickup
                pickup_id = list(names)[list(names.values()).index('Pickup')]
                pickup = res[0].boxes.cls.tolist().count(pickup_id)
                # st.text("Pickup: " + str(pickup))

                # SUV
                suv_id = list(names)[list(names.values()).index('SUV')]
                suv = res[0].boxes.cls.tolist().count(suv_id)
                # st.text("SUV: " + str(suv))

                # Suv
                suv2_id = list(names)[list(names.values()).index('Suv')]
                suv2 = res[0].boxes.cls.tolist().count(suv2_id)
                # st.text("SUV: " + str(suv2))

                # Sedan
                sedan_id = list(names)[list(names.values()).index('Sedan')]
                sedan = res[0].boxes.cls.tolist().count(sedan_id)
                # st.text("Sedan: " + str(sedan))

                # Truck
                truck_id = list(names)[list(names.values()).index('Truck')]
                truck = res[0].boxes.cls.tolist().count(truck_id)
                # st.text("Truck: " + str(truck))

                # Truck
                truck2_id = list(names)[list(names.values()).index('TRUCK')]
                truck2 = res[0].boxes.cls.tolist().count(truck2_id)
                # st.text("Truck: " + str(truck2))
                # Truck
                truck3_id = list(names)[list(names.values()).index('TUCK')]
                truck3 = res[0].boxes.cls.tolist().count(truck3_id)
                # st.text("Truck: " + str(truck3))

                # Van
                van_id = list(names)[list(names.values()).index('Van')]
                van = res[0].boxes.cls.tolist().count(van_id)
                # st.text("Van: " + str(van))

                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                        use_container_width=True)
                # Total Vehicles
                total_vehicles = moto + bus + pickup + suv + suv2 + sedan + truck + truck2 + truck3 + van
                st.success("Total Vehicles: " + str(total_vehicles))
                if total_vehicles <5:
                    st.info("Traffic Congestion: Light Congestion❌")
                elif 5< total_vehicles <15:
                    st.warning("Traffic Congestion: Normal Congestion✅")
                elif total_vehicles >15:
                    st.error("Traffic Congestion: Heavy Congestion🚨")
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    # st.write(ex)
                    st.write("No image is uploaded yet!")

elif source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model)

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)

elif source_radio == settings.RTSP:
    helper.play_rtsp_stream(confidence, model)

elif source_radio == settings.YOUTUBE:
    helper.play_youtube_video(confidence, model)

else:
    st.error("Please select a valid source type!")
