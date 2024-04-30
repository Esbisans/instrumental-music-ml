import cv2
import av
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from streamlit_option_menu import option_menu
from streamlit_webrtc import webrtc_streamer, WebRtcMode

from sample_utils.turn import get_ice_servers



model = tf.keras.models.load_model('models/modelo.h5')

# Etiquetas de las clases
class_names = ['Piano', 'Violin', 'Bateria', 'Congas', 'Trompeta', 'Cajon', 'Armonicas', 'Acordeon', 'Guitarra Electrica', 'Guitarra Acustica']

# Función para hacer la predicción
def predict_image(image):
    img_size = 150

    # Procesar la imagen
    img_width, img_height = 224, 224  # Tamaño deseado de la imagen
    img = Image.open(image)  # Convertir la imagen a PIL Image
    img = img.resize((img_size, img_size))  # Redimensionar la imagen
    img_array = tf.keras.preprocessing.image.img_to_array(img)  # Convertir a matriz
    img_array = tf.expand_dims(img_array, 0)  # Agregar una dimensión adicional

    # Realizar la predicción con el modelo
    predictions_single = model.predict(img_array)

    # Obtener la clase predicha y la confianza
    predicted_class_index = np.argmax(predictions_single)
    predicted_class = class_names[predicted_class_index]
    confidence_percentage = predictions_single[0][predicted_class_index] * 100

    # Mostrar el resultado
    return predicted_class, confidence_percentage

def stream_predict(frame: av.VideoFrame) -> av.VideoFrame:

    image = frame.to_ndarray(format="bgr24")

    im = Image.fromarray(image, 'RGB')
    im = im.resize((150, 150))
    img_array = np.array(im)
    img_array = np.expand_dims(img_array, 0)

    # Realizar la predicción con el modelo
    predictions_single = model.predict(img_array)

    # Obtener la clase predicha y la confianza
    predicted_class_index = np.argmax(predictions_single)
    predicted_class = class_names[predicted_class_index]
    confidence_percentage = predictions_single[0][predicted_class_index] * 100

    #print(predicted_class, confidence_percentage)
    cv2.putText(image, f"{predicted_class}, {confidence_percentage:.2f}%", (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 100, 10), 2)

    return av.VideoFrame.from_ndarray(image, format="bgr24")


def main():
    st.title("Clasificación de Instrumentos musicales")

    selected = option_menu( 
        menu_title=None,
        options = ["video en vivo","Cargar imagen"],
        orientation="horizontal",
    )


    if selected == "video en vivo":
        st.write("video en vivo")

        webrtc_ctx = webrtc_streamer(
            key="object-detection", 
            media_stream_constraints={"video": True, "audio": False},
            video_frame_callback=stream_predict,
            mode=WebRtcMode.SENDRECV,
            async_processing=True,
            rtc_configuration={
                "iceServers": get_ice_servers(),
                "iceTransportPolicy": "relay",
            },
        )

    if selected == "Cargar imagen":
        uploaded_image = st.file_uploader("Cargar una imagen", type=["jpg", "jpeg", "png"])

        if uploaded_image is not None:
            if st.button('clasificar'):
            # Obtener la predicción
                predicted_class, confidence_percentage = predict_image(uploaded_image)

            # Mostrar el resultado
                st.subheader(f"Predicción: {predicted_class}")
                st.subheader(f"Confianza: {confidence_percentage:.2f}%")

if __name__ == "__main__":
    main()