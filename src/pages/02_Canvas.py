import joblib
import cv2
import pandas as pd
import streamlit as st
from keras.datasets import mnist
from PIL import Image
from streamlit_drawable_canvas import st_canvas

im = Image.open("favicon.ico")

#st.set_page_config(
#    "EsKape Room",
#    im,
#    initial_sidebar_state="expanded",
#    layout="wide",
#)

if "number" not in st.session_state:
    st.session_state["number"] = 0


@st.cache_data
def get_mnist_data():
    return mnist.load_data()


def transform_image_to_mnist(image):
    # Check if the image has 4 channels (RGBA)
    st.write("Transform > Dimensiones de imagen de entrada")
    st.write(image.shape)
    if image.shape[2] == 4:
        # Remover el canal alpha
        st.write("Transform > Remover canal alpha")
        image = image[:, :, :3]
        st.write(image.shape)

    # Convertir imagen a escala de grises
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    st.write("Transform > Conversion escala de grises")
    st.write(gray_image.shape)
    
    # Undersampling de la imagen de INPUTxINPUT a 28x28
    resized_image = cv2.resize(gray_image, (28, 28), interpolation=cv2.INTER_AREA)

    # Preprocesamiento de la imagen para incrementar contraste
    equalized_image = cv2.equalizeHist(resized_image)

    # Imprimir dimensiones de salida
    st.write("Transform > Dimensiones imagen de salida")
    st.write(resized_image.shape)

    # Retornamos la imagen transformada de INPUTxINPUT a 28x28 y la imagen con contraste
    return resized_image, equalized_image

def convertir(image):
    return image.reshape(1, 28*28)

def mnist_dataset_viewer(x_train, y_train, x_test, y_test):
    st.header("Sección de mnist")
    option = st.sidebar.selectbox(
        "De cuál dataset quieres ver la imagen?", ("train", "test")
    )

    if option == "train":
        st.session_state["number"] = st.sidebar.slider(
            "Índice de la imagen en entrenamiento", 0, x_train.shape[0], 0
        )
        st.image(x_train[st.session_state["number"]], channels="gray")
    else:
        st.session_state["number"] = st.sidebar.slider(
            "Índice de la imagen en prueba", 0, x_test.shape[0], 0
        )
        st.image(x_test[st.session_state["number"]], channels="gray")

    st.write("Shape of mnist image")
    st.write(x_train[st.session_state["number"]].shape)

def play_canvas():
    # Cómo leer los datos de Keras
    (x_train, y_train), (x_test, y_test) = get_mnist_data()

    # Creando variables del sidebar
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
    realtime_update = st.sidebar.checkbox("Update in realtime", True)

    with st.container():
        (
            number_one,
            _,
            operator_one,
            number_two,
            _,
            operator_two,
            number_three,
        ) = st.columns([3, 1, 2, 3, 1, 2, 3])

        with number_one:
            c1, c2 = st.columns(2)
            with c1:
                st.empty()
            with c2:
                exponent_1 = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                    stroke_width=stroke_width,
                    stroke_color=stroke_color,
                    background_color=bg_color,
                    background_image=None,
                    update_streamlit=realtime_update,
                    height=50,
                    width=50,
                    drawing_mode="freedraw",
                    point_display_radius=0,
                    key="exponent_1",
                )

            number_1 = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_color=bg_color,
                background_image=None,
                update_streamlit=realtime_update,
                height=150,
                width=150,
                drawing_mode="freedraw",
                point_display_radius=0,
                key="number_1",
            )

        with operator_one:
            with st.container():
                st.markdown("#")
                st.markdown("#")
                operator_1 = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                    stroke_width=stroke_width,
                    stroke_color=stroke_color,
                    background_color=bg_color,
                    background_image=None,
                    update_streamlit=realtime_update,
                    height=100,
                    width=100,
                    drawing_mode="freedraw",
                    point_display_radius=0,
                    key="operator_1",
                )
        with number_two:
            c1, c2 = st.columns(2)
            with c1:
                st.empty()
            with c2:
                exponent_2 = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                    stroke_width=stroke_width,
                    stroke_color=stroke_color,
                    background_color=bg_color,
                    background_image=None,
                    update_streamlit=realtime_update,
                    height=50,
                    width=50,
                    drawing_mode="freedraw",
                    point_display_radius=0,
                    key="exponent_2",
                )
            number_2 = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_color=bg_color,
                background_image=None,
                update_streamlit=realtime_update,
                height=150,
                width=150,
                drawing_mode="freedraw",
                point_display_radius=0,
                key="number_2",
            )

        with operator_two:
            st.markdown("#")
            st.markdown("#")
            operator_2 = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_color=bg_color,
                background_image=None,
                update_streamlit=realtime_update,
                height=100,
                width=100,
                drawing_mode="freedraw",
                point_display_radius=0,
                key="operator_2",
            )

        with number_three:
            c1, c2 = st.columns(2)
            with c1:
                st.empty()
            with c2:
                exponent_3 = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                    stroke_width=stroke_width,
                    stroke_color=stroke_color,
                    background_color=bg_color,
                    background_image=None,
                    update_streamlit=realtime_update,
                    height=50,
                    width=50,
                    drawing_mode="freedraw",
                    point_display_radius=0,
                    key="exponent_3",
                )

            number_3 = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_color=bg_color,
                background_image=None,
                update_streamlit=realtime_update,
                height=150,
                width=150,
                drawing_mode="freedraw",
                point_display_radius=0,
                key="number_3",
            )

    number, exponent, operator, mnist = st.tabs(
        ["Número", "Exponente", "Operador", "Mnist"]
    )

    with number:
        st.header("Sección de Número")
        # Do something interesting with the image data and paths
        if number_1.image_data is not None:
            st.write("Image: ")
            st.image(number_1.image_data)

            st.write("Dimensiones de la imagen")
            st.write(number_1.image_data.shape)

            st.write("Matriz asociada a la imagen")
            st.write(number_1.image_data[1])

            st.write("Transforming image")


            st.write("Image Transformed: ")
            # Display the image with Streamlit
            st.image(image_mnist, channels="gray", caption="Grayscale Image")

            st.write("Image Transformed equalized: ")
            # Display the image with Streamlit
            st.image(image_mnist_eq, channels="gray", caption="Grayscale Image")

            st.write("Matriz asociada a la imagen transformada")
            st.write(image_mnist)

        if number_1.json_data is not None:
            objects = pd.json_normalize(
                number_1.json_data["objects"]
            )  # need to convert obj to str because PyArrow
            for col in objects.select_dtypes(include=["object"]).columns:
                objects[col] = objects[col].astype("str")
            st.dataframe(objects)

    with exponent:
        st.header("Sección de Exponente")
        if exponent_1.image_data is not None:
            st.write("Image: ")
            st.image(exponent_1.image_data)

            st.write("Matriz asociada a la imagen")
            st.write(exponent_1.image_data[1])

            st.write("Dimensiones de la imagen")
            st.write(exponent_1.image_data.shape)

            st.write("Transformando exponente")

            image_mnist_exp, image_mnist_exp_eq = transform_image_to_mnist(
                exponent_1.image_data
            )

            st.write("Exponente transformado: ")
            # Display the image with Streamlit
            st.image(image_mnist_exp, channels="gray", caption="Grayscale Image")

            st.write("Exponente transformado equalized: ")
            # Display the image with Streamlit
            st.image(image_mnist_exp_eq, channels="gray", caption="Grayscale Image")

            st.write("Matriz asociada al exponente transformado")
            st.write(image_mnist_exp)
        
        if exponent_1.json_data is not None:
            objects = pd.json_normalize(
                exponent_1.json_data["objects"]
            )  # need to convert obj to str because PyArrow
            for col in objects.select_dtypes(include=["object"]).columns:
                objects[col] = objects[col].astype("str")
            st.dataframe(objects)

    with operator:
        st.header("Sección de Operador")
        st.write(model_operadores)
        
        if operator_1.image_data is not None:
            st.write("Operador: ")
            st.image(operator_1.image_data)

            st.write("Matriz asociada al operador")
            st.write(operator_1.image_data[1])

            st.write("Dimensiones del operador")
            st.write(operator_1.image_data.shape)

            st.write("Transformando operador")

            image_mnist_op, image_mnist_op_eq = transform_image_to_mnist(
                operator_1.image_data
            )
            st.write(model_operadores.predict(convertir(image_mnist_op))[0])
            st.write("Exponente transformado: ")
            # Display the image with Streamlit
            st.image(image_mnist_op, channels="gray", caption="Grayscale Image")

            st.write("Exponente transformado: ")
            # Display the image with Streamlit
            st.image(image_mnist_op_eq, channels="gray", caption="Grayscale Image")

            st.write("Matriz asociada al exponente transformado")
            st.write(image_mnist_op)

        if operator_1.json_data is not None:
            objects = pd.json_normalize(
                operator_1.json_data["objects"]
            )  # need to convert obj to str because PyArrow
            for col in objects.select_dtypes(include=["object"]).columns:
                objects[col] = objects[col].astype("str")
            st.dataframe(objects)

    with mnist:
        mnist_dataset_viewer(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test
        )

    

def main():
    play_canvas()

if __name__ == "__main__":
    main()
