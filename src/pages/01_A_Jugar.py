import joblib
import cv2
import pandas as pd
import numpy as np
import streamlit as st
from keras.datasets import mnist
from PIL import Image
from streamlit_drawable_canvas import st_canvas

svm_numeros = joblib.load("src/models/output/svm_mnist.joblib")
model_operadores = joblib.load("src/models/output/svm_model_operadores.joblib")

im = Image.open("favicon.ico")
#st.set_page_config(
#    "EsKape Room",
#    im,
#    initial_sidebar_state="expanded",
#    layout="wide",
#)

def explanation():
    st.header("Página principal")

    st.subheader("Objetivo")

    st.write("""
             El objetivo es de la tarea es habilitar la sección `A jugar` para que tengamos un panel como el siguiente:
    """)

    st.image("src/img/canvas.png")

    st.write("""en el cuál podamos ejecutar una operación matemática sencilla.

    Tenemos entonces tres tipos de input en nuestro canvas:""")

    st.image("src/img/canvas2.png")

    st.write("""1. Exponentes: 3 posibles referentes a los cuadrados morados. Deben ser números del 0 al 9.
    2. Operadores: 2 posibles referentes a los cuadrados azules. Explicados en la siguiente sección.
    3. Números: 3 posibles referentes a los cuadrados rojos. Deben ser números del 0 al 9.""")

    st.subheader("Operadores")

    st.write("""Solo vamos a usar las 4 operaciones fundamentales: suma, resta, multiplicación y división. 

    En el caso de suma y resta las únicas opciones posibles son: + (ASCII Code 43) y - (ASCII Code 45), respectivamente.

    En el caso de multiplicación y división tendremos 2 opciones como sigue:""")

    st.subheader("Multiplicación")

    st.write("""Una × (ASCII Code 215) o un asterísco * (ASCII Code 42)""")

    st.image("src/img/mult2.png")


    st.subheader("División")

    st.write("""Un slash / (ASCII code 47) o el operando convencional ÷ (ASCII code 247)""")

    st.image("src/img/div1.png")


    st.subheader("Comentarios")

    st.subheader("Sobre las operaciones")

    st.write("""1. Asumimos que la aplicación siempre será usada por un agente honesto. No se debe validar para datos que no sean los referentes al modelo (aunque es un problema interesante de resolver)
    2. Somos consistentes en la entrada de cada canvas así como en el orden de las operaciones: de izquierda a derecha y con prioridad de operadores: ^, ( *, /), (+, -).""")

    st.subheader("Sobre la parte visual")

    st.write("""Escoger las secciones útiles de 02_Canvas.py y crear la vista referente a cada uno de los elementos de entrada:

    1. 3 Coeficientes
    2. 3 exponentes
    3. 2 operadores

    Para luego llamar a los modelos y evaluar la función.""")

def transform_image_to_mnist(image):
    # Check if the image has 4 channels (RGBA)

    if image.shape[2] == 4:
        # Remover el canal alpha
        image = image[:, :, :3]

    # Convertir imagen a escala de grises
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Undersampling de la imagen de INPUTxINPUT a 28x28
    resized_image = cv2.resize(gray_image, (28, 28), interpolation=cv2.INTER_AREA)

    # Preprocesamiento de la imagen para incrementar contraste
    equalized_image = cv2.equalizeHist(resized_image)

    # Retornamos la imagen transformada de INPUTxINPUT a 28x28 y la imagen con contraste
    return resized_image, equalized_image

def convertirOperador(image):
    return image.reshape(1, 28*28)

def convertirImagen(image):
    # Convertir la imagen a un array de numpy
    imagen_array = np.array(image)
    
    # Normalizar los valores
    imagen_normalizada = imagen_array.astype('float32') / 255
    
    # Redimensionar
    imagen_normalizada = imagen_normalizada.reshape(1, -1)  # Aplanar y agregar dimensión de lote
    
    return imagen_normalizada

def user_panel():
    # Esta sección de código debería ser st.write(Path("src/md/Objetivo.md").read_text())
    # Pero streamlit no soporta imágenes dentro de markdowns
    #explanation()
    
    #######################################################
    #               INGRESAR CÓDIGO ACÁ                   #
    #######################################################   
    
    # Creando variables del sidebar
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ", "#FFFFFF")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#000000")
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
            
    st.header("Prediccion")
    
    # Procesar number_1
    image_mnist_number, _ = transform_image_to_mnist(number_1.image_data)
    number1 = svm_numeros.predict(convertirImagen(image_mnist_number))[0]

    # Procesar operator_1
    image_mnist_op, image_mnist_op_eq = transform_image_to_mnist(operator_1.image_data)
    operador1 = model_operadores.predict(convertirOperador(image_mnist_op_eq))[0]

    # Procesar number_2
    image_mnist_number, _ = transform_image_to_mnist(number_2.image_data)
    number2 = svm_numeros.predict(convertirImagen(image_mnist_number))[0]

    # Procesar operator_2
    image_mnist_op, image_mnist_op_eq = transform_image_to_mnist(operator_2.image_data)
    operador2 = model_operadores.predict(convertirOperador(image_mnist_op_eq))[0]

    # Procesar number_3
    image_mnist_number, _ = transform_image_to_mnist(number_3.image_data)
    number3 = svm_numeros.predict(convertirImagen(image_mnist_number))[0]

    # Procesar exponent_1
    image_mnist_exp, _ = transform_image_to_mnist(exponent_1.image_data)
    exponent1 = svm_numeros.predict(convertirImagen(image_mnist_exp))[0]

    # Procesar exponent_2
    image_mnist_exp, _ = transform_image_to_mnist(exponent_2.image_data)
    exponent2 = svm_numeros.predict(convertirImagen(image_mnist_exp))[0]

    # Procesar exponent_3
    image_mnist_exp, _ = transform_image_to_mnist(exponent_3.image_data)
    exponent3 = svm_numeros.predict(convertirImagen(image_mnist_exp))[0]
    
    diccionario_operadores = {
        0: "+",
        1: "-",
        2: "/",
        3: "/",  
        4: "*",
        5: "*"   
    }

    operador1 = diccionario_operadores.get(operador1, "?")
    operador2 = diccionario_operadores.get(operador2, "?")
    
    #Crear la expresión matemática
    operacion = f"{number1}^{exponent1} {operador1} {number2}^{exponent2} {operador2} {number3}^{exponent3}"
    operacion_eval = f"{number1}**{exponent1} {operador1} {number2}**{exponent2} {operador2} {number3}**{exponent3}"
    resultado = eval(operacion_eval)

    st.markdown("""
        <style>
            .math-expression {
                font-family: 'Computer Modern', serif;
                font-size: 32px;
                color: #1f1f1f;
                text-align: center;
                padding: 20px;
                background-color: #f8f9fa;
                border-radius: 10px;
                margin: 20px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
        </style>
    """, unsafe_allow_html=True)

    # Mostrar la expresión con el estilo personalizado
    st.markdown(f'<div class="math-expression">{operacion} = {resultado}</div>', unsafe_allow_html=True)
    
  
def main():
    user_panel()

if __name__ == "__main__":
    main()