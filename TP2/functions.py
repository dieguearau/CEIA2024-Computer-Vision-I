# ---------------- Paquetes a utilizar ----------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


import cv2 as cv

# ------------------------------------------------------------------
# ---------------- Algoritmos detectores de enfoque ----------------
# ------------------------------------------------------------------


# ---------------- Algoritmo para medir Calidad de Imagen ----------------

def image_quality_measure(img):

    '''
    Input:      Imagen de M x N
    Output:     Medida de la calidad de la imagen (FM)
                FM = medida del desenfoque de la imagen en el dominio de la frecuencia

    '''

    # 1) Tranformada de Fourier
    TF = np.fft.fft2(img)

    # 2) Se lleva la baja frecuencia al origen 
    Fc = np.fft.fftshift(TF)
    
    # 3) Se calcula AF = abs(Fc)
    AF = np.abs(Fc)
    
    # 4) Se calcula M = max(AF)
    M = np.max(AF)
    
    # 5) Se calcula TH = número total de píxeles en F donde valor de píxel > M/1000
    thres = M / 1000
    TH = np.sum(AF > thres)
    
    # 6) Medida de la calidad de la imagen (FM)
    FM = TH / (img.shape[0] * img.shape[1])

    return FM

# ---------------- Algoritmo para medir Calidad de Imagen ----------------

def focus_ACMO(image):
    
    #image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Histograma
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256], density=True)
    
    # Valor de intensidad medio 
    intensity_values = np.arange(256)
    mean_intensity = np.sum(intensity_values * hist)
    
    # Cálculo ACMo (Momento Central Absoluto)
    ACMO = np.sum(np.abs(intensity_values - mean_intensity) * hist)
    
    return ACMO










# ---------------------------------------------------------------------------------
# ---------------- Algoritmos para visualización de métricas ----------------------
# ---------------------------------------------------------------------------------

# ---------------- Algoritmo para procesar video y detectar zonas de enfoque ----------------

def enfoque_video(FM, perc, plot = False, title_plot = None, focus_color = 'limegreen', desenfoc_color = 'tomato'):
    """
    Inputs:
    array_enfoque:      Array que contiene las medidas de enfoque para cada frame.
    percentil:          Percentil que se utilizará para definir la zona de enfoque.

    Outputs:
    gráfico
    frames_enfocados:   Dataframe de los frames que cumplen con el criterio de enfoque.
    """

    # Se convierte a DataFrame para tener pares de frame / medida de enfoque
    exp = pd.DataFrame({'frame': range(len(FM)), 'FM': FM})

    # Métrica objetivo para definir zona de enfoque
    metric = np.percentile(exp['FM'], q=perc)
    
    # Punto máximo
    max_metric = np.max(exp['FM'])
    max_frame = exp[exp['FM'] == max_metric]['frame']

    # Se guardan los índices que están enfocados para detectar en video después
    frames_enfocados = np.where(exp['FM'] >= metric, exp['frame'], None)
    frames_enfocados = frames_enfocados[frames_enfocados != None]

    if plot == True:
            # Se dibuja el gráfico para detectar zonas de enfoque
            plt.figure(figsize=(15, 5))

            # Línea horizontal para mostrar el límite de la métrica
            plt.axhline(y=metric, color='forestgreen', linestyle='dotted', label=f'Valor métrica perc. {perc}')

            # Gráfico de enfoque por frame (verde para enfocado, rojo para desenfocado)
            plt.scatter(exp['frame'][exp['FM'] >= metric], exp['FM'][exp['FM'] >= metric], 
                        color=focus_color, label='Zona enfocada')
            plt.scatter(exp['frame'][exp['FM'] < metric], exp['FM'][exp['FM'] < metric], 
                        color=desenfoc_color, label='Zona desenfocada')
            
            # Punto máximo
            plt.scatter(max_frame, max_metric, 
                        color='navy', label='Punto de enfoque máximo')
            
            # Texto con umbral
            plt.text(0, metric*1.02, f'Umbral enfoque: {np.round(metric,4)}', fontsize = 10, fontweight='bold',
                     color = focus_color)

            # Título y etiquetas
            plt.title(title_plot)
            plt.xlabel('Frame')
            plt.ylabel('Medida de enfoque')
            plt.legend()
            plt.grid(color='lightgray', alpha=0.5)
            plt.show()

    return metric, max_metric, max_frame, exp.loc[frames_enfocados,:]

# ---------------- Función para dibujar la grilla en ROI ----------------

def draw_spaced_grid(frame, num_rows, num_cols, square_size=20, spacing=15, color=(0, 0, 255), thickness=2):
    
    h, w, _ = frame.shape
    
    grid_height = num_rows * (square_size + spacing)
    grid_width = num_cols * (square_size + spacing)
    
    
    start_x = (w - grid_width) // 2
    start_y = (h - grid_height) // 2
    
    for i in range(num_rows):
        for j in range(num_cols):
            
            top_left = (start_x + j * (square_size + spacing), start_y + i * (square_size + spacing))
            bottom_right = (top_left[0] + square_size, top_left[1] + square_size)
            cv.rectangle(frame, top_left, bottom_right, color, thickness)
    
    return frame










# ---------------------------------------------------------------------------------
# ---------------- Algoritmos para procesar videos --------------------------------
# ---------------------------------------------------------------------------------

# ---------------- Función para procesar el video FULL ----------------

def process_video_FULL(video_path):
    cap = cv.VideoCapture(video_path)
    focus_measure = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # Experimento 1
        fm_full = image_quality_measure(gray_frame)

        focus_measure.append(fm_full)
        

    cap.release()
    return(focus_measure)

# ---------------- Función para procesar el video ROI ----------------

def process_video_ROI(video_path, size = 0.1):
    cap = cv.VideoCapture(video_path)
    focus_measure = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        height, width = gray_frame.shape

        # Punto medio para ubicar ROI
        med_h = height // 2
        med_w = width // 2

        # ROI
        area_ROI_h = int(size * height)  # Area de ROI = % size del total
        area_ROI_w = int(size * width)

        min_height = med_h - (area_ROI_h // 2)
        max_height = med_h + (area_ROI_h // 2)
        min_width =  med_w - (area_ROI_w // 2)
        max_width =  med_w + (area_ROI_w // 2)

        gray_frame = gray_frame[min_width:max_width, min_height:max_height]
        
        fm_ROI = image_quality_measure(gray_frame)

        focus_measure.append(fm_ROI)
        

    cap.release()
    return(focus_measure)


# ---------------- Función para procesar el video gridROI ----------------

def process_video_gridROI(video_path, N, M):
    cap = cv.VideoCapture(video_path)
    focus_measure = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Se convierte a escala de grises
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        height, width = gray_frame.shape

        # Tamaño de cada rectángulo en la matriz NxM
        rect_height = height // N
        rect_width = width // M

        # Lista para almacenar las medidas de enfoque de cada ROI
        fm_ROIs = []

        # Se recorren cada uno de los NxM rectángulos de la grilla
        for i in range(N):
            for j in range(M):
                # Definir los límites del ROI
                min_h = i * rect_height
                max_h = (i + 1) * rect_height
                min_w = j * rect_width
                max_w = (j + 1) * rect_width

                # se extrae el ROI correspondiente a cada rectángulo
                roi = gray_frame[min_h:max_h, min_w:max_w]
                
                # Se calcula la medida de enfoque de este ROI
                fm_ROI = image_quality_measure(roi)
                fm_ROIs.append(fm_ROI)
        
        # Se agrega el promedio de las medidas de enfoque de este frame
        focus_measure.append(np.mean(fm_ROIs))

    cap.release()
    return focus_measure


# ---------------- Función de unsharp masking K = 5 ----------------

def unsharp_masking1(image, kernel_size=(5, 5), sigma=1, amount=2, threshold=0):
    
    blurred = cv.GaussianBlur(image, kernel_size, sigma) #;print(blurred)

    sharpened = float(amount + 1) * image - float(amount) * blurred

    # Para llevar los valores entre rangos de 0-255
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))

    sharpened = sharpened.round().astype(np.uint8)
    
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)

    return sharpened

# ---------------- Función de unsharp masking K = 25 ----------------

def unsharp_masking2(image, kernel_size=(25, 25), sigma=5, amount=2, threshold=0):
    
    blurred = cv.GaussianBlur(image, kernel_size, sigma) #;print(blurred)

    sharpened = float(amount + 1) * image - float(amount) * blurred

    # Para llevar los valores entre rangos de 0-255
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))

    sharpened = sharpened.round().astype(np.uint8)
    
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)

    return sharpened



# ---------------- Función para procesar con unsharp masking K = 5 ----------------

def process_video_UM1(video_path, focus_threshold=100):
    cap = cv.VideoCapture(video_path)
    focus_measure = []  # Lista para guardar las métricas de enfoque

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # Métrica de calidad en el frame completo
        fm_full = image_quality_measure(gray_frame)

        # Si la métrica de enfoque es menor que el umbral, se aplica Unsharp Masking
        if fm_full < focus_threshold:
            gray_frame = unsharp_masking1(gray_frame)
            fm_full = image_quality_measure(gray_frame)  # Se recalcula la métrica sobre el frame procesado
        
        # Guardar el valor de la métrica (fm_full)
        focus_measure.append(fm_full)
    
    cap.release()
    
    return focus_measure

# ---------------- Función para procesar con unsharp masking K = 25 ----------------

def process_video_UM2(video_path, focus_threshold=100):
    cap = cv.VideoCapture(video_path)
    focus_measure = []  # Lista para guardar las métricas de enfoque

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # Métrica de calidad en el frame completo
        fm_full = image_quality_measure(gray_frame)

        # Si la métrica de enfoque es menor que el umbral, se aplica Unsharp Masking
        if fm_full < focus_threshold:
            gray_frame = unsharp_masking2(gray_frame)
            fm_full = image_quality_measure(gray_frame)  # Se recalcula la métrica sobre el frame procesado
        
        # Guardar el valor de la métrica (fm_full)
        focus_measure.append(fm_full)
    
    cap.release()
    
    return focus_measure

# ---------------- Función para procesar el video FULL ----------------

def process_video_FULL_ACMO(video_path):
    cap = cv.VideoCapture(video_path)
    focus_measure = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # Experimento 1
        fm_full = focus_ACMO(gray_frame)

        focus_measure.append(fm_full)
        

    cap.release()
    return(focus_measure)

# ---------------- Función para procesar el video ROI con ACMO ----------------

def process_video_ROI_ACMO(video_path, size = 0.1):
    cap = cv.VideoCapture(video_path)
    focus_measure = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        height, width = gray_frame.shape

        # Punto medio para ubicar ROI
        med_h = height // 2
        med_w = width // 2

        # ROI
        area_ROI_h = int(size * height)  # Area de ROI = % size del total
        area_ROI_w = int(size * width)

        min_height = med_h - (area_ROI_h // 2)
        max_height = med_h + (area_ROI_h // 2)
        min_width =  med_w - (area_ROI_w // 2)
        max_width =  med_w + (area_ROI_w // 2)

        gray_frame = gray_frame[min_width:max_width, min_height:max_height]
        
        fm_ROI = focus_ACMO(gray_frame)

        focus_measure.append(fm_ROI)
        

    cap.release()
    return(focus_measure)

# ---------------- Función para procesar el video gridROI con ACMO ----------------

def process_video_gridROI_ACMO(video_path, N, M):
    cap = cv.VideoCapture(video_path)
    focus_measure = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Se convierte a escala de grises
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        height, width = gray_frame.shape

        # Tamaño de cada rectángulo en la matriz NxM
        rect_height = height // N
        rect_width = width // M

        # Lista para almacenar las medidas de enfoque de cada ROI
        fm_ROIs = []

        # Se recorren cada uno de los NxM rectángulos de la grilla
        for i in range(N):
            for j in range(M):
                # Definir los límites del ROI
                min_h = i * rect_height
                max_h = (i + 1) * rect_height
                min_w = j * rect_width
                max_w = (j + 1) * rect_width

                # se extrae el ROI correspondiente a cada rectángulo
                roi = gray_frame[min_h:max_h, min_w:max_w]
                
                # Se calcula la medida de enfoque de este ROI
                fm_ROI = focus_ACMO(roi)
                fm_ROIs.append(fm_ROI)
        
        # Se agrega el promedio de las medidas de enfoque de este frame
        focus_measure.append(np.mean(fm_ROIs))

    cap.release()
    return focus_measure