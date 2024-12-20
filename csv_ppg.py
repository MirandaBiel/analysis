import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import math
import cv2
import rPPG_Methods as rppg
import process_functions as pf
import os, csv

def csv_generator(bvp_patch, signal_filtered, time_array, spectrum, freqs, method_label, patch_id):
    """
    Gera arquivos CSV para sinal bruto, sinal filtrado e análise espectral.
    
    Parâmetros:
    - bvp_patch: Sinal bruto do método rPPG.
    - signal_filtered: Sinal filtrado com filtro Butterworth.
    - time_array: Array de tempo correspondente aos sinais.
    - spectrum: Espectro de frequência do sinal.
    - freqs: Frequências correspondentes ao espectro.
    - method_label: Nome do método rPPG (definirá a pasta onde os arquivos serão salvos).
    - patch_id: Identificador do patch (número do landmark).
    """
    # Diretório base onde os arquivos serão salvos
    base_dir = "csv_outputs"
    
    # Cria a estrutura de diretórios: csv_outputs/{method_label}/patch_{patch_id}
    patch_dir = os.path.join(base_dir, method_label, f"patch_{patch_id}")
    os.makedirs(patch_dir, exist_ok=True)
    
    # 1. Arquivo CSV do sinal bruto
    raw_file_path = os.path.join(patch_dir, f"sinal_bruto_patch_{patch_id}.csv")
    with open(raw_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time (s)", "Raw Signal"])  # Cabeçalhos
        for t, value in zip(time_array, bvp_patch):
            writer.writerow([t, value])
    
    print(f"Arquivo CSV gerado: {raw_file_path}")
    
    # 2. Arquivo CSV do sinal filtrado
    filtered_file_path = os.path.join(patch_dir, f"sinal_filtrado_patch_{patch_id}.csv")
    with open(filtered_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time (s)", "Filtered Signal"])  # Cabeçalhos
        for t, value in zip(time_array, signal_filtered):
            writer.writerow([t, value])
    
    print(f"Arquivo CSV gerado: {filtered_file_path}")
    
    # 3. Arquivo CSV da análise espectral
    spectrum_file_path = os.path.join(patch_dir, f"espectro_frequencia_patch_{patch_id}.csv")
    with open(spectrum_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Frequency (Hz)", "Amplitude"])  # Cabeçalhos
        for f, value in zip(freqs, spectrum):
            writer.writerow([f, value])
    
    print(f"Arquivo CSV gerado: {spectrum_file_path}")

# Parâmetros
patches = [151, 101, 330, 10, 104, 107, 108, 109, 135, 18, 188, 199, 266, 280, 299, 333, 336, 337, 338, 347, 36, 364, 4, 50, 6, 69, 9]  # Regiões de interesse (números de landmarks)
face_mesh = mp.solutions.face_mesh.FaceMesh(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5, 
    max_num_faces=1
)
fs = 30  # Frequência de amostragem (Hz)

# Função para plotar os sinais BVP extraídos
def plot_bvp_signals_separately(bvp_signals, labels, fs):
    """
    Plota os sinais BVP em janelas separadas para cada método.
    
    Parâmetros:
    - bvp_signals: lista de arrays numpy, cada um com formato [n_patches, num_frames]
    - labels: lista de strings com os nomes dos métodos
    - fs: frequência de amostragem (Hz)
    """
    for bvp, label in zip(bvp_signals, labels):
        n_patches, num_frames = bvp.shape
        time = np.linspace(0, num_frames / fs, num_frames)

        for patch_idx in range(n_patches):
            plt.figure(figsize=(10, 6))
            plt.plot(time, bvp[patch_idx, :], label=f'Patch {patch_idx+1}')
            plt.title(f'Sinal BVP - Método: {label} (Patch {patch_idx+1})')
            plt.xlabel('Tempo (s)')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.grid(True)
            plt.show()

def processa_um_frame(frame):
    """Processa um único frame para extrair as médias RGB dos patches de interesse."""
    results = face_mesh.process(frame)
    patch_colors = []
    
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        
        # Converte coordenadas normalizadas para pixels
        landmarks_points = [
            (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
            for landmark in face_landmarks.landmark
        ]
        
        # Calcula o tamanho do patch
        l = int((math.sqrt((landmarks_points[337][0] - landmarks_points[108][0])**2 +
                           (landmarks_points[337][1] - landmarks_points[108][1])**2)) / 5)
        
        # Extrai os patches para análise
        for patch in patches:
            y_min = max(0, landmarks_points[patch][1] - l)
            y_max = min(frame.shape[0], landmarks_points[patch][1] + l)
            x_min = max(0, landmarks_points[patch][0] - l)
            x_max = min(frame.shape[1], landmarks_points[patch][0] + l)
            
            if y_max > y_min and x_max > x_min:
                crop_patch = frame[y_min:y_max, x_min:x_max]
                mean_red = np.mean(crop_patch[:, :, 0])  # Canal Vermelho
                mean_green = np.mean(crop_patch[:, :, 1])  # Canal Verde
                mean_blue = np.mean(crop_patch[:, :, 2])  # Canal Azul
                patch_colors.append([mean_red, mean_green, mean_blue])
    
    # Se não foram encontrados rostos, retorna zeros
    if not patch_colors:
        patch_colors = [[0, 0, 0] for _ in patches]
    
    return np.array(patch_colors)  # Retorna um array [num_patches, 3]

def processa_um_frame_ssr(frame, patch_id=151, target_size=(32, 32)):
    """
    Processa um único frame para extrair a região de interesse (patch 151) em formato adequado para a função SSR.
    Sempre retorna um ndarray com shape [target_size[0], target_size[1], 3].
    """
    # Processa o frame para obter landmarks
    results = face_mesh.process(frame)
    
    # Inicializa o patch com zeros caso não seja detectado rosto
    patch_crop = np.zeros((target_size[0], target_size[1], 3), dtype=np.float32)
    
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        
        # Converte coordenadas normalizadas para pixels
        landmarks_points = [
            (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
            for landmark in face_landmarks.landmark
        ]
        
        # Calcula o tamanho do patch (usando distância entre landmarks como referência)
        l = int((math.sqrt((landmarks_points[337][0] - landmarks_points[108][0]) ** 2 +
                           (landmarks_points[337][1] - landmarks_points[108][1]) ** 2)) / 5)
        
        # Extrai o patch 151 para análise
        y_min = max(0, landmarks_points[patch_id][1] - l)
        y_max = min(frame.shape[0], landmarks_points[patch_id][1] + l)
        x_min = max(0, landmarks_points[patch_id][0] - l)
        x_max = min(frame.shape[1], landmarks_points[patch_id][0] + l)
        
        if y_max > y_min and x_max > x_min:
            patch_crop_raw = frame[y_min:y_max, x_min:x_max]
            
            # Redimensiona para o tamanho fixo `target_size` (independente do tamanho original)
            patch_crop = cv2.resize(patch_crop_raw, target_size, interpolation=cv2.INTER_AREA)
    
    return patch_crop.astype(np.float32)  # Retorna um array [32, 32, 3]

def plot_rppg_signal(rppg_data, fs):
    """
    Plota o sinal RPPG extraído ao longo do tempo para cada patch e cada canal (R, G, B).
    
    Parâmetros:
    - rppg_data: numpy array com formato [n_patches, 3, n_frames]
    - fs: frequência de amostragem (Hz)
    """
    n_patches, _, num_frames = rppg_data.shape
    time = np.linspace(0, num_frames / fs, num_frames)
    
    for patch_idx in range(n_patches):
        plt.figure(figsize=(10, 6))
        plt.plot(time, rppg_data[patch_idx, 0, :], label=f'Red Channel (Patch {patch_idx+1})', color='red')
        plt.plot(time, rppg_data[patch_idx, 1, :], label=f'Green Channel (Patch {patch_idx+1})', color='green')
        plt.plot(time, rppg_data[patch_idx, 2, :], label=f'Blue Channel (Patch {patch_idx+1})', color='blue')
        
        plt.title(f"RPPG Signal Over Time - Patch {patch_idx+1}")
        plt.xlabel("Time (s)")
        plt.ylabel("Intensity")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    # Caminho correto do vídeo
    caminho_video = "video_face_7.h264"

    # Lista para armazenar os valores RGB
    rppg_channels = []
    rppg_channels_ssr = []

    # Abre o vídeo
    captura = cv2.VideoCapture(caminho_video)

    if not captura.isOpened():
        print("Erro ao abrir o vídeo.")
    else:
        while True:
            ret, frame = captura.read()
            if not ret:
                print("Fim do vídeo ou erro ao ler frame.")
                break
            
            # Processa o frame e armazena o resultado
            rgb_values = processa_um_frame(frame)  # Agora retorna [num_patches, 3]
            rppg_channels.append(rgb_values)

            # Processa o frame e extrai o patch 151 com tamanho fixo
            patch_crop = processa_um_frame_ssr(frame, target_size=(32, 32))
            rppg_channels_ssr.append(patch_crop)
            
            # Exibe o frame
            cv2.imshow('Frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    # Libera o objeto de captura e fecha a janela
    captura.release()
    cv2.destroyAllWindows()

    # Converte a lista para um ndarray com shape [num_patches, 3, num_frames]
    rppg_channels = np.array(rppg_channels, dtype=np.float32)
    rppg_channels = rppg_channels.transpose(1, 2, 0)

    # Converte a lista para um ndarray com o formato necessário [num_frames, rows, columns, rgb_channels]
    rppg_channels_ssr = np.array(rppg_channels_ssr, dtype=np.float32)

    # Mostra o gráfico das capturas no tempo
    #plot_rppg_signal(rppg_channels, fs)

    # Aplicar métodos rPPG
    bvp_chrom = rppg.CHROM(rppg_channels)
    bvp_green = rppg.GREEN(rppg_channels)
    bvp_lgi = rppg.LGI(rppg_channels)
    bvp_pos = rppg.POS(rppg_channels, fps=fs)
    bvp_gbgr = rppg.GBGR(rppg_channels)
    bvp_ica = rppg.ICA(rppg_channels, component='second_comp')
    bvp_omit = rppg.OMIT(rppg_channels)
    bvp_pbv = rppg.PBV(rppg_channels)
    bvp_pca = rppg.PCA(rppg_channels, component='second_comp')
    bvp_ssr = rppg.SSR(rppg_channels_ssr, fps=fs)

    # Lista de sinais e seus rótulos
    bvp_signals = [bvp_chrom, bvp_green, bvp_lgi, bvp_pos, bvp_gbgr, bvp_ica, bvp_omit, bvp_pbv, bvp_pca, bvp_ssr]
    labels = ['CHROM', 'GREEN', 'LGI', 'POS', 'GBGR', 'ICA', 'OMIT', 'PBV', 'PCA', 'SSR']

    # Analisa cada um do métodos
    for j, bvp_patches in enumerate(bvp_signals):
        print(f'Shape: {bvp_patches.shape}')

        # Para cada método, analisa cada um dos patches
        for i, bvp_patch in enumerate(bvp_patches):

            # Aplicar o filtro Butterworth
            signal_filtered = pf.filter_z_butterworth(bvp_patch, fs)
            time_array = np.linspace(0, len(signal_filtered) / fs, len(signal_filtered))
            spectrum, freqs = pf.calculate_fft(signal_filtered, fs)

            # Gera 3 arquivos csv, uma para o sinal bruto, outro para o filtrado e outro para a análise espectral
            csv_generator(bvp_patch, signal_filtered, time_array, spectrum, freqs, labels[j], patches[i])