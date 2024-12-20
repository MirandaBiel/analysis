from scipy.signal import butter, filtfilt, find_peaks
import numpy as np

# Apenas normaliza o sinal usando Z-score
def filter_z(signal):
    # Z-Score Normalizacao 
    x = np.mean(signal)
    dp = np.std(signal)
    norm_signal = (signal - x) / dp

    return norm_signal

# Apenas filtro de butterworth
def filter_butterworth(signal, fs, order=6):
    # Aplica o filtro de passa banda
    nyquist = 0.5 * fs
    f_low = 0.6
    f_high = 3.6

    # Calcular as frequências normalizadas
    wn_low =  f_low / nyquist
    wn_high = f_high / nyquist

    # Calcular os coeficientes do filtro Butterworth
    b, a = butter(order, [wn_low, wn_high], btype='band')

    # Aplicar o filtro passa-faixa
    y = filtfilt(b, a, signal)

    return y

# Normaliza Z-Score e aplica o filtro de ButterWorth
def filter_z_butterworth(signal, fs):
    
    y = filter_butterworth(filter_z(signal), fs)

    return y

# Alinha a onda no eixo x e aplica o filtro de Butterworth
def filter_align_butterworth(signal, fs):
    x = np.mean(signal)
    signal = (signal - x)

    y = filter_butterworth(signal, fs)

    return y

# Função que aprimora os picos do sinal
def peak_enhancement(sig):
    rb = 1024 # rb é o intervalo do limite superior
    l_lt = 0 # l_lt o intervalo dos limites inferiores
    P_aprimorado = rb*((sig-min(sig))/(max(sig) - min(sig))) + l_lt

    return P_aprimorado

# Função que aplica o filtro de hampel para remoção de pontos fora da curva
def hampel_filter(sig, window_size=6, n_sigma=3):
    hampel_filter_sig = sig.copy()
    for i in range(len(sig) - window_size + 1):
        window = sig[i:i + window_size]
        mediana = np.median(window)
        mad = np.median(np.abs(window - mediana))
        threshold = mediana + (n_sigma * mad)
        for j in range(i, i + window_size):
            if sig[j] > threshold:
                hampel_filter_sig[j] = mediana

    return hampel_filter_sig

# Passa alta detrending
def detrending_highpass_filter(signal, fs, lambda_value=300):
    # Calcula a frequência de corte normalizada com base no parâmetro lambda
    normalized_cutoff = 0.011 * lambda_value
    
    # Calcula a frequência de corte em Hz
    cutoff_freq = normalized_cutoff * fs
    
    # Calcula a matriz D de diferenças de segunda ordem
    n = len(signal)
    D = np.zeros((n - 3, n - 1))
    for i in range(n - 3):
        D[i, i:i+3] = [1, -2, 1]
    
    # Calcula a matriz I
    I = np.eye(n - 1)
    
    # Calcula a matriz A = (I + lambda^2 * D^T * D)
    A = I + (lambda_value**2) * np.dot(D.T, D)

    # Novos vetores de sinal e tempo
    n_signal = signal[:-1]
    
    # Calcula o sinal de baixa frequência
    #low_freq_signal = np.dot(n_signal, inv(A))

    # Calcula o sinal filtrado subtraindo o sinal de baixa frequência do sinal original
    filtered_signal = n_signal - low_freq_signal

    # Duplica o ultimo elemento dos sinais
    low_freq_signal = np.append(low_freq_signal, low_freq_signal[-1])
    filtered_signal = np.append(filtered_signal, filtered_signal[-1])
    
    return filtered_signal

# Transformada de Fourier
def calculate_fft(signal, fs, padding_factor=100):
    # Calcular a FFT com padding
    fft_result = np.fft.fft(signal, n=len(signal) * padding_factor)
    spectrum = np.abs(fft_result)  # Módulo do espectro
    freqs = np.fft.fftfreq(len(fft_result), 1/fs) * 60  # Converter para BPM

    return spectrum, freqs

# Estima bpm
def calc_frequencia_cardiaca(spectrum, freqs, min_freq=30, max_freq=216):
    # Encontrar os índices que correspondem à faixa de frequências desejada
    start_index = np.argmax(freqs >= min_freq)
    end_index = np.argmax(freqs >= max_freq)

    # Verificar se a faixa é válida
    if start_index >= end_index:
        return None, None

    # Encontrar o índice do maior pico dentro da faixa
    peak_index = np.argmax(spectrum[start_index:end_index]) + start_index

    # Obter a frequência e amplitude correspondentes ao maior pico
    peak_frequency = freqs[peak_index]
    peak_amplitude = spectrum[peak_index]

    return peak_frequency

def analyze_signal_spectrum(spectrum, freqs, min_bpm=30, max_bpm=200, num_peaks=20):
    # Encontrar os índices correspondentes à faixa de frequências entre min_bpm e max_bpm
    start_index = np.argmax(freqs >= min_bpm)
    end_index = np.argmax(freqs >= max_bpm)

    # Encontrar picos no espectro dentro da faixa desejada
    peaks, _ = find_peaks(spectrum[start_index:end_index])
    peaks += start_index  # Ajustar os índices dos picos para a faixa total

    # Ordenar os picos pela potência
    sorted_peaks = peaks[np.argsort(spectrum[peaks])[::-1]]

    # Selecionar os picos mais proeminentes
    if len(sorted_peaks) >= num_peaks:
        top_peaks = sorted_peaks[:num_peaks]
    else:
        top_peaks = sorted_peaks

    # Obter as frequências e amplitudes correspondentes aos picos
    peak_frequencies = freqs[top_peaks]
    peak_amplitudes = spectrum[top_peaks]

    # **Cálculo do índice de qualidade**
    if len(peak_amplitudes) > 1:
        # Normalizar as amplitudes dividindo pelo maior valor
        normalized_amplitudes = peak_amplitudes / np.max(peak_amplitudes)

        # Calcular o índice de qualidade: maior valor menos a média dos outros valores
        max_value = np.max(normalized_amplitudes)
        mean_others = np.mean(normalized_amplitudes[normalized_amplitudes != max_value])
        quality_index = max_value - mean_others
    else:
        # Caso haja apenas um pico, o índice de qualidade é zero
        quality_index = 0

    return quality_index

# Função que calcula a frequencia respiratoria
def calc_frequencia_respiratoria(signal, fs):
    # Pre-processamento do sinal
    nyquist = 0.5 * fs
    f_low = 0.1
    f_high = 0.4
    # Calcular as frequências normalizadas
    wn_low = f_low / nyquist
    wn_high = f_high / nyquist
    # Calcular os coeficientes do filtro Butterworth
    b, a = butter(2, [wn_low, wn_high], btype='band')
    # Aplicar o filtro passa-faixa
    y = filtfilt(b, a, signal)

    signal_pe = peak_enhancement(y)
    signal_hf = hampel_filter(signal_pe, fs*6, n_sigma=3)
    
    Median_sig = []
    tamanho_v_deslizante = fs
    for i in range(len(signal_hf) - tamanho_v_deslizante + 1):
        v_deslizante = signal_hf[i:i+tamanho_v_deslizante]
        Median = np.median(v_deslizante)
        Median_sig.append(Median)
    Median_sig = np.array(Median_sig)

    # 2: Aplicação da FFT no vetor resultante e seleção do pico de frequência dominante dentro da faixa RR válida - entre 4 rpm (0,06 Hz) e 60 rpm (1 Hz)
    padding_factor = 0
    FFT_sig = np.fft.fft(Median_sig, n=len(Median_sig)*(padding_factor+1))
    f = np.fft.fftfreq(len(FFT_sig), 1/fs)
    spectrum = np.abs(FFT_sig)  # Calcular o modulo espectral

    # Encontrar os indices correspondentes as frequencias entre 0.1 e 0.4 bpm
    indice_inicio = np.argmax(f >= f_low)
    indice_fim = np.argmax(f >= f_high)

    # Encontrar os picos no espectro dentro da faixa desejada
    peaks, _ = find_peaks(spectrum[indice_inicio:indice_fim])
    peaks += indice_inicio  # Ajustar os indices dos picos para a faixa total

    # Ordenar os picos pela potencia
    sorted_peaks = peaks[np.argsort(spectrum[peaks])[::-1]]

    # Selecionar os tres maiores picos
    if len(sorted_peaks) >= 20:
        top_peaks = sorted_peaks[:20]
    else:
        top_peaks = sorted_peaks

    # Obter as frequencias correspondentes aos maiores picos
    frequencias_picos = f[top_peaks]
    amplitudes_picos = spectrum[top_peaks]
    if len(frequencias_picos) != 0:
        fr = frequencias_picos[0]
    else:
        fr = 0

    # 3: Multiplicando a frequencia dominante por 60 para converter para "rpm"
    return fr * 60