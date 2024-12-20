def extract_raw_signal(self):
        # Zera as listas dos sinais
        self.rppg_channels = []
        self.rppg_channels_ssr = []

        self.ui.label_comunicacao_2.setText('Realizando processamento...')
        progress_value = 0
        self.ui.progressBar_2.setValue(progress_value)
        n_frames = self.frames_capturados
        processed_frames = 0

        captura = cv2.VideoCapture(self.video_path)
        print(f'ABRINDO: {self.video_path}')
        if not captura.isOpened():
            print("Erro ao abrir o vídeo.")
        else:
            while True:
                ret, frame = captura.read()
                if not ret:
                    print("Fim do vídeo ou erro ao ler frame.")
                    break
                
                # Processa o frame e armazena o resultado
                rgb_values = processa_um_frame(frame, self.landmarks)  # Agora retorna [num_patches, 3]
                self.rppg_channels.append(rgb_values)

                # Processa o frame e extrai o patch 151 com tamanho fixo
                patch_crop = processa_um_frame_ssr(frame, target_size=(32, 32))
                self.rppg_channels_ssr.append(patch_crop)
                
                processed_frames = processed_frames + 1
                progress_value = int((processed_frames / n_frames) * 100)
                self.ui.progressBar_2.setValue(progress_value)

            self.ui.progressBar_2.setValue(100)
            self.ui.label_comunicacao_2.setText('Análise concluida...')
            print(f'Frames processados: {processed_frames}')

        # Converte a lista para um ndarray com shape [num_patches, 3, num_frames]
        self.rppg_channels = np.array(self.rppg_channels, dtype=np.float32)
        print(self.rppg_channels.shape)
        self.rppg_channels = self.rppg_channels.transpose(1, 2, 0)

        # Converte a lista para um ndarray com o formato necessário [num_frames, rows, columns, rgb_channels]
        self.rppg_channels_ssr = np.array(self.rppg_channels_ssr, dtype=np.float32)

        # Mostra o gráfico da captura no tempo
        plot_rppg_signal(self.rppg_channels, self.fps, self.n_video)

    def process_raw_signal_mediana(self):
        iq_patches = []
        bpm = []
        irpm = []

        if self.method == 'CHROM':
            bvp_patches = rppg.CHROM(self.rppg_channels)
        elif self.method == 'GREEN':
            bvp_patches = rppg.GREEN(self.rppg_channels)
        elif self.method == 'LGI':
            bvp_patches = rppg.LGI(self.rppg_channels)
        elif self.method == 'POS':
            bvp_patches = rppg.POS(self.rppg_channels, fps=self.fps)
        elif self.method == 'GBGR':
            bvp_patches = rppg.GBGR(self.rppg_channels)
        elif self.method == 'ICA':
            bvp_patches = rppg.ICA(self.rppg_channels, component='second_comp')
        elif self.method == 'OMIT':
            bvp_patches = rppg.OMIT(self.rppg_channels)
        elif self.method == 'PBV':
            bvp_patches = rppg.PBV(self.rppg_channels)
        elif self.method == 'PCA':
            bvp_patches = rppg.PCA(self.rppg_channels, component='second_comp')
        elif self.method == 'SSR':
            bvp_patches = rppg.GREEN(self.rppg_channels)
            #bvp = rppg.SSR(self.rppg_channels, fps=self.fps)
        else:
            bvp_patches = rppg.GREEN(self.rppg_channels)

        for i, bvp_patch in enumerate(bvp_patches):
            
            # Aplicar o filtro Butterworth
            signal_filtered = pf.filter_z_butterworth(bvp_patch, self.fps)
            self.ppg.append(signal_filtered)
            time_array = np.linspace(0, len(signal_filtered) / self.fps, len(signal_filtered))

            # Gráfico do sinal original (antes de aplicar o filtro)
            graph_generic_signal(
                bvp_patch, 
                'bvp_patch', 
                time_array, 
                'Tempo', 
                'Sinal Original', 
                f'{self.n_video}_BVP_signal_{i}.png',  # Nome do arquivo com o índice
                ind_min=None, 
                ind_max=None
            )
            
            # Gráfico do sinal filtrado
            graph_generic_signal(
                signal_filtered, 
                'Amplitude', 
                time_array, 
                'Tempo', 
                'Sinal Filtrado', 
                f'{self.n_video}_BVP_filtered_{i}.png',  # Nome do arquivo com o índice
                ind_min=None, 
                ind_max=None
            )
            
            # Calcular a FFT
            spectrum, freqs = pf.calculate_fft(signal_filtered, self.fps)
            
            # Gráfico da análise espectral
            graph_generic_signal(
                spectrum, 
                'Amplitude', 
                freqs, 
                'Frequência (bpm)', 
                'Análise Espectral', 
                f'{self.n_video}BVP_spectrum_{i}.png',  # Nome do arquivo com o índice
                ind_min=20, 
                ind_max=200
            )
            
            # Calcular BPM e IMRP
            bpm_ = pf.calc_frequencia_cardiaca(spectrum, freqs)
            bpm.append(bpm_)
            iq_patches.append(pf.analyze_signal_spectrum(spectrum, freqs, min_bpm=30, max_bpm=200, num_peaks=20))
            print(f"BPM {i}: {bpm_}")
            irpm_ = pf.calc_frequencia_respiratoria(bvp_patch, self.fps)
            irpm.append(irpm_)
            print(f"irpm {i}: {irpm_}")
            
        max_index = np.argmax(iq_patches)
        self.bpm = bpm[max_index]
        self.irpm = irpm[max_index]
        self.best_patch = max_index
        self.iq1 = iq_patches[max_index]
        self.iq2 = sq.Kurtosis(self.ppg[max_index])
        #self.iq3 = sq.SNR(self.ppg[max_index])
        self.iq3 = self.iq2