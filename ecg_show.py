import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import sys

# Função principal
def plot_ecg_with_slider(csv_file):
    try:
        # Carregar os dados do arquivo CSV
        data = pd.read_csv(csv_file)
        
        # Verificar se as colunas necessárias existem
        if not {'Timestamp', 'ECGValue'}.issubset(data.columns):
            print("O arquivo CSV deve conter as colunas: 'Timestamp' e 'ECGValue'.")
            sys.exit(1)

        # Normalizar o tempo para começar em 0
        data['Time (s)'] = (data['Timestamp'] - data['Timestamp'].iloc[0]) / 1000.0
        
        # Definir os limites do tempo
        total_time = data['Time (s)'].iloc[-1]
        window_size = 10  # Tamanho da janela de tempo em segundos

        # Inicializar a figura
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)  # Ajustar espaço para o slider
        ax.set_title("Sinal ECG com Barra Deslizante")
        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel("ECG Value")

        # Plotar os primeiros 10 segundos
        start_time = 0
        end_time = start_time + window_size
        mask = (data['Time (s)'] >= start_time) & (data['Time (s)'] <= end_time)
        line, = ax.plot(data['Time (s)'][mask], data['ECGValue'][mask], color='green')

        ax.set_xlim(start_time, end_time)
        ax.set_ylim(data['ECGValue'].min(), data['ECGValue'].max())

        # Adicionar Slider para ajustar o tempo
        ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03], facecolor='lightgray')
        slider = Slider(ax_slider, 'Tempo (s)', 0, total_time - window_size, valinit=0)

        # Atualizar a janela de visualização ao mover o slider
        def update(val):
            start_time = slider.val
            end_time = start_time + window_size
            mask = (data['Time (s)'] >= start_time) & (data['Time (s)'] <= end_time)
            line.set_xdata(data['Time (s)'][mask])
            line.set_ydata(data['ECGValue'][mask])
            ax.set_xlim(start_time, end_time)
            fig.canvas.draw_idle()

        slider.on_changed(update)

        plt.show()
    
    except FileNotFoundError:
        print(f"Arquivo '{csv_file}' não encontrado. Verifique o caminho.")
    except Exception as e:
        print(f"Ocorreu um erro: {e}")

# Entrada do usuário
if __name__ == "__main__":
    csv_file = 'ecg_data_7.csv'
    plot_ecg_with_slider(csv_file)
