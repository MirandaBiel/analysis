import matplotlib.pyplot as plt
import csv
import os

def plot_csv(file_path):
    """
    Gera um gráfico a partir de um arquivo CSV.
    
    Parâmetro:
    - file_path: Caminho do arquivo CSV a ser lido.
    """
    if not os.path.isfile(file_path):
        print(f"Erro: O arquivo '{file_path}' não existe.")
        return
    
    # Identifica o tipo de arquivo com base nos cabeçalhos
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        headers = next(reader)  # Lê o cabeçalho
        print(f"Cabecalho encontrado: {headers}")
        
        # Determina o tipo de dado com base nos cabeçalhos
        if headers == ["Time (s)", "Raw Signal"]:
            signal_type = "Sinal Bruto"
            x_label = "Tempo (s)"
            y_label = "Amplitude"
        elif headers == ["Time (s)", "Filtered Signal"]:
            signal_type = "Sinal Filtrado"
            x_label = "Tempo (s)"
            y_label = "Amplitude"
        elif headers == ["Frequency (Hz)", "Amplitude"]:
            signal_type = "Espectro de Frequência"
            x_label = "Frequência (Hz)"
            y_label = "Amplitude"
        else:
            print("Erro: Formato de cabeçalho não reconhecido.")
            return
    
    # Lê os dados do arquivo CSV
    x_values = []
    y_values = []
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Pula o cabeçalho
        for row in reader:
            try:
                x_values.append(float(row[0]))
                y_values.append(float(row[1]))
            except ValueError:
                print(f"Erro ao processar linha: {row}")
                continue
    
    # Plota o gráfico
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, color='b', label=signal_type)
    plt.title(f"Gráfico: {signal_type}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Especifica o caminho do arquivo CSV
    caminho_csv = input("Digite o caminho do arquivo CSV: ").strip()
    
    # Plota o gráfico
    plot_csv(caminho_csv)
