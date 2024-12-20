import matplotlib.pyplot as plt
import csv
import os

def process_and_save_csv_graphs(input_dir, output_dir):
    """
    Lê arquivos CSV em uma estrutura de pastas, gera gráficos e os salva com uma organização similar.
    
    Parâmetros:
    - input_dir: Diretório raiz onde os arquivos CSV estão armazenados.
    - output_dir: Diretório raiz onde os gráficos serão salvos.
    """
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".csv"):
                csv_path = os.path.join(root, file)
                relative_path = os.path.relpath(csv_path, input_dir)  # Caminho relativo do arquivo
                save_dir = os.path.join(output_dir, os.path.dirname(relative_path))  # Espelha a estrutura
                os.makedirs(save_dir, exist_ok=True)  # Cria a pasta, se necessário

                # Lê e identifica o arquivo CSV
                with open(csv_path, mode='r') as csv_file:
                    reader = csv.reader(csv_file)
                    headers = next(reader)  # Lê o cabeçalho
                    
                    # Identifica o tipo de gráfico
                    if headers == ["Time (s)", "Raw Signal"]:
                        graph_type = "Sinal Bruto"
                        x_label = "Tempo (s)"
                        y_label = "Amplitude"
                    elif headers == ["Time (s)", "Filtered Signal"]:
                        graph_type = "Sinal Filtrado"
                        x_label = "Tempo (s)"
                        y_label = "Amplitude"
                    elif headers == ["Frequency (Hz)", "Amplitude"]:
                        graph_type = "Espectro de Frequência"
                        x_label = "Frequência (Hz)"
                        y_label = "Amplitude"
                    else:
                        print(f"Arquivo ignorado (formato desconhecido): {csv_path}")
                        continue

                # Lê os dados do arquivo CSV
                x_values = []
                y_values = []
                with open(csv_path, mode='r') as csv_file:
                    reader = csv.reader(csv_file)
                    next(reader)  # Pula o cabeçalho
                    for row in reader:
                        try:
                            x_values.append(float(row[0]))
                            y_values.append(float(row[1]))
                        except ValueError:
                            print(f"Erro ao processar linha: {row}")
                            continue

                # Plota e salva o gráfico
                plt.figure(figsize=(10, 6))
                plt.plot(x_values, y_values, color='b', label=graph_type)
                plt.title(f"Gráfico: {graph_type}")
                plt.xlabel(x_label)
                plt.ylabel(y_label)
                plt.legend()
                plt.grid(True)

                # Salva o gráfico como PNG
                graph_name = os.path.splitext(file)[0] + ".png"  # Nome do gráfico
                save_path = os.path.join(save_dir, graph_name)
                plt.savefig(save_path)
                print(f"Gráfico salvo: {save_path}")
                plt.close()

if __name__ == "__main__":
    # Diretórios de entrada e saída
    input_dir = input("Digite o diretório raiz dos arquivos CSV: ").strip()
    output_dir = input("Digite o diretório onde os gráficos serão salvos: ").strip()

    # Processa os arquivos e salva os gráficos
    process_and_save_csv_graphs(input_dir, output_dir)
