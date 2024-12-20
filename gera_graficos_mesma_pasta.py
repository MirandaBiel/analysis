import matplotlib.pyplot as plt
import csv
import os

def process_and_save_csv_graphs(input_dir, output_dir):
    """
    Lê arquivos CSV em uma estrutura de pastas, gera gráficos e os salva em uma única pasta.
    O nome do gráfico conterá informações sobre o método, patch e tipo do sinal.
    
    Parâmetros:
    - input_dir: Diretório raiz onde os arquivos CSV estão armazenados.
    - output_dir: Diretório onde os gráficos serão salvos.
    """
    os.makedirs(output_dir, exist_ok=True)  # Cria a pasta de saída, se necessário

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".csv"):
                csv_path = os.path.join(root, file)
                
                # Extrai informações do caminho e do nome do arquivo
                relative_path = os.path.relpath(csv_path, input_dir)  # Caminho relativo do arquivo
                path_parts = relative_path.split(os.sep)
                
                # Informações do método, patch e tipo do sinal
                method = path_parts[0] if len(path_parts) > 0 else "UnknownMethod"
                patch = path_parts[1] if len(path_parts) > 1 else "UnknownPatch"
                signal_type = os.path.splitext(file)[0]  # Nome do arquivo sem extensão
                
                # Lê o cabeçalho para identificar o tipo do gráfico
                with open(csv_path, mode='r') as csv_file:
                    reader = csv.reader(csv_file)
                    headers = next(reader)  # Lê o cabeçalho
                    
                    # Identifica o tipo de gráfico
                    if headers == ["Time (s)", "Raw Signal"]:
                        graph_label = "Sinal Bruto"
                        x_label = "Tempo (s)"
                        y_label = "Amplitude"
                    elif headers == ["Time (s)", "Filtered Signal"]:
                        graph_label = "Sinal Filtrado"
                        x_label = "Tempo (s)"
                        y_label = "Amplitude"
                    elif headers == ["Frequency (Hz)", "Amplitude"]:
                        graph_label = "Espectro de Frequência"
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
                plt.plot(x_values, y_values, color='b', label=graph_label)
                plt.title(f"{graph_label} - {method} - {patch}")
                plt.xlabel(x_label)
                plt.ylabel(y_label)
                plt.legend()
                plt.grid(True)

                # Gera o nome do gráfico
                graph_name = f"{method}_{patch}_{signal_type}.png"
                save_path = os.path.join(output_dir, graph_name)

                # Salva o gráfico
                plt.savefig(save_path)
                print(f"Gráfico salvo: {save_path}")
                plt.close()

if __name__ == "__main__":
    # Diretórios de entrada e saída
    input_dir = input("Digite o diretório raiz dos arquivos CSV: ").strip()
    output_dir = input("Digite o diretório onde os gráficos serão salvos: ").strip()

    # Processa os arquivos e salva os gráficos
    process_and_save_csv_graphs(input_dir, output_dir)
