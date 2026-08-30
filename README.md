# U-Net Audio Source Separation (Drum Separation)

Repositório dedicado ao desenvolvimento e treinamento de uma rede neural **U-Net** em PyTorch focada na separação de fontes de áudio, com ênfase no isolamento de bateria a partir de uma mistura musical estéreo.

O projeto utiliza processamento avançado de sinais de áudio combinado com aprendizado profundo para estimar máscaras espectrais ideais, utilizando o dataset padrão da indústria **MUSDB18**.

---

## Dataset Utilizado

O treinamento e a validação do modelo são baseados no **MUSDB18**, um dataset de referência para tarefas de separação de fontes musicais (*Music Source Separation*). 

* **Especificações:** O conjunto de dados contém faixas de áudio desmembradas em hastes (*stems*) separadas, incluindo vocais, baixo, bateria (*drums*) e outros instrumentos (*other*), acompanhadas de suas respectivas misturas estéreo (*mixture*).
* **Processamento:** As faixas brutas são convertidas para mono, reamostradas para uma taxa fixa e segmentadas em janelas temporais (*chunks*) para alimentar a rede neural de forma otimizada durante as épocas de treinamento.
* **Referência Oficial:** [Documentação e Downloads do MUSDB18](https://sigsep.github.io/datasets/musdb.html)

---

## Fluxo Detalhado do Pipeline

O pipeline do projeto transforma ondas sonoras brutas do domínio do tempo em representações espectrais, processa essas matrizes através de uma arquitetura U-Net convolucional e reconstrói o sinal de áudio isolado. O fluxo de dados segue estas etapas:

### 1. Carregamento, Recorte e Alinhamento
* Os arquivos de áudio da mistura e da bateria isolada são carregados utilizando a biblioteca `librosa` sob uma taxa de amostragem padronizada.
* São aplicados recortes temporais (*chunks*) para garantir que a rede receba entradas com dimensões espaciais perfeitamente consistentes, evitando variações de tamanho de lote.

### 2. Transformada de Fourier de Curto Tempo (STFT)
A conversão do sinal sonoro unidimensional do domínio do tempo para o domínio bidimensional de frequência e tempo é o pilar de engenharia de recursos deste projeto:
* **Princípio Matemático:** O áudio bruto consiste em uma única onda oscilando no tempo, o que dificulta para uma rede neural convolucional isolar timbres específicos. A STFT aplica a Transformada de Fourier em janelas deslizantes sobre o sinal.
* **Complexidade e Magnitude:** O resultado da STFT é uma matriz de números complexos contendo partes reais e imaginárias. Aplica-se a função de magnitude absoluta (`np.abs`) para extrair o perfil de energia de cada frequência ao longo dos frames temporais, descartando temporariamente a fase acústica para simplificar o aprendizado inicial do modelo.
* **Ajuste de Dimensão (Bins de Frequência):** Uma STFT padrão configurada com parâmetros usuais gera 1025 bins de frequência. No entanto, para que a matriz possa passar com sucesso pelas camadas de *max pooling* e *upsampling* da U-Net sem quebras dimensionais, o último bin é removido, resultando exatamente em **1024 bins** (número perfeitamente divisível por $2^4 = 16$).
* **Escala Logarítmica e Normalização:** O sistema auditivo humano e as amplitudes musicais possuem variações exponenciais extremas. O espectrograma de magnitude é convertido para uma escala logarítmica em Decibéis (`librosa.amplitude_to_db`) e normalizado estritamente para o intervalo de `0.0` a `1.0`. Isso estabiliza a escala numérica e evita a explosão de gradientes durante a retropropagação.

### 3. Geração da Máscara-Alvo (Target Mask)
* Para treinar o modelo a reconhecer o que deve ser isolado, calcula-se a máscara espectral ideal dividindo a magnitude do espectrograma da bateria isolada pelo espectrograma da mistura (`drums_mag / mix_mag`).
* Os valores resultantes representam a fração de energia da bateria presente em cada coordenada de frequência e tempo, sendo limitados pelo operador `np.clip` entre `0.0` e `1.0`.

### 4. Processamento via U-Net (PyTorch)
* **Encoder:** Camadas convolucionais sequenciais (`ConvBlock` + `InstanceNorm2d` + `ReLU`) que dobram progressivamente os canais de filtros, reduzindo a resolução espacial (`MaxPool2d`) para extrair padrões rítmicos globais de alto nível. Cópias intermediárias são preservadas por meio de conexões de pulo (*skip connections*).
* **Bottleneck:** O ponto mais profundo da rede, focado na abstração máxima dos timbres musicais.
* **Decoder:** Expande a matriz de volta (`Upsample` com interpolação bilinear) e a concatena (`torch.cat`) com as respectivas *skip connections* vindas do Encoder, devolvendo a precisão temporal cirúrgica aos transientes de ataque da bateria.
* **Camada Final:** Uma convolução final de kernel 1x1 reduz os canais internos de volta a 1, seguida por uma ativação **`Sigmoid`** que restringe a saída para uma matriz de probabilidade entre `0.0` e `1.0`.

### 5. Função de Perda e Otimização
* Utiliza uma função de perda baseada em L1 mascarada (`masked_l1_loss`). Esta abordagem calcula o erro absoluto entre a predição e o alvo real, aplicando uma máscara de validação para ignorar trechos de preenchimento artificial (*zero-padding*), garantindo que o modelo seja penalizado exclusivamente sobre dados de áudio válidos.

---

## Demonstração de Áudio

Para validar a eficácia do modelo, realizei testes práticos utilizando trechos de referência, incluindo faixas de metal como **"Psychosocial" (Slipknot)**. Abaixo está a comparação entre o áudio da mistura original e a predição gerada pela U-Net.

### Exemplo: Psychosocial (Slipknot)

**1. Mistura Original (Input):**
<audio controls src="./samples/Psycosocial.wav"></audio>

**2. Bateria Isolada pela U-Net (Prediction):**
<audio controls src="./checkpoints/10ep_lr5e-5_wd1e-3_sisdr-1.95/Psycosocial.wav"></audio>

---

## Estrutura do Repositório

```text
├── checkpoints/                                             # Pesos treinados do modelo (.pt) [Ignorado pelo Git]
|   └── [época]_[learning rate]_[weigth decay]_[SI-SDR]/     # Versionamentos de modelos
├── notebook/                                                # Jupyter Notebook principal contendo o pipeline de ponta a ponta
├── samples/                                                 # Faixas de áudios originais
└── README.md                                                # Documentação oficial do projeto