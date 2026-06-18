Aqui está o Relatório de Fluxo de Dados e Processos focado em Engenharia de Sistemas para o código fornecido.

# Relatório de Fluxo de Dados: Pipeline de Anotação Assistida (Cutie/SAM2)

## 1. Interface de Execução (Entry Point)

### `interactive_demo.py`

* **Comando/Call:** Execução via CLI python.
* `python interactive_demo.py [args]`


* **Parâmetros Críticos (Argumentos):**
* `--images`: Caminho para pasta de imagens fonte (Prioridade 2).
* `--video`: Caminho para arquivo de vídeo (Prioridade 3 - extrai frames).
* `--workspace`: Diretório raiz de estado. Se existir pasta "images" dentro, assume prioridade 1.
* `--num_objects`: Inteiro, pré-aloca buffers de memória para N objetos.
* `--workspace_init_only`: Flag booleana. Se true, configura pastas e sai sem abrir a GUI.


* **Inicialização de Ambiente:**
* Define variáveis de ambiente QT (`QT_QPA_PLATFORM_PLUGIN_PATH`).
* Inicializa Hydra (gerenciador de config) compondo `gui_config`.
* Detecta Device (CUDA > MPS > CPU).



---

## 2. Pipeline de Processamento (Passo-a-Passo)

### A. Ingestão e Gerenciamento de Recursos (`context/resource_manager.py`)

Esta classe atua como camada de abstração de I/O e Cache.

1. **Setup do Workspace:**
* Cria estrutura de diretórios: `images/`, `masks/`, `visualization/`, `soft_masks/{obj_id}/`.
* **Extração/Cópia:** Se entrada for vídeo -> `cv2.VideoCapture` -> extrai frames JPEG. Se for pasta -> copia/redimensiona imagens.


2. **Leitura Sob Demanda (LRU Cache):**
* `get_image(ti)`: Lê JPEG do disco -> Return `np.array (H, W, 3) uint8`.
* `get_mask(ti)`: Lê PNG do disco (se existir) -> Return `np.array (H, W) uint8`.


3. **Persistência Assíncrona (Threaded Writer):**
* Possui uma `Queue` e Worker Threads dedicadas para salvar imagens.
* **Tipos de Save:**
* `mask`: PNG monocromático (índices 0-255).
* `visualization`: JPG/PNG (RGB/RGBA compositado).
* `soft_mask`: PNG (Probabilidades float convertidas para uint8).





### B. Orquestração e Lógica de Estado (`context/main_controller.py`)

O "Cérebro" que mantém o estado da sessão.

1. **Carregamento de Metadados:**
* Lê `object_labels.json`, `models.json`, `sizes.json` do workspace para mapear IDs numéricos -> Strings (Labels/Modelos).


2. **Carga de Dados Externos (YOLO):**
* **Input:** Arquivo JSON via `on_load_yolo_json`.
* **Estrutura:** Mapeia `frame_index` -> Lista de Detecções (`bbox_xyxy`, `class_name`, `confidence`).


3. **Interação do Usuário (Clicks/BBox):**
* Recebe coordenadas (x, y) da GUI.
* **Rota 1 (RITM Local):** Chama `click_ctrl.interact` -> Inferência Local.
* **Rota 2 (SAM2 Remoto):** Chama `sam2_ctrl.interact` ou `predict_bbox`.
* *Fluxo SAM2:* Prepara Payload JSON -> POST HTTP -> Recebe Máscara Base64 -> Decoda para Tensor.




4. **Processamento de Máscara (Cutie Model):**
* Mantém tensor de probabilidade: `curr_prob` (Num_Objs+1, H, W) float.
* **Propagação:** `processor.step()`
* *Input:* Imagem atual + Máscara anterior.
* *Transformação:* Modelos de memória de vídeo propagam a segmentação temporalmente.
* *Trigger de Parada:* Se detectar BBox YOLO "vazia" na máscara atual, pausa a propagação e chama auto-init do SAM2.




5. **Commit de Memória:**
* Atualiza a memória de longo prazo do modelo Cutie com a máscara refinada.



### C. Visualização e Interface (`context/gui.py`)

1. **Rendering:**
* `OverlayCanvas`: Subclasse de QLabel.
* *Camadas:* Desenha Imagem Base -> Desenha Overlay de Máscara (RGBA) -> Desenha Retângulos YOLO (QPainter).


2. **Event Handling:**
* Captura cliques e desenha `QRubberBand` (seleção de retângulo).
* Converte coordenadas de Tela (Widget) <-> Imagem (Real) considerando aspect ratio e padding.



---

## 3. Contratos de Dados (Data Specs)

### Arquivos em Disco (Workspace)

* **Imagens:** `*.jpg` (Frames extraídos/redimensionados).
* **Máscaras (Hard):** `*.png` (Single channel, Palettized). Pixel value = Object ID.
* **Máscaras (Soft):** `soft_masks/{id}/*.png`.
* **Metadados:**
* `object_labels.json`: `{ "1": "Válvula", "2": "Cano" }`
* `models.json`: `{ "1": "VPI-500" }`
* `sizes.json`: `{ "1": "DN200" }`


* **Exportação:**
* Vídeo: `.mp4` (H264, YUV420p).
* Máscaras Binárias: `.png` (0 ou 255).



### Estruturas em Memória (Runtime)

* **Probabilidades (Core):** `torch.Tensor`
* Shape: `(K+1, H, W)` onde K = num_objects.
* Type: `float32` (Logits ou Softmax probabilities).
* Device: CUDA ou CPU.


* **Máscara (Display/Save):** `numpy.ndarray`
* Shape: `(H, W)`
* Type: `uint8`


* **YOLO Data:** `Dict[int, List[Dict]]`
* Key: Frame Index (int).
* Value: Lista de dicts com chaves `bbox_xyxy`, `class_name`, `confidence`.



### Interface API Remota (SAM2)

* **Request:** Multipart/Form-data ou JSON.
* Campos: `image` (JPEG bytes), `points` (JSON list), `labels` (JSON list), `bbox` (JSON list), `frame_idx`, `obj_id`.


* **Response:** JSON `{ "mask_base64": "..." }`.

---

## 4. Pontos de Atenção para Debug

1. **Dependência de API Externa:**
* O arquivo `context/remote_sam_controller.py` tem hardcoded a URL `http://localhost:7263`. Se o serviço SAM2 não estiver rodando nessa porta exata, o pipeline falhará silenciosamente ou retornará erro de conexão.


2. **Discrepância de Estado (Sync):**
* O `ResourceManager` usa Threads para salvar. Se o programa for encerrado abruptamente (SIGKILL), as últimas máscaras na fila (`save_queue`) podem ser perdidas, corrompendo o estado para a próxima execução (`resume`).


3. **Gerenciamento de Memória Dinâmico:**
* O `MainController` recria o `InferenceCore` (`self.processor`) quando o número de objetos muda (expansão dinâmica). Isso é um ponto crítico de falha se a memória da GPU estiver fragmentada.


4. **Sistema de Coordenadas YOLO:**
* O código assume que o JSON do YOLO contém coordenadas absolutas de pixel (`x1, y1, x2, y2`) compatíveis com a resolução das imagens na pasta `workspace/images`. Se o YOLO rodou em resolução diferente e não foi normalizado, as BBoxes ficarão desalinhadas na GUI.


5. **Paths Relativos:**
* O código faz uso de `path.join('./workspace', basename)`. A execução depende estritamente do *current working directory* do terminal ser a raiz do projeto.
