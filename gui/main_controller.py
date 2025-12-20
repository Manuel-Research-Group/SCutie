import os
from os import path
import logging
from typing import Literal, Dict
import json
import cv2
# fix conflicts between qt5 and cv2
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")

import torch
try:
    from torch import mps
except:
    print('torch.MPS not available.')
from torch import autocast
from torchvision.transforms.functional import to_tensor
import numpy as np
from omegaconf import DictConfig, open_dict

from cutie.model.cutie import CUTIE
from cutie.inference.inference_core import InferenceCore

from gui.interaction import *
from gui.interactive_utils import *
from gui.resource_manager import ResourceManager
from gui.gui import GUI
from gui.click_controller import ClickController
#from gui.sam2_click_controller import SAM2ClickController
from gui.reader import PropagationReader, get_data_loader
from gui.exporter import convert_frames_to_video, convert_mask_to_binary
from gui.remote_sam_controller import RemoteSAMController

from cutie.utils.download_models import download_models_if_needed

import shutil
from PySide6.QtWidgets import QMessageBox

log = logging.getLogger()

class MainController():

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.initialized = False

        # setting up the workspace
        if cfg["workspace"] is None:
            if cfg["images"] is not None:
                basename = path.basename(cfg["images"])
            elif cfg["video"] is not None:
                basename = path.basename(cfg["video"])[:-4]
            else:
                raise NotImplementedError('Either images, video, or workspace has to be specified')

            cfg["workspace"] = path.join(cfg['workspace_root'], basename)

        # reading arguments
        self.cfg = cfg
        self.num_objects = cfg['num_objects']
        self.device = cfg['device']
        self.amp = cfg['amp']

        # initializing the network(s)
        self.object_labels: Dict[int, str] = {}
        self.labels_file_path = path.join(self.cfg['workspace'], 'object_labels.json')
        self.load_labels()

        self.object_models: Dict[int, str] = {}
        self.models_file_path = path.join(self.cfg['workspace'], 'models.json')
        self.load_models()

        self.object_sizes: Dict[int, str] = {}
        self.sizes_file_path = path.join(self.cfg['workspace'], 'sizes.json')
        self.load_sizes()


        # Update num_objects from loaded labels if higher
        cfg['num_objects'] = max(cfg['num_objects'], max(self.object_labels.keys()) if self.object_labels else 0)
        self.initialize_networks()

        # main components
        self.res_man = ResourceManager(cfg)
        if 'workspace_init_only' in cfg and cfg['workspace_init_only']:
            return
        self.processor = InferenceCore(self.cutie, self.cfg)
        self.num_objects = cfg['num_objects'] # Re-set num_objects after potential update
        self.gui = GUI(self, self.cfg)

        # initialize control info
        self.length: int = self.res_man.length
        self.interaction: Interaction = None
        self.interaction_type: str = 'Click'
        self.app_mode: str = 'annotation'
        self.selection_tool: str = 'click'
        self.last_deleted_info: Dict = None
        self.curr_ti: int = 0
        self.curr_object: int = 1
        self.propagating: bool = False
        self.propagate_direction: Literal['forward', 'backward', 'none'] = 'none'
        self.last_ex = self.last_ey = 0

        # current frame info
        self.curr_frame_dirty: bool = False
        self.curr_image_np: np.ndarray = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.curr_image_torch: torch.Tensor = None
        self.curr_mask: np.ndarray = np.zeros((self.h, self.w), dtype=np.uint8)
        self.curr_prob: torch.Tensor = torch.zeros((self.num_objects + 1, self.h, self.w),
                                                   dtype=torch.float).to(self.device)
        self.curr_prob[0] = 1

        # visualization info
        self.vis_mode: str = 'davis'
        self.vis_image: np.ndarray = None
        self.save_visualization_mode: str = 'None'
        self.save_soft_mask: bool = False

        self.interacted_prob: torch.Tensor = None
        self.overlay_layer: np.ndarray = None
        self.overlay_layer_torch: torch.Tensor = None

        # the object id used for popup/layer overlay
        self.vis_target_objects = list(range(1, self.num_objects + 1))

        self.load_current_image_mask()
        self.show_current_frame()

        # initialize stuff
        self.update_memory_gauges()
        self.update_gpu_gauges()
        self.gui.work_mem_min.setValue(self.processor.memory.min_mem_frames)
        self.gui.work_mem_max.setValue(self.processor.memory.max_mem_frames)
        self.gui.long_mem_max.setValue(self.processor.memory.max_long_tokens)
        self.gui.mem_every_box.setValue(self.processor.mem_every)

        # for exporting videos
        self.output_fps = cfg['output_fps']
        self.output_bitrate = cfg['output_bitrate']

        # set callbacks
        self.gui.on_mouse_motion_xy = self.on_mouse_motion_xy
        self.gui.click_fn = self.click_fn

        self.show_yolo_suggestions = True
        self.yolo_data = {} # Dict[int, List[Dict]] -> {frame_idx: [detections]}

        self.active_model_name = 'RITM'

        self.gui.showMaximized()
        self.gui.text('Initialized.')
        self.initialized = True

        # try to load the default overlay
        self._try_load_layer('./docs/uiuc.png')
        self.gui.set_object_color(self.curr_object)
        self.update_config()
        self.gui.update_object_label(self.object_labels.get(self.curr_object, ""))
        self.gui.update_object_model(self.object_models.get(self.curr_object, ""))
        self.gui.update_object_size(self.object_sizes.get(self.curr_object, ""))

    def on_load_yolo_json(self):
        file_name = self.gui.open_file('YOLO JSON') # Pode precisar adaptar open_file para aceitar JSON ou usar QFileDialog direto
        if not file_name or not file_name.endswith('.json'):
            return
            
        try:
            with open(file_name, 'r') as f:
                raw_data = json.load(f)
            
            self.yolo_data = {}
            loaded_count = 0
            
            # Mapear nomes de arquivos para índices
            name_to_idx = {name: i for i, name in enumerate(self.res_man.names)}
            
            for key, detections in raw_data.items():
                # O JSON chave pode ser "00000.jpg" ou "00000"
                clean_key = key.replace('.jpg', '').replace('.png', '')
                
                if clean_key in name_to_idx:
                    idx = name_to_idx[clean_key]
                    self.yolo_data[idx] = detections
                    loaded_count += len(detections)
            
            self.gui.text(f"Carregadas {loaded_count} detecções em {len(self.yolo_data)} frames.")
            self.show_current_frame() # Repaint
            
        except Exception as e:
            self.gui.text(f"Erro ao carregar JSON: {e}")

    def on_init_frame_from_yolo(self):
        """
        Inicializa o frame atual processando TODAS as detecções YOLO já carregadas na memória.
        Gera máscaras com SAM 2 e injeta no sistema.
        """
        # 1. Obter candidatos da memória (Hashmap self.yolo_data)
        candidates = self.get_current_yolo_candidates()
        
        if not candidates:
            self.gui.text(f"Nenhuma detecção YOLO carregada para o frame {self.curr_ti}.")
            self.gui.text("Dica: Carregue o JSON em 'YOLO -> Load YOLO JSON' primeiro.")
            return

        # 2. Configuração de Limiares
        # Ajuste este valor conforme a qualidade do seu detector
        CONFIDENCE_THRESHOLD = 0.5
        
        # Filtra e ordena por confiança (do maior para o menor)
        valid_candidates = [c for c in candidates if c.get('confidence', 0) >= CONFIDENCE_THRESHOLD]
        valid_candidates.sort(key=lambda x: x['confidence'], reverse=True)
        
        if not valid_candidates:
            self.gui.text(f"Existem detecções, mas nenhuma com confiança > {CONFIDENCE_THRESHOLD}.")
            return

        self.gui.text(f"Auto-Init: Processando {len(valid_candidates)} objetos...")
        self.gui.process_events()

        # 3. Preparar o Canvas Temporário
        # Se já existir máscara no frame (ex: desenhada à mão), começamos dela.
        # Se não, criamos uma vazia.
        if self.curr_mask is not None:
            final_mask = self.curr_mask.copy()
        else:
            final_mask = np.zeros((self.h, self.w), dtype=np.uint8)

        # Determinar o próximo ID disponível
        # Começamos do maior ID presente na máscara atual + 1, ou do ID 1 se estiver vazia.
        current_max_id = int(final_mask.max())
        # Se max for 0 (fundo), começamos do 1. Se for 5, começamos do 6.
        next_obj_id = current_max_id + 1
        
        added_count = 0

        # 4. Loop de Processamento
        for i, det in enumerate(valid_candidates):
            bbox = det['bbox_xyxy']
            class_name = det.get('class_name', 'obj')
            
            # Otimização: Verifica se a caixa já está "muito ocupada" antes de chamar a API
            # Isso economiza tempo de rede se o objeto já estiver segmentado
            # Mas como é só bbox, o cálculo de IoU aqui seria impreciso, melhor deixar a máscara decidir.

            self.gui.text(f" -> SAM 2 gerando: {class_name} ({i+1}/{len(valid_candidates)})...")
            self.gui.process_events()

            # Chama a API do SAM 2 (usando seu controlador remoto existente)
            # Passamos IDs temporários (-1) para não sujar o histórico de cliques de refinamento
            mask_tensor = self.sam2_ctrl.predict_bbox(
                self.curr_image_np, 
                bbox, 
                frame_idx=self.curr_ti, 
                obj_id=next_obj_id
            )

            if mask_tensor is not None:
                # Converter Tensor (0.0-1.0) para Máscara Booleana Numpy
                mask_bool = mask_tensor.squeeze().cpu().numpy() > 0.5
                
                # --- Lógica de Sobreposição (Overlap) ---
                # Verifica quantos pixels da nova máscara cairiam em cima de objetos já existentes
                overlap = np.logical_and(final_mask > 0, mask_bool)
                overlap_area = np.sum(overlap)
                mask_area = np.sum(mask_bool)
                
                # Se a nova máscara for > 50% coberta por coisas que já existem, ignoramos.
                # Isso evita duplicatas se o YOLO detectou o mesmo cano duas vezes.
                if mask_area > 0 and (overlap_area / mask_area) < 0.5:
                    
                    # 5. Expandir capacidade do Cutie se necessário
                    # Se vamos adicionar o objeto 10 e o sistema só suporta 5, precisamos expandir
                    if next_obj_id > self.num_objects:
                        self.num_objects = next_obj_id
                        self.cfg['num_objects'] = self.num_objects
                        self.gui.object_dial.setMaximum(self.num_objects)
                        self.res_man.add_object_directory(self.num_objects)

                    # 6. Pintar no Canvas
                    final_mask[mask_bool] = next_obj_id
                    
                    # Registrar Label
                    self.object_labels[next_obj_id] = f"{class_name}"
                    
                    next_obj_id += 1
                    added_count += 1
                else:
                    # Opcional: logar que foi ignorado por sobreposição
                    pass

        # 7. Atualizar o Sistema (Cutie e UI)
        if added_count > 0:
            self.curr_mask = final_mask
            
            # Recria o tensor de probabilidades (One-Hot) com o novo tamanho e dados
            # Isso é CRÍTICO para o Cutie saber que os objetos existem
            self.curr_prob = index_numpy_to_one_hot_torch(self.curr_mask, self.num_objects + 1).to(
                self.device, non_blocking=True)
            
            self.save_labels()       # Salva os nomes (valves, pipes) no JSON
            self.save_current_mask() # Salva o PNG no disco
            self.show_current_frame() # Atualiza a tela
            
            self.gui.text(f"Sucesso! {added_count} objetos inicializados no Frame {self.curr_ti}.")
            
            # Muda o foco para o último objeto criado para facilitar edição
            self.hit_number_key(next_obj_id - 1)
        else:
            self.gui.text("Processo finalizado, mas nenhum objeto novo foi adicionado (sobreposição ou falha na API).")

    def on_toggle_yolo(self, checked):
        self.show_yolo_suggestions = checked
        self.show_current_frame()

    def get_current_yolo_candidates(self):
        return self.yolo_data.get(self.curr_ti, [])

    def get_box_index_at(self, x, y):
        """ Retorna o índice da caixa YOLO sob o mouse, ou -1 """
        candidates = self.get_current_yolo_candidates()
        # Itera de trás pra frente (caso haja sobreposição, pega a mais "recente/topo")
        for i in range(len(candidates) - 1, -1, -1):
            box = candidates[i]
            x1, y1, x2, y2 = box['bbox_xyxy']
            if x1 <= x <= x2 and y1 <= y <= y2:
                return i
        return -1

    def get_yolo_box_rect(self, idx):
        candidates = self.get_current_yolo_candidates()
        if 0 <= idx < len(candidates):
            return candidates[idx]['bbox_xyxy']
        return None

    def remove_yolo_candidate(self, idx):
        if self.curr_ti in self.yolo_data:
             if 0 <= idx < len(self.yolo_data[self.curr_ti]):
                 self.yolo_data[self.curr_ti].pop(idx)
                 self.gui.main_canvas.update()

    def accept_yolo_candidate(self, idx):
        candidates = self.get_current_yolo_candidates()
        if idx < 0 or idx >= len(candidates):
            return
            
        box = candidates[idx]
        x1, y1, x2, y2 = box['bbox_xyxy']
        class_name = box.get('class_name', 'Unknown')
        
        # 1. Cria novo objeto e configura label
        self.on_add_object() 
        curr_id = self.curr_object
        self.object_labels[curr_id] = class_name
        self.gui.update_object_label(class_name)
        self.save_labels()
        
        # 2. Gera a Máscara
        if self.active_model_name == 'SAM2':
            self.gui.text(f"Enviando BBox para SAM 2 API...")
            self.gui.process_events()
            
            image_np = self.curr_image_np 
            bbox = [x1, y1, x2, y2]
            
            mask_tensor = self.sam2_ctrl.predict_bbox(
                image_np, 
                bbox, 
                frame_idx=self.curr_ti, 
                obj_id=curr_id
            )
        
            if mask_tensor is not None:
                
                if self.curr_prob is None:
                    self.convert_current_image_mask_torch()
                
                prob_map = self.curr_prob.clone()
                
                prob_map[curr_id] = mask_tensor

                self.interacted_prob = aggregate_wbg(prob_map[1:], keep_bg=True, hard=True)
                
                self.update_interacted_mask()
                self.gui.text(f"Máscara SAM 2 recebida para objeto {curr_id}.")
            else:
                self.gui.text("Erro ao obter máscara do SAM 2.")
                
        else:
            self.on_bbox_complete(x1, y1, x2, y2)
        
        self.remove_yolo_candidate(idx)
        
    def set_app_mode(self, mode: str):
        if mode not in ['annotation', 'view']:
            return
        
        self.app_mode = mode
        self.gui.text(f"Modo alterado para: {mode}")
        
        # Se mudar para visualização, cancela interações pendentes
        if mode == 'view':
            self.reset_this_interaction()
            
        # Atualiza a UI
        self.gui.toggle_mode_ui(mode)

    def set_selection_tool(self, tool: str):
        self.selection_tool = tool
        self.gui.text(f"Ferramenta de seleção: {tool}")

    def on_bbox_complete(self, x1, y1, x2, y2):
        """
        Chamado quando o usuário solta o mouse após desenhar um retângulo.
        Converte a caixa em um clique no centro.
        """
        # Evita caixas muito pequenas (cliques acidentais arrastados)
        if abs(x2 - x1) < 2 or abs(y2 - y1) < 2:
            return

        # Calcula o centro
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        self.gui.text(f"Bounding Box detectada: Centro ({center_x:.1f}, {center_y:.1f})")

        # Envia como um clique esquerdo (positivo)
        # Reutiliza a lógica existente de clique
        self.click_fn('left', center_x, center_y)

    def get_object_info_at(self, x: float, y: float) -> str:
        """
        Retorna uma string com ID e Label do objeto na coordenada x, y.
        Retorna None se for fundo ou inválido.
        """
        if self.curr_mask is None:
            return None

        ix, iy = int(x), int(y)
        
        # Garante que está dentro dos limites da imagem
        if 0 <= iy < self.h and 0 <= ix < self.w:
            obj_id = self.curr_mask[iy, ix]
            
            if obj_id > 0:
                lbl = self.object_labels.get(obj_id, "No Label")
                mod = self.object_models.get(obj_id, "")
                siz = self.object_sizes.get(obj_id, "")
                
                # Creates a string in the following pattern (ID: 1 | Butterfly Valve | VPI 54059 | DN200)
                info = f"ID: {obj_id} | {lbl}"
                if mod: info += f" | {mod}"
                if siz: info += f" | {siz}"
                
                return info
                
        return None

    def load_sizes(self):
        if path.exists(self.sizes_file_path):
            try:
                with open(self.sizes_file_path, 'r') as f:
                    loaded_data = json.load(f)
                    self.object_sizes = {int(k): v for k, v in loaded_data.items()}
                log.info(f"Loaded {len(self.object_sizes)} object sizes from {self.sizes_file_path}")
            except Exception as e:
                log.error(f"Failed to load sizes: {e}")

    def save_sizes(self):
        try:
            with open(self.sizes_file_path, 'w') as f:
                json.dump(self.object_sizes, f, indent=4)
        except Exception as e:
            self.gui.text(f"Error saving sizes: {e}")

    def on_size_changed(self):
        new_size = self.gui.object_size_edit.text().strip()
        
        if new_size.isdigit():
            new_size = f"DN{new_size}"
            self.gui.update_object_size(new_size)
        
        if not new_size:
            if self.curr_object in self.object_sizes:
                del self.object_sizes[self.curr_object]
                self.gui.text(f"Removed size for object {self.curr_object}")
        else:
            self.object_sizes[self.curr_object] = new_size
            self.gui.text(f"Set size for object {self.curr_object} to: {new_size}")
            
        self.save_sizes()

    def load_labels(self):
        if path.exists(self.labels_file_path):
            try:
                with open(self.labels_file_path, 'r') as f:
                    loaded_data = json.load(f)
                    # JSON keys are strings, convert them back to int
                    self.object_labels = {int(k): v for k, v in loaded_data.items()}
                log.info(f"Loaded {len(self.object_labels)} object labels from {self.labels_file_path}")
            except Exception as e:
                log.error(f"Failed to load labels: {e}")

    def save_labels(self):
        try:
            with open(self.labels_file_path, 'w') as f:
                json.dump(self.object_labels, f, indent=4)
            # self.gui.text(f"Saved labels to {self.labels_file_path}")
        except Exception as e:
            self.gui.text(f"Error saving labels: {e}")

    def load_models(self):
        if path.exists(self.models_file_path):
            try:
                with open(self.models_file_path, 'r') as f:
                    data = json.load(f)
                    self.object_models = {int(k): v for k, v in data.items()}
                log.info(f"Loaded {len(self.object_models)} models.")
            except Exception as e:
                log.error(f"Failed to load models: {e}")

    def save_models(self):
        try:
            with open(self.models_file_path, 'w') as f:
                json.dump(self.object_models, f, indent=4)
        except Exception as e:
            self.gui.text(f"Error saving models: {e}")

    def on_model_changed(self):
        text = self.gui.object_model_edit.text().strip()
        
        if not text:
            if self.curr_object in self.object_models:
                del self.object_models[self.curr_object]
                self.gui.text(f"Removed model for object {self.curr_object}")
        else:
            self.object_models[self.curr_object] = text
            self.gui.text(f"Set model for object {self.curr_object} to: {text}")
            
        self.save_models()

    def initialize_networks(self) -> None:
        download_models_if_needed()
        self.cutie = CUTIE(self.cfg).eval().to(self.device)
        # Update num_objects in cfg before loading weights if it changed
        model_weights = torch.load(self.cfg.weights, map_location=self.device, weights_only=True)
        self.cutie.load_weights(model_weights)

        self.click_ctrl = ClickController(self.cfg.ritm_weights, device=self.device)
        self.sam2_ctrl = RemoteSAMController(api_url="http://localhost:7263", device=self.device)

        self.ritm_ctrl = ClickController(self.cfg.ritm_weights, device=self.device)
        self.sam2_ctrl = RemoteSAMController(api_url="http://localhost:7263", device=self.device)
        self.click_ctrl = self.ritm_ctrl


        ''' 
        #self.click_ctrl = ClickController(self.cfg.ritm_weights, device=self.device)
        if not hasattr(self.cfg, 'sam2_weights') or not hasattr(self.cfg, 'sam2_config'):
            log.error("sam2_weights ou sam2_config não encontrados no seu .yaml.")
            raise ValueError("Faltando caminhos de configuração do SAM 2")
            
        self.click_ctrl = SAM2ClickController(
            checkpoint_path=self.cfg.sam2_weights,
            model_cfg_path=self.cfg.sam2_config,
            device=self.device
        )
        '''

    def set_segmentation_model(self, model_name: str):
        """Alterna entre os modelos"""
        if model_name == 'RITM':
            self.click_ctrl = self.ritm_ctrl
            self.active_model_name = 'RITM'
            self.gui.text("Modelo alterado para: RITM (Local)")
        elif model_name == 'SAM2':
            self.click_ctrl = self.sam2_ctrl
            self.active_model_name = 'SAM2'
            self.gui.text("Modelo alterado para: SAM 2 (API Remota)")

    def hit_number_key(self, number: int):
        number = int(number)

        if number == self.curr_object:
            return

        if hasattr(self.click_ctrl, 'reset_context'):
            self.click_ctrl.reset_context()

        self.curr_object = number
        self.gui.object_dial.setValue(number)
        if self.click_ctrl is not None:
            self.click_ctrl.unanchor()
        self.gui.text(f'Current object changed to {number}.')
        self.gui.set_object_color(number)
        self.show_current_frame()
        self.gui.update_object_label(self.object_labels.get(self.curr_object, ""))
        self.gui.update_object_model(self.object_models.get(self.curr_object, ""))
        self.gui.update_object_size(self.object_sizes.get(self.curr_object, ""))

    def click_fn(self, action: Literal['left', 'right', 'middle', 'pick'], x: int, y: int):
        if self.propagating:
            return

        if self.app_mode == 'view':
            if action in ['left', 'right']:
                self.gui.text("Edição bloqueada no Modo de Visualização.")
                return

        if action == 'pick':
            target_object = self.curr_mask[int(y), int(x)]
            
            if target_object == 0:
                self.gui.text("Fundo (Background) clicado. Nenhum objeto selecionado.")
                return

            self.gui.text(f"Objeto {target_object} selecionado pelo clique.")
            self.hit_number_key(target_object)
            return

        last_interaction = self.interaction
        new_interaction = None

        with autocast(self.device, enabled=(self.amp and self.device == 'cuda')):
            if action in ['left', 'right']:
                # left: positive click
                # right: negative click
                self.convert_current_image_mask_torch()
                image = self.curr_image_torch
                if (last_interaction is None or last_interaction.tar_obj != self.curr_object):
                    # create new interaction is needed
                    self.complete_interaction()
                    self.click_ctrl.unanchor()
                    new_interaction = ClickInteraction(image, self.curr_prob, (self.h, self.w),
                                                       self.click_ctrl, self.curr_object)
                    if new_interaction is not None:
                        self.interaction = new_interaction

                self.interaction.push_point(x, y, is_neg=(action == 'right'))
                self.interacted_prob = self.interaction.predict().to(self.device, non_blocking=True)
                self.update_interacted_mask()
                self.update_gpu_gauges()

            elif action == 'middle':
                # middle: select a new visualization object
                target_object = self.curr_mask[int(y), int(x)]
                if target_object in self.vis_target_objects:
                    self.vis_target_objects.remove(target_object)
                else:
                    self.vis_target_objects.append(target_object)
                self.gui.text(f'Overlay target(s) changed to {self.vis_target_objects}')
                self.show_current_frame()
                return
            else:
                raise NotImplementedError

    def load_current_image_mask(self, no_mask: bool = False):
        self.curr_image_np = self.res_man.get_image(self.curr_ti)
        self.curr_image_torch = None

        if not no_mask:
            loaded_mask = self.res_man.get_mask(self.curr_ti)
            if loaded_mask is None:
                self.curr_mask.fill(0)
            else:
                self.curr_mask = loaded_mask.copy()
                
                max_id_in_mask = int(self.curr_mask.max())
                if max_id_in_mask > self.num_objects:
                    self.gui.text(f"Mask has object {max_id_in_mask}. Updating object count from {self.num_objects}.")
                    for new_id in range(self.num_objects + 1, max_id_in_mask + 1):
                        self.res_man.add_object_directory(new_id)
                    
                    self.num_objects = max_id_in_mask
                    self.cfg['num_objects'] = self.num_objects
                    self.gui.object_dial.setMaximum(self.num_objects)
                    self.gui.text(f"Object count set to {self.num_objects}.")

            self.curr_prob = None

    def convert_current_image_mask_torch(self, no_mask: bool = False):
        if self.curr_image_torch is None:
            self.curr_image_torch = to_tensor(self.curr_image_np).to(self.device, non_blocking=True)

        if self.curr_prob is None and not no_mask:
            self.curr_prob = index_numpy_to_one_hot_torch(self.curr_mask, self.num_objects + 1).to(
                self.device, non_blocking=True)

    def compose_current_im(self):
        self.vis_image = get_visualization(self.vis_mode, self.curr_image_np, self.curr_mask,
                                           self.overlay_layer, self.vis_target_objects)

    def update_canvas(self):
        self.gui.set_canvas(self.vis_image)

    def update_current_image_fast(self, invalid_soft_mask: bool = False):
        # fast path, uses gpu. Changes the image in-place to avoid copying
        # thus current_image_torch must be voided afterwards
        # do_no_save_soft_mask is an override to solve #41
        self.vis_image = get_visualization_torch(self.vis_mode, self.curr_image_torch,
                                                 self.curr_prob, self.overlay_layer_torch,
                                                 self.vis_target_objects)
        self.curr_image_torch = None
        self.vis_image = np.ascontiguousarray(self.vis_image)
        save_visualization = self.save_visualization_mode in [
            'Propagation only (higher quality)', 'Always'
        ]
        if save_visualization and not invalid_soft_mask:
            self.res_man.save_visualization(self.curr_ti, self.vis_mode, self.vis_image)
        if self.save_soft_mask and not invalid_soft_mask:
            self.res_man.save_soft_mask(self.curr_ti, self.curr_prob.cpu().numpy())
        self.gui.set_canvas(self.vis_image)

    def show_current_frame(self, fast: bool = False, invalid_soft_mask: bool = False):
        # Re-compute overlay and show the image
        if fast:
            self.update_current_image_fast(invalid_soft_mask)
        else:
            self.compose_current_im()
            if self.save_visualization_mode == 'Always':
                self.res_man.save_visualization(self.curr_ti, self.vis_mode, self.vis_image)
            self.update_canvas()

        self.gui.update_slider(self.curr_ti)
        self.gui.frame_name.setText(self.res_man.names[self.curr_ti] + '.jpg')

    def set_vis_mode(self):
        self.vis_mode = self.gui.combo.currentText()
        self.show_current_frame()

    def save_current_mask(self):
        # save mask to hard disk
        self.res_man.save_mask(self.curr_ti, self.curr_mask)

    def on_slider_update(self):
        # 1. Captura para onde o slider foi movido
        target_ti = self.gui.tl_slider.value()

        if not self.propagating:
            # --- MODO EDIÇÃO / PLAYBACK ---
            
            # A. SALVAR O PASSADO: 
            # Verifica se o frame ANTIGO (self.curr_ti atual) precisa ser salvo.
            # Como ainda não atualizamos self.curr_ti, isso salva no arquivo correto (o anterior).
            if self.curr_frame_dirty:
                self.save_current_mask()
            
            self.curr_frame_dirty = False

            if hasattr(self.click_ctrl, 'reset_context'):
                self.click_ctrl.reset_context()

            self.reset_this_interaction()
            
            self.curr_ti = target_ti
            
            self.load_current_image_mask()
            self.show_current_frame()

        else:
            self.curr_ti = target_ti

    def on_jump_to_frame(self):
        if self.propagating:
            return
        
        current_text = self.gui.lcd.text()
        
        try:
            target_frame_str = current_text.split('/')[0].strip()
            target_frame = int(target_frame_str)

            if 0 <= target_frame < self.length:
                self.gui.tl_slider.setValue(target_frame)
            else:
                self.gui.text(f"Frame inválido: {target_frame}. Deve estar entre 0 e {self.length - 1}.")
                self.gui.lcd.blockSignals(True)
                self.gui.lcd.setText(f'{self.curr_ti} / {self.length - 1}')
                self.gui.lcd.blockSignals(False)

        except ValueError:
            self.gui.text(f"Entrada inválida: '{current_text}' não é um número.")
            self.gui.lcd.blockSignals(True)
            self.gui.lcd.setText(f'{self.curr_ti} / {self.length - 1}')
            self.gui.lcd.blockSignals(False)

    def on_forward_propagation(self):
        if self.propagating:
            # acts as a pause button
            self.propagating = False
            self.propagate_direction = 'none'
        else:
            self.propagate_fn = self.on_next_frame
            self.gui.forward_propagation_start()
            self.propagate_direction = 'forward'
            self.on_propagate()

    def on_backward_propagation(self):
        if self.propagating:
            # acts as a pause button
            self.propagating = False
            self.propagate_direction = 'none'
        else:
            self.propagate_fn = self.on_prev_frame
            self.gui.backward_propagation_start()
            self.propagate_direction = 'backward'
            self.on_propagate()

    def on_pause(self):
        self.propagating = False
        self.gui.text(f'Propagation stopped at t={self.curr_ti}.')
        self.gui.pause_propagation()
    
    def on_propagate(self):
        with autocast(self.device, enabled=(self.amp and self.device == 'cuda')):
            self.convert_current_image_mask_torch()

            # *** DEBUG LOG 1 ***
            print(f"\n=== DEBUG: Início on_propagate ===")
            print(f"curr_ti: {self.curr_ti}")
            print(f"num_objects (self): {self.num_objects}")
            print(f"num_objects (cfg): {self.cfg['num_objects']}")
            print(f"curr_prob.shape: {self.curr_prob.shape if self.curr_prob is not None else 'None'}")
            print(f"curr_mask unique values: {np.unique(self.curr_mask)}")
            
            self.gui.text(f'Propagation started at t={self.curr_ti}.')
            
            # --- CORREÇÃO: Gerenciamento Inteligente de Memória ---
            # Detecta se houve mudança no número de objetos desde a última propagação
            # Registra o estado atual para próxima comparação
            if not hasattr(self, '_last_propagation_num_objects'):
                self._last_propagation_num_objects = self.num_objects
            
            # Sempre limpa no frame 0 (início do vídeo)
            if self.curr_ti == 0:
                print(f"DEBUG: Limpando memória sensorial (frame 0)")
                self.processor.clear_sensory_memory()
            # ------------------------------------------------------

            # 1. Limpeza de Fantasmas (Mantido)
            self.curr_mask = torch_prob_to_numpy_mask(self.curr_prob)
            self.curr_prob = index_numpy_to_one_hot_torch(self.curr_mask, self.num_objects + 1).to(
                self.device, non_blocking=True)

            # *** DEBUG LOG 2 ***
            print(f"DEBUG: Após limpeza de fantasmas")
            print(f"curr_prob.shape: {self.curr_prob.shape}")
            print(f"curr_mask unique values: {np.unique(self.curr_mask)}")

            # *** CORREÇÃO: Atualiza a configuração do processador ***
            print(f"DEBUG: Chamando processor.update_config() com num_objects={self.cfg['num_objects']}")
            self.processor.update_config(self.cfg)
            
            # *** CORREÇÃO CRÍTICA: Reinicializa o processador se o num_objects mudou ***
            # O update_config não é suficiente - precisamos recriar a instância
            if hasattr(self, '_last_propagation_num_objects'):
                if self._last_propagation_num_objects != self.num_objects:
                    print(f"DEBUG: Recriando InferenceCore para suportar {self.num_objects} objetos")
                    # Mantém a referência ao modelo Cutie, mas recria o processador
                    self.processor = InferenceCore(self.cutie, self.cfg)
                    print(f"DEBUG: InferenceCore recriado com sucesso")
            
            # *** DEBUG LOG 3: Verificar estado interno do processador ***
            try:
                if hasattr(self.processor, 'num_objects'):
                    print(f"DEBUG: processor.num_objects = {self.processor.num_objects}")
                if hasattr(self.processor.memory, 'num_objects'):
                    print(f"DEBUG: processor.memory.num_objects = {self.processor.memory.num_objects}")
            except Exception as e:
                print(f"DEBUG: Erro ao verificar processor state: {e}")

            # 2. Inicialização / Reinserção
            print(f"DEBUG: Chamando processor.step() com force_permanent=True")
            
            # *** CORREÇÃO DEFINITIVA: Converter para formato idx_mask ***
            # O problema: quando passamos curr_prob[1:] (4 canais), o código em inference_core
            # linha 280 faz mask[tmp_id] onde tmp_id vai de 1-4, mas mask tem índices 0-3
            # Solução: Converter probabilidades para máscara de índices antes de passar
            idx_mask_np = torch_prob_to_numpy_mask(self.curr_prob)  # H x W com valores 0-4
            idx_mask_torch = torch.from_numpy(idx_mask_np).to(self.device)
            
            all_object_ids = list(range(1, self.num_objects + 1))
            print(f"DEBUG: idx_mask unique values: {torch.unique(idx_mask_torch)}")
            print(f"DEBUG: Passando object_ids: {all_object_ids}")
            
            self.curr_prob = self.processor.step(
                self.curr_image_torch,
                idx_mask_torch,              # Passa como H x W (não como canais separados)
                objects=all_object_ids,
                idx_mask=True,               # *** CRUCIAL: True aqui ***
                force_permanent=True
            )
            
            print(f"DEBUG: Após primeiro step(), curr_prob.shape = {self.curr_prob.shape}")
            
            self.curr_mask = torch_prob_to_numpy_mask(self.curr_prob)
            
            self.interacted_prob = None
            self.reset_this_interaction()
            self.show_current_frame(fast=True, invalid_soft_mask=True)

            self.propagating = True
            self.gui.clear_all_mem_button.setEnabled(False)
            self.gui.clear_non_perm_mem_button.setEnabled(False)
            self.gui.tl_slider.setEnabled(False)

            dataset = PropagationReader(self.res_man, self.curr_ti, self.propagate_direction)
            loader = get_data_loader(dataset, self.cfg.num_read_workers)

            # --- LOOP DE PROPAGAÇÃO ---
            frame_count = 0
            for data in loader:
                if not self.propagating:
                    break
                
                frame_count += 1
                self.curr_image_np, self.curr_image_torch = data
                self.curr_image_torch = self.curr_image_torch.to(self.device, non_blocking=True)
                self.propagate_fn()

                # *** DEBUG LOG 4 ***
                print(f"\n--- DEBUG: Frame {self.curr_ti} (iteração {frame_count}) ---")
                print(f"curr_prob.shape antes do step: {self.curr_prob.shape}")

                # Cutie Propaga
                self.curr_prob = self.processor.step(self.curr_image_torch)
                
                print(f"curr_prob.shape após step: {self.curr_prob.shape}")
                
                self.curr_mask = torch_prob_to_numpy_mask(self.curr_prob)

                # --- FISCAL YOLO ---
                yolo_candidates = self.yolo_data.get(self.curr_ti, [])
                stop_propagation = False
                if yolo_candidates:
                    high_conf_candidates = [c for c in yolo_candidates if c['confidence'] > 0.8]
                    print(f"DEBUG: {len(high_conf_candidates)} candidatos YOLO com confiança > 0.8")
                    
                    for det in high_conf_candidates:
                        bbox = det['bbox_xyxy']
                        if self.is_box_empty_in_mask(bbox, self.curr_mask):
                            class_name = det.get('class_name', 'Unknown')
                            self.gui.text(f"ALERTA: Novo objeto detectado: {class_name}")
                            
                            # Ação Automática SAM
                            next_obj_id = self.num_objects + 1
                            
                            print(f"DEBUG: Tentando adicionar objeto {next_obj_id}")
                            print(f"DEBUG: curr_prob.shape ANTES de expansão: {self.curr_prob.shape}")
                            
                            # Chama API
                            mask_tensor = self.sam2_ctrl.predict_bbox(
                                self.curr_image_np, bbox, frame_idx=self.curr_ti, obj_id=next_obj_id
                            )
                            
                            if mask_tensor is not None:
                                print(f"DEBUG: Máscara SAM2 recebida, shape: {mask_tensor.shape}")
                                
                                # *** CORREÇÃO: Expande curr_prob ANTES ***
                                old_prob = self.curr_prob
                                new_shape = (next_obj_id + 1, self.h, self.w)
                                print(f"DEBUG: Expandindo curr_prob de {old_prob.shape} para {new_shape}")
                                
                                self.curr_prob = torch.zeros(new_shape, dtype=old_prob.dtype, device=self.device)
                                self.curr_prob[:old_prob.shape[0]] = old_prob
                                
                                print(f"DEBUG: curr_prob.shape APÓS expansão: {self.curr_prob.shape}")
                                
                                # Atualiza metadados
                                self.on_add_object()
                                
                                print(f"DEBUG: Atribuindo máscara ao canal {next_obj_id}")
                                self.curr_prob[next_obj_id] = mask_tensor
                                
                                print(f"DEBUG: Chamando aggregate_wbg")
                                self.curr_prob = aggregate_wbg(self.curr_prob[1:], keep_bg=True, hard=True)
                                
                                print(f"DEBUG: curr_prob.shape após aggregate: {self.curr_prob.shape}")
                                
                                self.curr_mask = torch_prob_to_numpy_mask(self.curr_prob)
                                self.object_labels[next_obj_id] = class_name
                                self.save_labels()
                                
                                print(f"DEBUG: Objeto {next_obj_id} adicionado com sucesso")
                                print(f"DEBUG: num_objects atualizado para {self.num_objects}")
                                
                                stop_propagation = True
                                break
                # -------------------

                self.save_current_mask()
                self.show_current_frame(fast=True)
                self.update_memory_gauges()
                self.gui.process_events()

                if stop_propagation:
                    self.propagating = False
                    self.gui.text(f"Pausado para novo objeto {self.num_objects}.")
                    self.hit_number_key(self.num_objects)
                    print(f"DEBUG: Propagação pausada no frame {self.curr_ti}")
                    break

                if self.curr_ti == 0 or self.curr_ti == self.T - 1:
                    print(f"DEBUG: Fim do vídeo alcançado (curr_ti={self.curr_ti})")
                    break

            print(f"=== DEBUG: Fim on_propagate ===\n")
            
            # Atualiza o rastreamento para próxima propagação
            self._last_propagation_num_objects = self.num_objects
            
            self.propagating = False
            self.curr_frame_dirty = False
            self.on_pause()
            self.on_slider_update()
            self.gui.process_events()

    def is_box_empty_in_mask(self, bbox, mask_np, threshold=0.1):
        """
        Verifica se a região da BBox está 'vazia' na máscara atual.
        Retorna True se a interseção for desprezível (objeto novo).
        """
        x1, y1, x2, y2 = [int(c) for c in bbox]
        
        # Garante limites da imagem
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(self.w, x2); y2 = min(self.h, y2)
        
        if x2 <= x1 or y2 <= y1: return False

        # Recorta a região da máscara correspondente à caixa
        mask_crop = mask_np[y1:y2, x1:x2]
        
        # Conta pixels que NÃO são fundo ( > 0 )
        occupied_pixels = np.count_nonzero(mask_crop)
        total_pixels = (x2 - x1) * (y2 - y1)
        
        occupancy_rate = occupied_pixels / total_pixels
        
        # Se menos de 10% (threshold) da caixa estiver ocupada, consideramos "vazia"
        return occupancy_rate < threshold

    def pause_propagation(self):
        self.propagating = False

    def on_commit(self):
        if self.interacted_prob is None:
            self.load_current_image_mask()
        else:
            self.complete_interaction()
            self.update_interacted_mask()

        with autocast(self.device, enabled=(self.amp and self.device == 'cuda')):
            self.convert_current_image_mask_torch()
            
            # Limpeza rápida antes de salvar memória permanente para evitar lixo
            self.curr_mask = torch_prob_to_numpy_mask(self.curr_prob)
            self.curr_prob = index_numpy_to_one_hot_torch(self.curr_mask, self.num_objects + 1).to(
                self.device, non_blocking=True)

            self.gui.text(f'Permanent memory saved at {self.curr_ti}.')
            
            self.curr_prob = self.processor.step(self.curr_image_torch,
                                                 self.curr_prob[1:], # <--- [1:] AQUI TAMBÉM
                                                 idx_mask=False,
                                                 force_permanent=True)
            self.update_memory_gauges()
            self.update_gpu_gauges()

    def on_play_video_timer(self):
        self.curr_ti += 1
        if self.curr_ti > self.T - 1:
            self.curr_ti = 0
        self.gui.tl_slider.setValue(self.curr_ti)

    def on_export_visualization(self):
        # NOTE: Save visualization at the end of propagation
        image_folder = path.join(self.cfg['workspace'], 'visualization', self.vis_mode)
        save_folder = self.cfg['workspace']
        if path.exists(image_folder):
            # Sorted so frames will be in order
            output_path = path.join(save_folder, f'visualization_{self.vis_mode}.mp4')
            self.gui.text(f'Exporting visualization -- please wait')
            self.gui.process_events()
            convert_frames_to_video(image_folder,
                                    output_path,
                                    fps=self.output_fps,
                                    bitrate=self.output_bitrate,
                                    progress_callback=self.gui.progressbar_update)
            self.gui.text(f'Visualization exported to {output_path}')
            self.gui.progressbar_update(0)
        else:
            self.gui.text(f'No visualization images found in {image_folder}')

    def on_export_binary(self):
        # export masks in binary format for other applications, e.g., ProPainter
        mask_folder = path.join(self.cfg['workspace'], 'masks')
        save_folder = path.join(self.cfg['workspace'], 'binary_masks')
        if path.exists(mask_folder):
            os.makedirs(save_folder, exist_ok=True)
            self.gui.text(f'Exporting binary masks -- please wait')
            self.gui.process_events()
            convert_mask_to_binary(mask_folder,
                                   save_folder,
                                   self.vis_target_objects,
                                   progress_callback=self.gui.progressbar_update)
            self.gui.text(f'Binary masks exported to {save_folder}')
            self.gui.progressbar_update(0)
        else:
            self.gui.text(f'No masks found in {mask_folder}')

    def on_object_dial_change(self):
        object_id = self.gui.object_dial.value()
        self.hit_number_key(object_id)

    def on_fps_dial_change(self):
        self.output_fps = self.gui.fps_dial.value()

    def on_bitrate_dial_change(self):
        self.output_bitrate = self.gui.bitrate_dial.value()

    def on_add_object(self):
        self.reset_this_interaction() # Clear pending clicks

        if hasattr(self.click_ctrl, 'reset_context'):
            self.click_ctrl.reset_context()

        self.num_objects += 1
        self.cfg['num_objects'] = self.num_objects # Update config
        self.gui.object_dial.setMaximum(self.num_objects) # Update UI
        self.res_man.add_object_directory(self.num_objects) # Create dir

        # Resize probability tensor
        if self.curr_prob is not None:
            old_prob = self.curr_prob
            new_shape = (self.num_objects + 1, self.h, self.w)
            self.curr_prob = torch.zeros(new_shape, dtype=old_prob.dtype, device=self.device)
            self.curr_prob[:old_prob.shape[0]] = old_prob
        else:
            # It's None, it will be created with the correct size 
            # by convert_current_image_mask_torch when needed.
            pass

        self.gui.text(f"Added new object. Total objects: {self.num_objects}")
        self.hit_number_key(self.num_objects) # Switch to it

    def on_label_changed(self):
        new_label = self.gui.object_label_edit.text()
        if not new_label.strip():
            if self.curr_object in self.object_labels:
                del self.object_labels[self.curr_object]
                self.gui.text(f"Removed label for object {self.curr_object}")
        else:
            self.object_labels[self.curr_object] = new_label
            self.gui.text(f"Set label for object {self.curr_object} to: {new_label}")
        self.save_labels()

    def update_interacted_mask(self):
        self.curr_prob = self.interacted_prob
        self.curr_mask = torch_prob_to_numpy_mask(self.interacted_prob)
        self.save_current_mask()
        self.show_current_frame()
        self.curr_frame_dirty = False

    def reset_this_interaction(self):
        self.complete_interaction()
        self.interacted_prob = None
        if self.click_ctrl is not None:
            self.click_ctrl.unanchor()

    def on_reset_mask(self):
        self.curr_mask.fill(0)
        if self.curr_prob is not None:
            self.curr_prob.fill_(0)
        self.curr_frame_dirty = True
        self.save_current_mask()
        self.reset_this_interaction()
        self.show_current_frame()

    def on_reset_object(self):
        self.curr_mask[self.curr_mask == self.curr_object] = 0
        if self.curr_prob is not None:
            self.curr_prob[self.curr_object] = 0
        self.curr_frame_dirty = True
        self.save_current_mask()
        self.reset_this_interaction()
        self.show_current_frame()

    def on_remove_object_all_frames(self):
        if self.propagating:
            self.gui.text("Cannot remove object while propagating.")
            return

        object_id = self.curr_object
        if object_id == 0:
            self.gui.text("Cannot remove background (object 0).")
            return

        reply = QMessageBox.question(self.gui, 'Confirm Permanent Removal',
                                     f"Are you sure you want to permanently remove object {object_id} from all {self.length} frames?\n\nThis will delete all masks for this object from disk and cannot be undone.",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)

        if reply != QMessageBox.StandardButton.Yes:
            self.gui.text(f"Removal of object {object_id} cancelled.")
            return

        self.gui.text(f"Removing object {object_id} from all frames. Please wait...")
        self.gui.process_events()

        try:
            # 1. Preparar pasta de backup temporária
            backup_root = path.join(self.cfg['workspace'], 'undo_cache')
            os.makedirs(backup_root, exist_ok=True)
            
            # Limpa backup anterior (só suportamos 1 nível de undo para essa operação pesada por enquanto)
            if self.last_deleted_info is not None:
                shutil.rmtree(self.last_deleted_info['backup_dir'], ignore_errors=True)

            current_backup_dir = path.join(backup_root, f'obj_{object_id}')
            os.makedirs(current_backup_dir, exist_ok=True)
            os.makedirs(path.join(current_backup_dir, 'masks'), exist_ok=True)

            frames_affected = []

            # 2. Backup e Remoção do Soft Mask (Pasta)
            soft_mask_dir_obj = path.join(self.res_man.soft_mask_dir, f'{object_id}')
            soft_mask_backup_dest = path.join(current_backup_dir, 'soft_masks')
            
            has_soft_mask = path.exists(soft_mask_dir_obj)
            if has_soft_mask:
                # Movemos em vez de deletar (muito mais rápido e serve como backup)
                shutil.move(soft_mask_dir_obj, soft_mask_backup_dest)
                # Recriamos a pasta vazia para não quebrar o resto do código
                self.res_man.add_object_directory(object_id)

            # 3. Backup e Modificação das Máscaras (Hard Masks)
            for ti in range(self.length):
                mask = self.res_man.get_mask(ti)
                if mask is not None:
                    if np.any(mask == object_id):
                        # Salva cópia do original ANTES de alterar
                        frames_affected.append(ti)
                        mask_name = self.res_man.names[ti]
                        original_mask_path = path.join(self.res_man.mask_dir, mask_name + '.png')
                        backup_mask_path = path.join(current_backup_dir, 'masks', mask_name + '.png')
                        shutil.copy2(original_mask_path, backup_mask_path)

                        # Altera o atual (remove o objeto)
                        mask[mask == object_id] = 0
                        self.res_man.save_mask(ti, mask)
                
                if ti % 20 == 0:
                    self.gui.progressbar_update((ti + 1) / self.length)
                    self.gui.process_events()

            # 4. Salvar Metadados do Undo
            self.last_deleted_info = {
                'id': object_id,
                'label': self.object_labels.get(object_id, ""),
                'model': self.object_models.get(object_id, ""),
                'size': self.object_sizes.get(object_id, ""),
                'backup_dir': current_backup_dir,
                'has_soft_mask': has_soft_mask,
                'frames_affected': frames_affected
            }

            # 5. Remoção de Label e Atualização UI
            if object_id in self.object_labels:
                del self.object_labels[object_id]
                self.save_labels()
                if object_id == self.curr_object:
                    self.gui.update_object_label("")

            if object_id in self.object_models: 
                del self.object_models[object_id]
                self.save_models()

            if object_id in self.object_sizes:
                del self.object_sizes[object_id]
                self.save_sizes()

            self.gui.progressbar_update(0)
            self.gui.text(f"Object {object_id} removed. Press Ctrl+Z to Undo.")
            self.load_current_image_mask()
            self.show_current_frame()

        except Exception as e:
            self.gui.text(f"Error removing object: {e}")
            log.error(f"Failed to remove object {object_id}: {e}")
            self.gui.progressbar_update(0)
    
    def on_undo_delete(self):
        if self.last_deleted_info is None:
            self.gui.text("Nothing to undo.")
            return

        info = self.last_deleted_info
        obj_id = info['id']
        backup_dir = info['backup_dir']

        self.gui.text(f"Undoing deletion of Object {obj_id}...")
        self.gui.process_events()

        try:
            # 1. Restaurar Soft Masks
            if info['has_soft_mask']:
                # Remove a pasta vazia/nova que foi criada na deleção
                current_soft_dir = path.join(self.res_man.soft_mask_dir, f'{obj_id}')
                if path.exists(current_soft_dir):
                    shutil.rmtree(current_soft_dir)
                
                # Move de volta a pasta de backup
                backup_soft_dir = path.join(backup_dir, 'soft_masks')
                shutil.move(backup_soft_dir, current_soft_dir)

            # 2. Restaurar Hard Masks (apenas frames afetados)
            for ti in info['frames_affected']:
                mask_name = self.res_man.names[ti]
                backup_mask_path = path.join(backup_dir, 'masks', mask_name + '.png')
                target_mask_path = path.join(self.res_man.mask_dir, mask_name + '.png')
                
                if path.exists(backup_mask_path):
                    shutil.copy2(backup_mask_path, target_mask_path)
                    # Invalidar cache do ResourceManager para recarregar do disco
                    self.res_man.invalidate(ti)

            # 3. Restaurar Label
            if info['label']:
                self.object_labels[obj_id] = info['label']
                self.save_labels()

            if info.get('model'):
                self.object_models[obj_id] = info['model']
                self.save_models()

            if info.get('size'): # .get() para compatibilidade caso info antigo não tenha size
                self.object_sizes[obj_id] = info['size']
                self.save_sizes()

            # 4. Limpeza e UI
            # Se restauramos o objeto que estamos vendo agora, atualiza
            if self.curr_object == obj_id:
                self.gui.update_object_label(info['label'])
                self.gui.update_object_model(info.get('model', ""))
                self.gui.update_object_size(info.get('size', ""))
            
            # Garante que o seletor de objetos vá até esse ID (caso fosse o último)
            if obj_id > self.num_objects:
                self.num_objects = obj_id
                self.cfg['num_objects'] = self.num_objects
                self.gui.object_dial.setMaximum(self.num_objects)

            self.gui.text(f"Object {obj_id} restored successfully.")
            self.last_deleted_info = None # Limpa o stack de undo (só 1 nível)
            
            # Atualiza frame atual
            self.load_current_image_mask()
            self.show_current_frame()

        except Exception as e:
            self.gui.text(f"Error performing undo: {e}")
            log.error(f"Undo failed: {e}")

    def complete_interaction(self):
        if self.interaction is not None:
            self.interaction = None

    def on_prev_frame(self, step=1):
        new_ti = max(0, self.curr_ti - step)
        self.gui.tl_slider.setValue(new_ti)

    def on_next_frame(self, step=1):
        new_ti = min(self.curr_ti + step, self.length - 1)
        self.gui.tl_slider.setValue(new_ti)

    def update_gpu_gauges(self):
        if 'cuda' in self.device:
            info = torch.cuda.mem_get_info()
            global_free, global_total = info
            global_free /= (2**30)
            global_total /= (2**30)
            global_used = global_total - global_free

            self.gui.gpu_mem_gauge.setFormat(f'{global_used:.1f} GB / {global_total:.1f} GB')
            self.gui.gpu_mem_gauge.setValue(round(global_used / global_total * 100))

            used_by_torch = torch.cuda.max_memory_allocated() / (2**30)
            self.gui.torch_mem_gauge.setFormat(f'{used_by_torch:.1f} GB / {global_total:.1f} GB')
            self.gui.torch_mem_gauge.setValue(round(used_by_torch / global_total * 100 / 1024))
        elif 'mps' in self.device:
            mem_used = mps.current_allocated_memory() / (2**30)
            self.gui.gpu_mem_gauge.setFormat(f'{mem_used:.1f} GB')
            self.gui.gpu_mem_gauge.setValue(0)
            self.gui.torch_mem_gauge.setFormat('N/A')
            self.gui.torch_mem_gauge.setValue(0)
        else:
            self.gui.gpu_mem_gauge.setFormat('N/A')
            self.gui.gpu_mem_gauge.setValue(0)
            self.gui.torch_mem_gauge.setFormat('N/A')
            self.gui.torch_mem_gauge.setValue(0)

    def on_gpu_timer(self):
        self.update_gpu_gauges()

    def update_memory_gauges(self):
        try:
            curr_perm_tokens = self.processor.memory.work_mem.perm_size(0)
            self.gui.perm_mem_gauge.setFormat(f'{curr_perm_tokens} / {curr_perm_tokens}')
            self.gui.perm_mem_gauge.setValue(100)

            max_work_tokens = self.processor.memory.max_work_tokens
            max_long_tokens = self.processor.memory.max_long_tokens

            curr_work_tokens = self.processor.memory.work_mem.non_perm_size(0)
            curr_long_tokens = self.processor.memory.long_mem.non_perm_size(0)

            self.gui.work_mem_gauge.setFormat(f'{curr_work_tokens} / {max_work_tokens}')
            self.gui.work_mem_gauge.setValue(round(curr_work_tokens / max_work_tokens * 100))

            self.gui.long_mem_gauge.setFormat(f'{curr_long_tokens} / {max_long_tokens}')
            self.gui.long_mem_gauge.setValue(round(curr_long_tokens / max_long_tokens * 100))

        except AttributeError as e:
            self.gui.work_mem_gauge.setFormat('Unknown')
            self.gui.long_mem_gauge.setFormat('Unknown')
            self.gui.work_mem_gauge.setValue(0)
            self.gui.long_mem_gauge.setValue(0)

    def on_work_min_change(self):
        if self.initialized:
            self.gui.work_mem_min.setValue(
                min(self.gui.work_mem_min.value(),
                    self.gui.work_mem_max.value() - 1))
            self.update_config()

    def on_work_max_change(self):
        if self.initialized:
            self.gui.work_mem_max.setValue(
                max(self.gui.work_mem_max.value(),
                    self.gui.work_mem_min.value() + 1))
            self.update_config()

    def update_config(self):
        if self.initialized:
            with open_dict(self.cfg):
                self.cfg.long_term['min_mem_frames'] = self.gui.work_mem_min.value()
                self.cfg.long_term['max_mem_frames'] = self.gui.work_mem_max.value()
                self.cfg.long_term['max_num_tokens'] = self.gui.long_mem_max.value()
                self.cfg['mem_every'] = self.gui.mem_every_box.value()

            self.processor.update_config(self.cfg)

    def on_clear_memory(self):
        self.processor.clear_memory()
        if 'cuda' in self.device:
            torch.cuda.empty_cache()
        elif 'mps' in self.device:
            mps.empty_cache()
        self.processor.update_config(self.cfg)
        self.update_gpu_gauges()
        self.update_memory_gauges()

    def on_clear_non_permanent_memory(self):
        self.processor.clear_non_permanent_memory()
        if 'cuda' in self.device:
            torch.cuda.empty_cache()
        elif 'mps' in self.device:
            mps.empty_cache()
        self.processor.update_config(self.cfg)
        self.update_gpu_gauges()
        self.update_memory_gauges()

    def on_import_mask(self):
        file_name = self.gui.open_file('Mask')
        if len(file_name) == 0:
            return

        mask = self.res_man.import_mask(file_name, size=(self.h, self.w))

        shape_condition = ((len(mask.shape) == 2) and (mask.shape[-1] == self.w)
                           and (mask.shape[-2] == self.h))

        object_condition = (mask.max() <= self.num_objects)

        if not shape_condition:
            self.gui.text(f'Expected ({self.h}, {self.w}). Got {mask.shape} instead.')
        elif not object_condition:
            self.gui.text(f'Expected {self.num_objects} objects. Got {mask.max()} objects instead.')
        else:
            self.gui.text(f'Mask file {file_name} loaded.')
            self.curr_image_torch = self.curr_prob = None
            self.curr_mask = mask
            self.show_current_frame()
            self.save_current_mask()

        ''' 
        # --- INÍCIO DA CORREÇÃO ---
        max_id_in_mask = int(mask.max())
        if max_id_in_mask > self.num_objects:
            self.gui.text(f"Imported mask has {max_id_in_mask} objects. Updating app count from {self.num_objects}.")
            
            # Adiciona novos diretórios de objeto
            for new_id in range(self.num_objects + 1, max_id_in_mask + 1):
                self.res_man.add_object_directory(new_id)
            
            # Atualiza a contagem de objetos do app
            self.num_objects = max_id_in_mask
            self.cfg['num_objects'] = self.num_objects
            self.gui.object_dial.setMaximum(self.num_objects)
        # --- FIM DA CORREÇÃO ---

        # Agora que a contagem está correta, carregue a máscara
        self.gui.text(f'Mask file {file_name} loaded.')
        self.curr_image_torch = self.curr_prob = None
        self.curr_mask = mask
        self.show_current_frame()
        self.save_current_mask()
        '''

    def on_import_layer(self):
        file_name = self.gui.open_file('Layer')
        if len(file_name) == 0:
            return

        self._try_load_layer(file_name)

    def _try_load_layer(self, file_name):
        try:
            layer = self.res_man.import_layer(file_name, size=(self.h, self.w))

            self.gui.text(f'Layer file {file_name} loaded.')
            self.overlay_layer = layer
            self.overlay_layer_torch = torch.from_numpy(layer).float().to(self.device) / 255
            self.show_current_frame()
        except FileNotFoundError:
            self.gui.text(f'{file_name} not found.')

    def on_set_save_visualization_mode(self):
        self.save_visualization_mode = self.gui.save_visualization_combo.currentText()

    def on_save_soft_mask_toggle(self):
        self.save_soft_mask = self.gui.save_soft_mask_checkbox.isChecked()

    def on_mouse_motion_xy(self, x, y):
        self.last_ex = x
        self.last_ey = y

    @property
    def h(self) -> int:
        return self.res_man.h

    @property
    def w(self) -> int:
        return self.res_man.w

    @property
    def T(self) -> int:
        return self.res_man.T
