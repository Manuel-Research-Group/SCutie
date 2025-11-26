import requests
import torch
import numpy as np
import io
import base64
import json
from PIL import Image

class RemoteSAMController:
    def __init__(self, api_url: str, device: str = 'cuda'):
        self.api_url = api_url
        self.device = device
        
        # --- CONTEXTO PERSISTENTE ---
        # Estes valores devem sobreviver ao 'unanchor' normal para permitir edição contínua
        self.points = []       
        self.labels = []       
        self.current_bbox = None 
        self.current_frame_idx = -1 
        self.current_obj_id = -1
        
    def unanchor(self):
        """
        Chamado pela GUI antes de iniciar uma nova sequência de cliques no MESMO objeto.
        Nós limpamos apenas os pontos recentes, mas MANTEMOS a BBox e o Frame,
        pois o usuário provavelmente está refinando o objeto que o YOLO acabou de achar.
        """
        self.points = []
        self.labels = []
        # IMPORTANTE: NÃO limpamos current_bbox, current_frame_idx ou current_obj_id aqui.

    def reset_context(self):
        """
        Função nova para limpar TUDO (chamada quando trocamos de objeto explicitamente).
        """
        self.unanchor()
        self.current_bbox = None
        self.current_frame_idx = -1
        self.current_obj_id = -1

    def _send_request(self, image_np: np.ndarray, payload: dict) -> torch.Tensor:
        # Converter imagem para JPG em memória
        pil_img = Image.fromarray(image_np)
        img_byte_arr = io.BytesIO()
        pil_img.save(img_byte_arr, format='JPEG', quality=95)
        img_byte_arr.seek(0)

        files = {'file': ('image.jpg', img_byte_arr, 'image/jpeg')}
        
        try:
            response = requests.post(f"{self.api_url}/predict", files=files, data=payload, timeout=5)
            
            if response.status_code != 200:
                print(f"Erro API SAM2: {response.text}")
                return None

            resp_json = response.json()
            mask_b64 = resp_json['mask_base64']
            mask_data = base64.b64decode(mask_b64)
            mask_img = Image.open(io.BytesIO(mask_data))
            
            mask_np = np.array(mask_img).astype(np.float32) / 255.0
            mask_tensor = torch.from_numpy(mask_np).to(self.device)
            return mask_tensor

        except Exception as e:
            print(f"Falha na conexão com SAM2: {e}")
            return None

    def predict_bbox(self, image_np: np.ndarray, bbox: list, frame_idx: int = -1, obj_id: int = -1) -> torch.Tensor:
        """
        Define a âncora inicial (geralmente via YOLO).
        """
        # Limpa tudo, pois é um novo começo definido pelo YOLO
        self.reset_context()
        
        # Salva o contexto novo
        self.current_bbox = bbox
        self.current_frame_idx = frame_idx
        self.current_obj_id = obj_id
        
        payload = {
            'bbox': json.dumps(bbox),
            'frame_idx': str(frame_idx),
            'obj_id': str(obj_id)
        }
        return self._send_request(image_np, payload)

    def interact(self, image: torch.Tensor, x: int, y: int, is_positive: bool, prev_mask: torch.Tensor = None):
        """
        Refinamento (Human-in-the-Loop).
        Agora envia os pontos + a bbox original para manter o SAM focado.
        """
        # --- CORREÇÃO DO ERRO DE DIMENSÃO (RuntimeError) ---
        if image.dim() == 4:
            image = image.squeeze(0)
        # ---------------------------------------------------

        image_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        
        self.points.append([int(x), int(y)])
        self.labels.append(1 if is_positive else 0)
        
        payload = {
            'points': json.dumps(self.points),
            'labels': json.dumps(self.labels),
            'bbox': json.dumps(self.current_bbox) if self.current_bbox else "", 
            'frame_idx': str(self.current_frame_idx),
            'obj_id': str(self.current_obj_id)
        }
        
        return self._send_request(image_np, payload)

    def undo(self):
        if self.points:
            self.points.pop()
            self.labels.pop()
            return None
        return None