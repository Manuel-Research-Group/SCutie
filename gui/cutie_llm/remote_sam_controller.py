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
        
        # --- PERSISTENT CONTEXT ---
        self.points = []       
        self.labels = []       
        self.current_bbox = None 
        self.current_frame_idx = -1 
        self.current_obj_id = -1
        
    def unanchor(self):
        """
        Reseta apenas os pontos da interação atual, mantendo o contexto do objeto (BBox/Frame).
        """
        self.points = []
        self.labels = []

    def reset_context(self):
        """
        Limpa TUDO. Chamado ao trocar de objeto.
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
                print(f"SAM2 API Error: {response.text}")
                return None

            resp_json = response.json()
            mask_b64 = resp_json['mask_base64']
            mask_data = base64.b64decode(mask_b64)
            mask_img = Image.open(io.BytesIO(mask_data))
            
            mask_np = np.array(mask_img).astype(np.float32) / 255.0
            mask_tensor = torch.from_numpy(mask_np).to(self.device)
            return mask_tensor

        except Exception as e:
            print(f"SAM2 Connection Failed: {e}")
            return None

    def predict_bbox(self, image_np: np.ndarray, bbox: list, frame_idx: int = -1, obj_id: int = -1) -> torch.Tensor:
        """
        Define a âncora inicial (YOLO).
        """
        self.reset_context()
        
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
        Refinamento com Lógica Inteligente:
        Se o usuário clicar FORA da caixa original, paramos de enviar a caixa
        para permitir que o SAM expanda a máscara livremente.
        """
        # Correção de dimensão
        if image.dim() == 4:
            image = image.squeeze(0)

        image_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        
        # Adiciona o novo clique
        self.points.append([int(x), int(y)])
        self.labels.append(1 if is_positive else 0)
        
        # --- LÓGICA INTELIGENTE (O que faltava no seu código) ---
        send_bbox = True
        
        if self.current_bbox:
            x1, y1, x2, y2 = self.current_bbox
            
            # Verifica TODOS os pontos positivos dados até agora
            for pt, lbl in zip(self.points, self.labels):
                if lbl == 1: # Se for clique de "Adicionar"
                    px, py = pt
                    # Se o ponto estiver FORA da caixa...
                    if not (x1 <= px <= x2 and y1 <= py <= y2):
                        print(f"DEBUG: Clique ({px},{py}) fora da BBox. Soltando a restrição!")
                        send_bbox = False
                        break
        
        # Se já tivermos muitos cliques, também soltamos a caixa por segurança
        if len(self.points) > 3:
            send_bbox = False
        # --------------------------------------------------------

        payload = {
            'points': json.dumps(self.points),
            'labels': json.dumps(self.labels),
            # AQUI ESTÁ A MÁGICA: Só envia a bbox se 'send_bbox' ainda for True
            'bbox': json.dumps(self.current_bbox) if (self.current_bbox and send_bbox) else "", 
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