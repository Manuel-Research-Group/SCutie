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
        self.last_mask = None

    def unanchor(self):
        # API é stateless, não precisa resetar nada
        pass

    def predict_bbox(self, image_np: np.ndarray, bbox: list, frame_idx: int = -1, obj_id: int = -1) -> torch.Tensor:
        """
        Envia imagem e bbox para a API.
        bbox format: [x_min, y_min, x_max, y_max]
        """
        # 1. Converter imagem numpy para bytes (JPG/PNG) para envio
        pil_img = Image.fromarray(image_np)
        img_byte_arr = io.BytesIO()
        pil_img.save(img_byte_arr, format='JPEG', quality=95)
        img_byte_arr.seek(0)

        # 2. Preparar payload
        files = {'file': ('image.jpg', img_byte_arr, 'image/jpeg')}
        data = {
            'bbox': json.dumps(bbox),
            'frame_idx': str(frame_idx), # Enviar como string
            'obj_id': str(obj_id)        # Enviar como string
        }

        try:
            # 3. Request POST
            response = requests.post(f"{self.api_url}/predict", files=files, data=data, timeout=10)
            
            if response.status_code != 200:
                print(f"Erro API SAM2: {response.text}")
                return None

            resp_json = response.json()
            
            # 4. Decodificar Base64 para Máscara
            mask_b64 = resp_json['mask_base64']
            mask_data = base64.b64decode(mask_b64)
            mask_img = Image.open(io.BytesIO(mask_data))
            
            # Converter para Tensor torch (como o RITM retornaria)
            # O RITM retorna probabilidade, mas o SAM retorna binário (0 ou 255).
            # Vamos normalizar para 0.0 - 1.0 float
            mask_np = np.array(mask_img).astype(np.float32) / 255.0
            mask_tensor = torch.from_numpy(mask_np).to(self.device)
            
            return mask_tensor

        except Exception as e:
            print(f"Falha na conexão com SAM2: {e}")
            return None

    def interact(self, image: torch.Tensor, x: int, y: int, is_positive: bool, prev_mask: torch.Tensor = None):
        """
        Adaptador para funcionar com cliques normais.
        Como a API espera Bbox, criamos uma pequena bbox ao redor do clique.
        """
        # Converter Tensor (0-1 float) de volta para Numpy (0-255 uint8)
        image_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        
        # Criar uma "caixa falsa" pequena ao redor do clique
        margin = 10
        h, w = image_np.shape[:2]
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(w, x + margin)
        y2 = min(h, y + margin)
        
        bbox = [x1, y1, x2, y2]
        
        return self.predict_bbox(image_np, bbox)

    def undo(self):
        # A API não tem histórico, undo local teria que ser gerenciado pelo MainController
        return None