import torch
import numpy as np
from omegaconf import OmegaConf

import logging
log = logging.getLogger()

from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.modeling.sam2_base import SAM2Base
from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.memory_encoder import (
    MemoryEncoder, 
    MaskDownSampler, 
    Fuser,
    CXBlock
)
from sam2.modeling.position_encoding import PositionEmbeddingSine
from sam2.modeling.memory_attention import (
    MemoryAttention,
    MemoryAttentionLayer
)
from sam2.modeling.sam.transformer import RoPEAttention

class SAM2ClickController:
    """
    Um wrapper para o SAM2ImagePredictor que imita a interface 
    esperada pelo ClickInteraction do Cutie.
    
    Carrega o modelo MANUALMENTE para evitar conflitos com o Hydra.
    """
    
    def __init__(self, checkpoint_path: str, model_cfg_path: str, device: str = 'cuda'):
        log.info(f"Iniciando carregamento manual do SAM 2...")
        self.device = torch.device(device)
        self.autocast_ctx = torch.autocast(self.device.type, dtype=torch.bfloat16)

        try:
            log.info(f"Carregando config: {model_cfg_path}")
            cfg = OmegaConf.load(model_cfg_path)
            
            model_cfg = cfg.model
            
            log.info("Construindo sub-módulos...")
            img_enc_cfg = model_cfg.image_encoder.copy()
            img_enc_cfg.pop('_target_', None)

            trunk_cfg = img_enc_cfg.pop('trunk')
            trunk_cfg.pop('_target_', None)
            trunk_obj = Hiera(**trunk_cfg)
            
            neck_cfg = img_enc_cfg.pop('neck')
            neck_cfg.pop('_target_', None)
            
            neck_pe_cfg = neck_cfg.pop('position_encoding')
            neck_pe_cfg.pop('_target_', None)
            neck_pe_obj = PositionEmbeddingSine(**neck_pe_cfg)
            
            neck_obj = FpnNeck(position_encoding=neck_pe_obj, **neck_cfg)
            
            image_encoder_obj = ImageEncoder(trunk=trunk_obj, neck=neck_obj, **img_enc_cfg)

            mem_enc_cfg = model_cfg.memory_encoder.copy()
            mem_enc_cfg.pop('_target_', None)

            pe_cfg = mem_enc_cfg.pop('position_encoding')
            pe_cfg.pop('_target_', None) 
            position_encoding = PositionEmbeddingSine(**pe_cfg)

            md_cfg = mem_enc_cfg.pop('mask_downsampler')
            md_cfg.pop('_target_', None)
            mask_downsampler = MaskDownSampler(**md_cfg)
            
            fuser_cfg = mem_enc_cfg.pop('fuser')
            fuser_cfg.pop('_target_', None)
            fuser_layer_cfg = fuser_cfg.pop('layer')
            fuser_layer_cfg.pop('_target_', None)
            fuser_layer_cfg.pop('activation', None)
            fuser_layer_cfg.pop('dim_feedforward', None)
            fuser_layer_cfg.pop('dropout', None)
            fuser_layer_cfg.pop('pos_enc_at_attn', None)
            fuser_layer_cfg.pop('self_attention', None)
            fuser_layer_cfg.pop('d_model', None)
            
            cxblock_layer = CXBlock(**fuser_layer_cfg)
            fuser = Fuser(layer=cxblock_layer, **fuser_cfg)

            memory_encoder_obj = MemoryEncoder(
                mask_downsampler=mask_downsampler, 
                fuser=fuser, 
                position_encoding=position_encoding,
                **mem_enc_cfg
            )
            
            mem_attn_cfg = model_cfg.memory_attention.copy()
            mem_attn_cfg.pop('_target_', None)
            
            ma_layer_cfg = mem_attn_cfg.pop('layer')
            ma_layer_cfg.pop('_target_', None)
            
            self_attn_cfg = ma_layer_cfg.pop('self_attention')
            self_attn_cfg.pop('_target_', None)
            self_attn_obj = RoPEAttention(**self_attn_cfg)
            
            cross_attn_cfg = ma_layer_cfg.pop('cross_attention')
            cross_attn_cfg.pop('_target_', None)
            cross_attn_obj = RoPEAttention(**cross_attn_cfg)
            
            ma_layer_obj = MemoryAttentionLayer(
                self_attention=self_attn_obj,
                cross_attention=cross_attn_obj,
                **ma_layer_cfg
            )
            
            memory_attention_obj = MemoryAttention(
                layer=ma_layer_obj, 
                **mem_attn_cfg
            )

            model_cfg.pop('image_encoder', None)
            model_cfg.pop('memory_attention', None)
            model_cfg.pop('memory_encoder', None)
            model_cfg.pop('_target_', None)
            
            log.info("Instanciando SAM2Base...")
            sam2_model = SAM2Base(
                image_encoder=image_encoder_obj, 
                memory_attention=memory_attention_obj, 
                memory_encoder=memory_encoder_obj,
                **model_cfg
            )
            
            log.info(f"Carregando pesos: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            sam2_model.load_state_dict(ckpt["model"], strict=True)
            
            sam2_model.to(self.device).eval()
            self.predictor = SAM2ImagePredictor(sam2_model)
            
            log.info("Modelo SAM 2 carregado com sucesso (modo manual).")

        except FileNotFoundError as e:
            log.error(f"Erro: Arquivo não encontrado. Verifique os caminhos no seu .yaml.")
            log.error(f"  Caminho do Checkpoint: {checkpoint_path}")
            log.error(f"  Caminho da Config: {model_cfg_path}")
            raise e
        except Exception as e:
            log.error(f"Falha ao carregar modelo SAM 2 manualmente: {e}")
            raise

        self.image_set = False
        self.points = []
        self.labels = []

    def unanchor(self):
        """
        Chamado pelo Cutie quando trocamos de frame ou objeto.
        Reseta o estado da imagem e os cliques.
        """
        self.image_set = False
        self.points = []
        self.labels = []

    def interact(self, image: torch.Tensor, x: int, y: int, is_positive: bool,
                 prev_mask: torch.Tensor):
        """
        Função principal chamada pelo ClickInteraction.
        Recebe um clique e retorna a máscara prevista.
        """
        
        # O 'image' que recebemos é 4D: [1, C, H, W]
        # Removemos a dimensão do batch (índice 0)
        image_3d = image.squeeze(0)
        
        # 1. Configura a imagem no SAM 2 (apenas na primeira interação)
        if not self.image_set:
            image_np = (image_3d.permute(1, 2, 0) * 255).cpu().numpy().astype(np.uint8)
            
            with torch.inference_mode(), self.autocast_ctx:
                # set_image armazena o 'original_size' (ex: 1080x1920)
                # e o 'input_size' (ex: 1024x1024)
                self.predictor.set_image(image_np)
            self.image_set = True

        # 2. Adiciona o novo clique ao estado
        self.points.append([x, y])
        self.labels.append(1 if is_positive else 0)

        points_np = np.array(self.points)
        labels_np = np.array(self.labels)

        # --- INÍCIO DA CORREÇÃO ---
        
        # 3. Executa a predição (Manual, para obter logits)
        
        with torch.inference_mode(), self.autocast_ctx:
            
            # 3.A. Converte pontos para o formato do modelo
            point_coords_torch = torch.as_tensor(points_np, dtype=torch.float, device=self.predictor.device)
            point_labels_torch = torch.as_tensor(labels_np, dtype=torch.int, device=self.predictor.device)
            
            # Adiciona uma dimensão de "batch"
            point_coords_torch = point_coords_torch[None, :, :]
            point_labels_torch = point_labels_torch[None, :]

            # 3.B. Chama o previsor INTERNO (_predict)
            # A API é: (high_res_logits, iou_preds, low_res_logits)
            upscaled_logits_batch, iou_pred, low_res_logits = self.predictor._predict(
                point_coords=point_coords_torch,
                point_labels=point_labels_torch,
                multimask_output=False,
                return_logits=True  # Importante: Pede logits, não máscaras binárias
            )
            
            # 3.C. Remove a dimensão de "batch" e "multimask"
            # O formato final é [H_full, W_full] (ex: 1080, 1920)
            final_logits = upscaled_logits_batch.squeeze()

            # 3.D. Converte para o formato de probabilidade que o Cutie espera
            final_probs = torch.sigmoid(final_logits)
    
        return final_probs

    def undo(self):
        """
        Remove o último clique e refaz a predição.
        """
        if self.points:
            self.points.pop()
            self.labels.pop()

        if not self.points:
            return None 

        points_np = np.array(self.points)
        labels_np = np.array(self.labels)

        with torch.inference_mode(), self.autocast_ctx:
            masks, scores, logits = self.predictor.predict(
                points=points_np,
                point_labels=labels_np,
                multimask_output=False
            )
            
        best_logits = logits[0]
        best_logits_tensor = torch.from_numpy(best_logits).to(self.device)
        final_probs = torch.sigmoid(best_logits_tensor)
        
        return final_probs