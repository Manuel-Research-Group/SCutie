# Roadmap de melhorias — segmentação interativa (cliques)

Levantamento de gargalos de qualidade/performance no fluxo de marcação por
cliques (RITM/Cutie/SAM2) e plano de upgrades, com estimativas de tempo.
Itens 1-3 estão detalhados em
`docs/superpowers/specs/2026-06-11-click-segmentation-quickwins-design.md`.

## Gargalos de qualidade

| # | Gargalo | Onde | Impacto |
|---|---------|------|---------|
| Q1 | Deriva do `_object_roi` no `ZoomIn` — bbox calculado sobre toda a máscara prevista, então um pixel "fantasma" longe do objeto faz o crop explodir e nunca mais voltar ao normal naquela interação | `gui/ritm/inference/transforms/zoom_in.py` | Causa raiz da fragmentação/máscaras "jogadas longe" |
| Q2 | RITM (HRNet/DeepLab + f-BRS, ~2021) é uma arquitetura datada; pior generalização para objetos pequenos/alongados (válvulas, tubos) e degrada mais com cliques distantes | `ClickController` | Qualidade de máscara inferior ao SAM2 em geral |
| Q3 | `prev_mask` "sujo" do Cutie alimentado como canal extra no 1º clique — mapa de prob. raramente é binário limpo, então o RITM já começa com ruído | `interaction.py` → `predictors/base.py` | Alimenta Q1 |
| Q4 | `SAM2ClickController.undo()` inconsistente — `interact()` usa `_predict()` com logits crus (sem zoom-in), mas `undo()` usa `predictor.predict()` (pipeline diferente) | `sam2_click_controller.py:224-249` | Inconsistência visual após undo |
| Q5 | Sem continuidade de prompt no SAM2 — cada clique reprocessa do zero os pontos acumulados, sem usar `mask_input` (máscara anterior como prompt) | `sam2_click_controller.py:189-222` | Máscara pode "pular" entre cliques mesmo no SAM2 |

## Gargalos de performance

| # | Gargalo | Custo estimado |
|---|---------|-----------------|
| P1 | f-BRS-B com L-BFGS (scipy, CPU) — a cada clique após o 1º, otimização iterativa fora do grafo GPU | ~150-400ms/clique extra, sobe com imagens grandes |
| P2 | `with_flip=True` — dobra o forward pass (imagem + flip horizontal) | ~2x tempo de inferência por clique |
| P3 | SAM2 com `bfloat16` autocast + imagem 1024x1024 — mais pesado por clique que RITM puro, mas recodifica a imagem só 1x (`set_image`) | ~50-150ms/clique em GPU média |
| P4 | `_apply_polyline_barrier` + loop "proteger outros objetos" — operações booleanas full-frame por objeto, a cada clique | Cresce linear com nº de objetos; desprezível com poucos objetos |
| P5 | Redesenho síncrono da GUI (`update_interacted_mask`, `update_gpu_gauges`) a cada clique | ~10-50ms, perceptível em sequência rápida de cliques |

## Roadmap de upgrades

### 🟢 Rápidos (horas, baixo risco)

1. **Guard-rail no ZoomIn** (fix da fragmentação/Q1) — descartar atualização
   de `_object_roi` quando o bbox "explode" (>70% da imagem e >3x o ROI
   anterior), com log de aviso.
   - **Estimativa: ~2-4h** (implementação + teste)
   - *Em andamento — spec escrito.*

2. **Desativar `with_flip`** no `ClickController` (RITM) → ~2x mais rápido
   por clique, perda de qualidade mínima.
   - **Estimativa: ~30min** (parte do item 3 abaixo, via toggle)

3. **`brs_mode: 'f-BRS-B'` → `'NoBRS'`** como "modo rápido" opcional, com
   toggle na GUI.
   - **Estimativa: ~1-2h** (lógica em `ClickController` + checkbox na GUI)
   - *Itens 2 e 3 combinados no spec como "toggle modo rápido".*

### 🟡 Médios (1-2 dias)

4. **Migrar o controlador de clique padrão de RITM → SAM2**
   (`sam2_click_controller.py` já existe, mas não é o default). Requer:
   - Ativar `is_new_interaction`/`unanchor()` corretamente (já existe)
   - Corrigir `undo()` para usar o mesmo caminho `_predict()`/logits do
     `interact()` (Q4)
   - Validar consumo de VRAM com múltiplos objetos simultâneos
   - **Estimativa: ~1 dia** implementação + **~0.5 dia** testes em vídeos

5. **Limpar o `prev_mask`** antes de passar ao segmentador — binarizar/
   suavizar (morphological open) o canal de prob. do Cutie antes de
   concatenar como prior, reduzindo ruído que alimenta Q1/Q3.
   - **Estimativa: ~3-4h**

6. **Usar `mask_input` do SAM2** entre cliques consecutivos do mesmo objeto
   (refinamento iterativo nativo), em vez de re-passar todos os pontos
   acumulados.
   - **Estimativa: ~1 dia** (mudança de API + testes de regressão de
     qualidade)

### 🔴 Maiores (3-5+ dias)

7. **Pipeline de propagação assistida por SAM2** — usar a saída do SAM2
   (alta qualidade) como máscara semente sempre que o Cutie reiniciar
   memória (frame 0 / novo objeto), reduzindo erro acumulado na propagação.
   - **Estimativa: ~2-3 dias**

8. **Throttling/Debounce de redesenho da GUI** — agrupar redraws durante
   sequência rápida de cliques (ex.: só redesenhar após 100ms de inatividade
   do mouse), mantendo a predição síncrona mas a UI assíncrona.
   - **Estimativa: ~1 dia** (requer cuidado com Qt threads)

9. **Cache de embeddings de imagem do SAM2 entre frames já visitados** (se o
   usuário volta a clicar em frames já processados durante revisão), evitando
   reencodar.
   - **Estimativa: ~1-2 dias** (gestão de memória/LRU cache de embeddings)

## Ordem de execução recomendada

1. Item 1 (fix do zoom-in) → ataca a causa raiz, sem custo de risco.
2. Itens 2+3 (toggle modo rápido) → reduz latência, risco zero.
3. Itens 4+6 (migrar para SAM2 com refinamento iterativo) → maior ganho de
   qualidade, mas exige mais validação visual.
