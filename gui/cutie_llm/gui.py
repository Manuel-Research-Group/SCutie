import functools
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from PySide6.QtWidgets import (QWidget, QComboBox, QCheckBox, QHBoxLayout, QLabel, QPushButton,
                               QTextEdit, QSpinBox, QPlainTextEdit, QVBoxLayout, QSizePolicy,
                               QButtonGroup, QSlider, QRadioButton, QApplication, QFileDialog, 
                               QLineEdit, QMenuBar, QMenu, QToolTip, QRubberBand, QGridLayout)
from PySide6.QtGui import (QKeySequence, QShortcut, QTextCursor, QImage, QPixmap, QIcon, QAction, QActionGroup)
from PySide6.QtCore import Qt, QTimer, QRect, QPoint, QSize
from PySide6.QtGui import QPainter, QPen, QColor

from cutie.utils.palette import davis_palette_np
from gui.gui_utils import *

class OverlayCanvas(QLabel):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        # Cores para as sugestões (Amarelo vibrante por padrão)
        self.pen_candidate = QPen(QColor(255, 255, 0), 2, Qt.PenStyle.DashLine)
        self.pen_hover = QPen(QColor(0, 255, 0), 2, Qt.PenStyle.SolidLine)
        self.hovered_box_idx = -1

    def paintEvent(self, event):
        # 1. Desenha a imagem (Pixmap) original
        super().paintEvent(event)

        # 2. Desenha as caixas do YOLO se houverem
        if not self.controller.show_yolo_suggestions:
            return

        candidates = self.controller.get_current_yolo_candidates()
        if not candidates:
            return

        painter = QPainter(self)
        
        for i, box in enumerate(candidates):
            x1, y1, x2, y2 = box['bbox_xyxy']
            
            # Converter coordenadas da imagem real para coordenadas da tela (zoom/scale)
            sx1, sy1 = self.controller.gui.image_pos_to_pixel_pos(x1, y1)
            sx2, sy2 = self.controller.gui.image_pos_to_pixel_pos(x2, y2)
            
            w = sx2 - sx1
            h = sy2 - sy1
            
            if i == self.hovered_box_idx:
                painter.setPen(self.pen_hover)
            else:
                painter.setPen(self.pen_candidate)
                
            painter.drawRect(int(sx1), int(sy1), int(w), int(h))
            
            # Opcional: Desenhar o nome da classe
            painter.drawText(int(sx1), int(sy1) - 5, f"{box['class_name']} ({box['confidence']:.2f})")
            
        painter.end()

    def mouseMoveEvent(self, event):
        # Detectar hover para destacar a caixa
        if self.controller.show_yolo_suggestions:
            mx, my = self.controller.gui.pixel_pos_to_image_pos(event.position().x(), event.position().y())
            self.hovered_box_idx = self.controller.get_box_index_at(mx, my)
            self.update() # Força repaint
        
        # Passa o evento para o pai (GUI) para lidar com a lógica normal
        super().mouseMoveEvent(event)

class GUI(QWidget):

    def __init__(self, controller, cfg: DictConfig) -> None:
        super().__init__()

        # callbacks to be set by the controller
        self.on_mouse_motion_xy = None
        self.click_fn = None

        self.controller = controller
        self.cfg = cfg
        self.h = controller.h
        self.w = controller.w
        self.T = controller.T

        # set up the window
        self.setWindowTitle(f'Cutie demo: {cfg["workspace"]}')
        self.setGeometry(100, 100, self.w + 200, self.h + 200)
        self.setWindowIcon(QIcon('docs/icon.png'))

        self.menu_bar = QMenuBar(self)
        
        # Menu de Modos
        self.mode_menu = self.menu_bar.addMenu("Mode")
        self.mode_action_group = QActionGroup(self)
        self.mode_action_group.setExclusive(True)

        # Ação: Anotação
        self.act_annotation = QAction("Annotation", self)
        self.act_annotation.setCheckable(True)
        self.act_annotation.setChecked(True) # Padrão
        self.act_annotation.setShortcut("Ctrl+1")
        self.act_annotation.triggered.connect(lambda: controller.set_app_mode('annotation'))
        self.mode_action_group.addAction(self.act_annotation)
        self.mode_menu.addAction(self.act_annotation)

        # Ação: Visualização
        self.act_view = QAction("Visualization", self)
        self.act_view.setCheckable(True)
        self.act_view.setShortcut("Ctrl+2")
        self.act_view.triggered.connect(lambda: controller.set_app_mode('view'))
        self.mode_action_group.addAction(self.act_view)
        self.mode_menu.addAction(self.act_view)

        self.selection_menu = self.menu_bar.addMenu("Seleção")
        self.selection_action_group = QActionGroup(self)
        self.selection_action_group.setExclusive(True)

        # Ferramenta: Clique (Padrão)
        self.act_sel_click = QAction("Clique Pontual", self)
        self.act_sel_click.setCheckable(True)
        self.act_sel_click.setChecked(True)
        self.act_sel_click.setShortcut("Q") # Atalho sugerido
        self.act_sel_click.triggered.connect(lambda: controller.set_selection_tool('click'))
        self.selection_action_group.addAction(self.act_sel_click)
        self.selection_menu.addAction(self.act_sel_click)

        # Ferramenta: Bounding Box
        self.act_sel_bbox = QAction("Bounding Box (Retângulo)", self)
        self.act_sel_bbox.setCheckable(True)
        self.act_sel_bbox.setShortcut("W") # Atalho sugerido
        self.act_sel_bbox.triggered.connect(lambda: controller.set_selection_tool('bbox'))
        self.selection_action_group.addAction(self.act_sel_bbox)
        self.selection_menu.addAction(self.act_sel_bbox)

        # Menu YOLO
        self.menu_yolo = self.menu_bar.addMenu("YOLO")
        
        self.act_load_yolo = QAction("Load YOLO JSON...", self)
        self.act_load_yolo.triggered.connect(controller.on_load_yolo_json)
        self.menu_yolo.addAction(self.act_load_yolo)
        
        self.act_toggle_yolo = QAction("Show Suggestions", self)
        self.act_toggle_yolo.setCheckable(True)
        self.act_toggle_yolo.setChecked(True)
        self.act_toggle_yolo.toggled.connect(controller.on_toggle_yolo)
        self.menu_yolo.addAction(self.act_toggle_yolo)

        # set up some buttons
        self.play_button = QPushButton('Play video')
        self.play_button.clicked.connect(self.on_play_video)

        self.undo_button = QPushButton("Undo (Ctrl+Z)")
        self.undo_button.clicked.connect(controller.on_undo_delete)
        
        self.commit_button = QPushButton('Commit to permanent memory')
        self.commit_button.clicked.connect(controller.on_commit)

        self.init_frame_button = QPushButton('Auto-Init Frame (YOLO)')
        self.init_frame_button.clicked.connect(controller.on_init_frame_from_yolo)
        self.init_frame_button.setToolTip("Gera máscaras automaticamente para todas as detecções YOLO neste frame")

        self.export_video_button = QPushButton('Export as video')
        self.export_video_button.clicked.connect(controller.on_export_visualization)
        self.export_binary_button = QPushButton('Export binary masks')
        self.export_binary_button.clicked.connect(controller.on_export_binary)

        self.forward_run_button = QPushButton('Propagate forward')
        self.forward_run_button.clicked.connect(controller.on_forward_propagation)
        #self.forward_run_button.setMinimumWidth(150)

        self.backward_run_button = QPushButton('Propagate backward')
        self.backward_run_button.clicked.connect(controller.on_backward_propagation)
        #self.backward_run_button.setMinimumWidth(150)

        self.menu_model = self.menu_bar.addMenu("Model AI")
        self.model_action_group = QActionGroup(self)
        self.model_action_group.setExclusive(True)

        # universal progressbar
        self.progressbar = QProgressBar()
        self.progressbar.setMinimum(0)
        self.progressbar.setMaximum(100)
        self.progressbar.setValue(0)
        self.progressbar.setMinimumWidth(300)

        # Opção RITM
        self.act_ritm = QAction("RITM (Local - Rápido)", self)
        self.act_ritm.setCheckable(True)
        self.act_ritm.setChecked(True)
        self.act_ritm.triggered.connect(lambda: controller.set_segmentation_model('RITM'))
        self.model_action_group.addAction(self.act_ritm)
        self.menu_model.addAction(self.act_ritm)

        # Opção SAM 2
        self.act_sam2 = QAction("SAM 2 (API - Preciso)", self)
        self.act_sam2.setCheckable(True)
        self.act_sam2.triggered.connect(lambda: controller.set_segmentation_model('SAM2'))
        self.model_action_group.addAction(self.act_sam2)
        self.menu_model.addAction(self.act_sam2)

        self.reset_frame_button = QPushButton('Reset frame')
        self.reset_frame_button.clicked.connect(controller.on_reset_mask)
        self.reset_object_button = QPushButton('Reset object')
        self.reset_object_button.clicked.connect(controller.on_reset_object)

        self.remove_object_button = QPushButton('Remove object (all frames)')
        self.remove_object_button.clicked.connect(controller.on_remove_object_all_frames)

        # set up the LCD
        self.lcd = QLineEdit()
        self.lcd.setMaximumHeight(28)
        self.lcd.setMaximumWidth(150)
        self.lcd.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.lcd.setText('{: 5d} / {: 5d}'.format(0, controller.T - 1))
        # Conecta o "Enter" para a nova função no controller
        self.lcd.returnPressed.connect(controller.on_jump_to_frame)

        # current object id
        self.object_dial = QSpinBox()
        self.object_dial.setReadOnly(False)
        self.object_dial.setMinimumSize(50, 30)
        self.object_dial.setMinimum(1)
        self.object_dial.setMaximum(controller.num_objects)
        self.object_dial.editingFinished.connect(controller.on_object_dial_change)

        self.add_object_button = QPushButton("New Object (+)")
        self.add_object_button.clicked.connect(controller.on_add_object)

        self.object_color = QLabel()
        self.object_color.setMinimumSize(100, 30)
        self.object_color.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.frame_name = QLabel()
        self.object_label_edit = QLineEdit()
        self.object_label_edit.setPlaceholderText("Label for object...")
        self.object_label_edit.editingFinished.connect(controller.on_label_changed)
        self.frame_name.setMinimumSize(100, 30)
        self.frame_name.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.object_model_edit = QLineEdit()
        self.object_model_edit.setPlaceholderText("Model (VPI)...")
        self.object_model_edit.setMinimumWidth(120)
        self.object_model_edit.editingFinished.connect(controller.on_model_changed)

        self.object_size_edit = QLineEdit()
        self.object_size_edit.setPlaceholderText("Size (DN)...")
        self.object_size_edit.setMaximumWidth(100) 
        self.object_size_edit.editingFinished.connect(controller.on_size_changed)

        self.reference_checkbox = QCheckBox("Ref")
        self.reference_checkbox.stateChanged.connect(controller.on_reference_changed)

        self.inverted_checkbox = QCheckBox("Inverted")
        self.inverted_checkbox.stateChanged.connect(controller.on_inverted_changed)

        # timeline slider
        self.tl_slider = QSlider(Qt.Orientation.Horizontal)
        self.tl_slider.valueChanged.connect(controller.on_slider_update)
        self.tl_slider.setMinimum(0)
        self.tl_slider.setMaximum(controller.T - 1)
        self.tl_slider.setValue(0)
        self.tl_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.tl_slider.setTickInterval(1)

        # combobox
        self.combo = QComboBox(self)
        self.combo.addItem("mask")
        self.combo.addItem("davis")
        self.combo.addItem("fade")
        self.combo.addItem("light")
        self.combo.addItem("popup")
        self.combo.addItem("layer")
        self.combo.addItem("rgba")
        self.combo.setCurrentText('davis')
        self.combo.currentTextChanged.connect(controller.set_vis_mode)

        self.save_visualization_combo = QComboBox(self)
        self.save_visualization_combo.addItem("None")
        self.save_visualization_combo.addItem("Always")
        self.save_visualization_combo.addItem("Propagation only (higher quality)")
        self.combo.setCurrentText('None')
        self.save_visualization_combo.currentTextChanged.connect(
            controller.on_set_save_visualization_mode)

        self.save_soft_mask_checkbox = QCheckBox(self)
        self.save_soft_mask_checkbox.toggled.connect(controller.on_save_soft_mask_toggle)
        self.save_soft_mask_checkbox.setChecked(False)

        # controls for output FPS and bitrate
        self.fps_dial = QSpinBox()
        self.fps_dial.setReadOnly(False)
        self.fps_dial.setMinimumSize(40, 30)
        self.fps_dial.setMinimum(1)
        self.fps_dial.setMaximum(60)
        self.fps_dial.setValue(cfg['output_fps'])
        self.fps_dial.editingFinished.connect(controller.on_fps_dial_change)

        self.bitrate_dial = QSpinBox()
        self.bitrate_dial.setReadOnly(False)
        self.bitrate_dial.setMinimumSize(40, 30)
        self.bitrate_dial.setMinimum(1)
        self.bitrate_dial.setMaximum(100)
        self.bitrate_dial.setValue(cfg['output_bitrate'])
        self.bitrate_dial.editingFinished.connect(controller.on_bitrate_dial_change)

        # Main canvas -> QLabel
        #self.main_canvas = QLabel()
        self.main_canvas = OverlayCanvas(controller)
        self.main_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.main_canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_canvas.setMinimumSize(100, 100)

        self.main_canvas.mousePressEvent = self.on_mouse_press
        self.main_canvas.mouseMoveEvent = self.on_mouse_motion
        self.main_canvas.setMouseTracking(True)  # Required for all-time tracking
        self.main_canvas.mouseReleaseEvent = self.on_mouse_release

        self.rubber_band = QRubberBand(QRubberBand.Rectangle, self.main_canvas)
        self.origin_mouse_pos = QPoint()

        # clearing memory
        self.clear_all_mem_button = QPushButton('Reset all memory')
        self.clear_all_mem_button.clicked.connect(controller.on_clear_memory)
        self.clear_non_perm_mem_button = QPushButton('Reset non-permanent memory')
        self.clear_non_perm_mem_button.clicked.connect(controller.on_clear_non_permanent_memory)

        # displaying memory usage
        self.perm_mem_gauge, self.perm_mem_gauge_layout = create_gauge('Permanent memory size')
        self.work_mem_gauge, self.work_mem_gauge_layout = create_gauge('Working memory size')
        self.long_mem_gauge, self.long_mem_gauge_layout = create_gauge('Long-term memory size')
        self.gpu_mem_gauge, self.gpu_mem_gauge_layout = create_gauge(
            'GPU mem. (all proc, w/ caching)')
        self.torch_mem_gauge, self.torch_mem_gauge_layout = create_gauge(
            'GPU mem. (torch, w/o caching)')

        # Parameters setting
        self.work_mem_min, self.work_mem_min_layout = create_parameter_box(
            1, 100, 'Min. working memory frames', callback=controller.on_work_min_change)
        self.work_mem_max, self.work_mem_max_layout = create_parameter_box(
            2, 100, 'Max. working memory frames', callback=controller.on_work_max_change)
        self.long_mem_max, self.long_mem_max_layout = create_parameter_box(
            1000,
            100000,
            'Max. long-term memory size',
            step=1000,
            callback=controller.update_config)
        self.mem_every_box, self.mem_every_box_layout = create_parameter_box(
            1, 100, 'Memory frame every (r)', callback=controller.update_config)

        # import mask/layer
        self.import_mask_button = QPushButton('Import mask')
        self.import_mask_button.clicked.connect(controller.on_import_mask)
        self.import_layer_button = QPushButton('Import layer')
        self.import_layer_button.clicked.connect(controller.on_import_layer)

        # Console on the GUI
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMinimumHeight(100)
        self.console.setMaximumHeight(100)

        # Tips for the users
        self.tips = QTextEdit()
        self.tips.setReadOnly(True)
        self.tips.setTextInteractionFlags(Qt.NoTextInteraction)
        self.tips.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        with open(Path(__file__).parent / 'TIPS.md', 'r') as f:
            self.tips.setMarkdown(f.read())

        navi = QHBoxLayout()

        apply_fixed_size_policy = lambda x: x.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed) if x else None

        # ---------------------------------------------------------
        # BLOCO 1 (ESQUERDA): Interação e Propriedades do Objeto
        # ---------------------------------------------------------
        # ---------------------------------------------------------
        # BLOCO 1 (ESQUERDA): Controle de Vídeo e Propriedades do Objeto
        # ---------------------------------------------------------
        interact_subbox = QVBoxLayout()
        interact_subbox.setSpacing(8) # Espaço entre as linhas verticais

        # Linha 1: Controles de Navegação (Timeline/Play/Undo/Resets)
        video_nav_layout = QHBoxLayout()
        video_nav_layout.addWidget(self.lcd)
        video_nav_layout.addWidget(self.play_button)
        video_nav_layout.addWidget(self.undo_button) # Undo movido para cá para poupar espaço vertical
        video_nav_layout.addSpacing(10)
        video_nav_layout.addWidget(self.reset_frame_button)
        video_nav_layout.addWidget(self.reset_object_button)
        video_nav_layout.addStretch(1) # Mantém tudo à esquerda
        
        # Linha 2 e 3: Propriedades do Objeto (Usando Grid para alinhar colunas)
        obj_props_grid = QGridLayout()
        obj_props_grid.setContentsMargins(0, 0, 0, 0)
        obj_props_grid.setHorizontalSpacing(10)

        # Fila A: Identificação
        obj_props_grid.addWidget(QLabel('ID:'), 0, 0)
        obj_props_grid.addWidget(self.object_dial, 0, 1)
        obj_props_grid.addWidget(self.add_object_button, 0, 2)
        obj_props_grid.addWidget(self.object_color, 0, 3)
        self.object_label_edit.setMinimumWidth(150) # Dá mais espaço para o nome
        obj_props_grid.addWidget(self.object_label_edit, 0, 4, 1, 2) # Estende o label

        # Fila B: Metadados (Model, Size, Side, Ref)
        obj_props_grid.addWidget(self.object_model_edit, 1, 0, 1, 2) # Ocupa 2 colunas
        obj_props_grid.addWidget(self.object_size_edit, 1, 2)
        obj_props_grid.addWidget(self.inverted_checkbox, 1, 3)
        obj_props_grid.addWidget(self.reference_checkbox, 1, 4)
        obj_props_grid.addWidget(self.remove_object_button, 1, 5) # Botão perigoso no final da linha

        interact_subbox.addLayout(video_nav_layout)
        interact_subbox.addLayout(obj_props_grid)
        navi.addLayout(interact_subbox)

        # Ajuste de política de tamanho para evitar que os widgets "estiquem" estranhamente
        for i in range(video_nav_layout.count()):
            item = video_nav_layout.itemAt(i).widget()
            if item: apply_fixed_size_policy(item)

        # ---------------------------------------------------------
        # BLOCO 2 (MEIO): Visualização e Exportação + Undo
        # ---------------------------------------------------------
        overlay_subbox = QVBoxLayout()
        
        overlay_topbox = QHBoxLayout()
        overlay_botbox = QHBoxLayout()
        
        overlay_topbox.setAlignment(Qt.AlignmentFlag.AlignLeft)
        overlay_botbox.setAlignment(Qt.AlignmentFlag.AlignLeft)
        
        overlay_topbox.addWidget(QLabel('Vis:'))
        overlay_topbox.addWidget(self.combo)
        overlay_topbox.addWidget(self.save_soft_mask_checkbox)
        overlay_topbox.addWidget(self.export_binary_button)
        
        overlay_botbox.addWidget(QLabel('Save:'))
        overlay_botbox.addWidget(self.save_visualization_combo)
        overlay_botbox.addWidget(self.export_video_button)
        overlay_botbox.addWidget(QLabel('FPS:'))
        overlay_botbox.addWidget(self.fps_dial)
        overlay_botbox.addWidget(QLabel('Mbps:'))
        overlay_botbox.addWidget(self.bitrate_dial)
        
        overlay_subbox.addLayout(overlay_topbox)
        overlay_subbox.addLayout(overlay_botbox)
        navi.addLayout(overlay_subbox)
        
        apply_to_all_children_widget(overlay_topbox, apply_fixed_size_policy)
        apply_to_all_children_widget(overlay_botbox, apply_fixed_size_policy)

        navi.addSpacing(15)

        # ---------------------------------------------------------
        # BLOCO 3 (DIREITA): Controles de Propagação
        # ---------------------------------------------------------
        control_subbox = QVBoxLayout()
        control_topbox = QHBoxLayout()
        control_botbox = QHBoxLayout()
        
        control_topbox.addWidget(self.commit_button)
        control_topbox.addWidget(self.init_frame_button)
        control_topbox.addWidget(self.forward_run_button)
        control_topbox.addWidget(self.backward_run_button)
        
        control_botbox.addWidget(self.progressbar)
        
        control_subbox.addLayout(control_topbox)
        control_subbox.addLayout(control_botbox)
        navi.addLayout(control_subbox)
        
        navi.addStretch(1)

        # Drawing area main canvas
        draw_area = QHBoxLayout()
        draw_area.addWidget(self.main_canvas, 4)

        # right area
        right_area = QVBoxLayout()
        right_area.setAlignment(Qt.AlignmentFlag.AlignBottom)
        right_area.addWidget(self.tips)
        # right_area.addStretch(1)

        # Parameters
        right_area.addLayout(self.perm_mem_gauge_layout)
        right_area.addLayout(self.work_mem_gauge_layout)
        right_area.addLayout(self.long_mem_gauge_layout)
        right_area.addLayout(self.gpu_mem_gauge_layout)
        right_area.addLayout(self.torch_mem_gauge_layout)
        right_area.addWidget(self.clear_all_mem_button)
        right_area.addWidget(self.clear_non_perm_mem_button)
        right_area.addLayout(self.work_mem_min_layout)
        right_area.addLayout(self.work_mem_max_layout)
        right_area.addLayout(self.long_mem_max_layout)
        right_area.addLayout(self.mem_every_box_layout)

        # import mask/layer
        import_area = QHBoxLayout()
        import_area.setAlignment(Qt.AlignmentFlag.AlignBottom)
        import_area.addWidget(self.import_mask_button)
        import_area.addWidget(self.import_layer_button)
        right_area.addLayout(import_area)

        # console
        right_area.addWidget(self.console)

        draw_area.addLayout(right_area, 1)

        layout = QVBoxLayout()
        layout.setMenuBar(self.menu_bar)
        layout.addLayout(draw_area)
        layout.addWidget(self.tl_slider)
        layout.addLayout(navi)
        layout.setContentsMargins(5, 5, 5, 15)
        self.setLayout(layout)

        # timer to play video
        self.timer = QTimer()
        self.timer.setSingleShot(False)
        self.timer.timeout.connect(controller.on_play_video_timer)

        # timer to update GPU usage
        self.gpu_timer = QTimer()
        self.gpu_timer.setSingleShot(False)
        self.gpu_timer.timeout.connect(controller.on_gpu_timer)
        self.gpu_timer.setInterval(2000)
        self.gpu_timer.start()

        # Objects shortcuts
        for i in range(1, controller.num_objects + 1):
            QShortcut(QKeySequence(str(i)), self).activated.connect(functools.partial(controller.hit_number_key, i))

        # next/prev frame shortcuts
        QShortcut(QKeySequence(Qt.Key.Key_Left), self).activated.connect(controller.on_prev_frame)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self).activated.connect(controller.on_next_frame)

        # +/- 10 frames shortcuts
        QShortcut(QKeySequence(Qt.Key.Key_Left | Qt.KeyboardModifier.ShiftModifier),
                    self).activated.connect(functools.partial(controller.on_prev_frame, 10))
        QShortcut(QKeySequence(Qt.Key.Key_Right | Qt.KeyboardModifier.ShiftModifier),
                    self).activated.connect(functools.partial(controller.on_next_frame, 10))
        
        # first/last frame shortcuts
        QShortcut(QKeySequence(Qt.Key.Key_Left | Qt.KeyboardModifier.AltModifier),
                    self).activated.connect(functools.partial(controller.on_prev_frame, 999999))
        QShortcut(QKeySequence(Qt.Key.Key_Right | Qt.KeyboardModifier.AltModifier),
                    self).activated.connect(functools.partial(controller.on_next_frame, 999999))
        
        # commit to permanent memory shortcut
        QShortcut(QKeySequence(Qt.Key.Key_C), self).activated.connect(controller.on_commit)

        # --- NOVO: Undo Shortcut ---
        self.undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.undo_shortcut.activated.connect(controller.on_undo_delete)
        
        # propagate forward/backward/pause shortcuts
        QShortcut(QKeySequence(Qt.Key.Key_F), self).activated.connect(controller.on_forward_propagation)
        QShortcut(QKeySequence(Qt.Key.Key_Space), self).activated.connect(controller.on_forward_propagation)
        QShortcut(QKeySequence(Qt.Key.Key_B), self).activated.connect(controller.on_backward_propagation)

        # quit shortcut
        QShortcut(QKeySequence(Qt.Key.Key_Q), self).activated.connect(self.close)


    def toggle_mode_ui(self, mode: str):
        """
        Habilita ou desabilita widgets baseados no modo.
        """
        is_annotation = (mode == 'annotation')
        
        # Lista de widgets que permitem edição/escrita
        widgets_to_toggle = [
            self.commit_button,
            self.init_frame_button,
            self.forward_run_button,
            self.backward_run_button,
            self.reset_frame_button,
            self.reset_object_button,
            self.remove_object_button,
            self.add_object_button,
            self.import_mask_button,
            self.clear_all_mem_button,
            self.clear_non_perm_mem_button,
            self.object_dial,
            self.object_label_edit,
            self.object_model_edit,
            self.object_size_edit,
            self.save_soft_mask_checkbox,
            self.reference_checkbox,
            self.inverted_checkbox
        ]

        for widget in widgets_to_toggle:
            widget.setEnabled(is_annotation)

        # Atualiza o check do menu visualmente caso a mudança venha de outro lugar
        if is_annotation:
            self.act_annotation.setChecked(True)
        else:
            self.act_view.setChecked(True)

    def image_pos_to_pixel_pos(self, x, y):
        # Convert image coordinates back to screen coordinates for drawing
        oh, ow = self.image_size.height(), self.image_size.width()
        nh, nw = self.main_canvas_size.height(), self.main_canvas_size.width()

        h_ratio = nh / oh
        w_ratio = nw / ow
        dominate_ratio = min(h_ratio, w_ratio)

        # Padding
        fh, fw = nh / dominate_ratio, nw / dominate_ratio
        x += (fw - ow) / 2
        y += (fh - oh) / 2
        
        # Scale
        x *= dominate_ratio
        y *= dominate_ratio
        
        return x, y

    def resizeEvent(self, event):
        self.controller.show_current_frame()

    def text(self, text):
        self.console.moveCursor(QTextCursor.MoveOperation.End)
        self.console.insertPlainText(text + '\n')

    def update_object_label(self, label_text: str):
        self.object_label_edit.blockSignals(True)
        self.object_label_edit.setText(label_text)
        self.object_label_edit.blockSignals(False)

    def update_object_size(self, size_text: str):
        self.object_size_edit.blockSignals(True)
        self.object_size_edit.setText(size_text)
        self.object_size_edit.blockSignals(False)

    def update_object_model(self, model_text: str):
        self.object_model_edit.blockSignals(True)
        self.object_model_edit.setText(model_text)
        self.object_model_edit.blockSignals(False)

    def update_object_reference(self, is_reference: bool):
        self.reference_checkbox.blockSignals(True)
        self.reference_checkbox.setChecked(is_reference)
        self.reference_checkbox.blockSignals(False)

    def update_object_inverted(self, is_inverted: bool):
        self.inverted_checkbox.blockSignals(True)
        self.inverted_checkbox.setChecked(is_inverted)
        self.inverted_checkbox.blockSignals(False)

    def set_canvas(self, image):
        height, width, channel = image.shape
        # if the image is RGBA, convert to RGB first by coloring the background green
        if channel == 4:
            image_rgb = image[:, :, :3].copy()
            alpha = image[:, :, 3].astype(np.float32) / 255
            green_bg = np.array([0, 255, 0])
            # soft blending
            image = (image_rgb * alpha[:, :, np.newaxis] + green_bg[np.newaxis, np.newaxis, :] *
                     (1 - alpha[:, :, np.newaxis])).astype(np.uint8)

        bytesPerLine = 3 * width

        qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)
        self.main_canvas.setPixmap(
            QPixmap(
                qImg.scaled(self.main_canvas.size(), Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.FastTransformation)))

        self.main_canvas_size = self.main_canvas.size()
        self.image_size = qImg.size()

    def update_slider(self, value):
        self.lcd.blockSignals(True)
        self.lcd.setText('{: 3d} / {: 3d}'.format(value, self.controller.T - 1))
        self.lcd.blockSignals(False)
        self.tl_slider.setValue(value)

    def pixel_pos_to_image_pos(self, x, y):
        # Un-scale and un-pad the label coordinates into image coordinates
        oh, ow = self.image_size.height(), self.image_size.width()
        nh, nw = self.main_canvas_size.height(), self.main_canvas_size.width()

        h_ratio = nh / oh
        w_ratio = nw / ow
        dominate_ratio = min(h_ratio, w_ratio)

        # Solve scale
        x /= dominate_ratio
        y /= dominate_ratio

        # Solve padding
        fh, fw = nh / dominate_ratio, nw / dominate_ratio
        x -= (fw - ow) / 2
        y -= (fh - oh) / 2

        return x, y

    def is_pos_out_of_bound(self, x, y):
        x, y = self.pixel_pos_to_image_pos(x, y)

        out_of_bound = ((x < 0) or (y < 0) or (x > self.w - 1) or (y > self.h - 1))

        return out_of_bound

    def get_scaled_pos(self, x, y):
        x, y = self.pixel_pos_to_image_pos(x, y)

        x = max(0, min(self.w - 1, x))
        y = max(0, min(self.h - 1, y))

        return x, y

    def forward_propagation_start(self):
        self.backward_run_button.setEnabled(False)
        self.lcd.setReadOnly(True)
        self.forward_run_button.setText('Pause propagation')

    def backward_propagation_start(self):
        self.forward_run_button.setEnabled(False)
        self.lcd.setReadOnly(True)
        self.backward_run_button.setText('Pause propagation')

    def pause_propagation(self):
        self.forward_run_button.setEnabled(True)
        self.backward_run_button.setEnabled(True)
        self.clear_all_mem_button.setEnabled(True)
        self.clear_non_perm_mem_button.setEnabled(True)
        self.lcd.setReadOnly(False)
        self.forward_run_button.setText('Propagate forward')
        self.backward_run_button.setText('propagate backward')
        self.tl_slider.setEnabled(True)

    def process_events(self):
        QApplication.processEvents()

    def on_mouse_press(self, event):
        # -------------------------------------------------------------------------
        # 1. DEBUG E VERIFICAÇÕES INICIAIS
        # -------------------------------------------------------------------------
        btn_name = "Esquerdo" if event.button() == Qt.MouseButton.LeftButton else "Direito" if event.button() == Qt.MouseButton.RightButton else "Outro"
        self.text(f"--- DEBUG: Clique {btn_name} detectado em ({event.position().x():.1f}, {event.position().y():.1f}) ---")

        if self.is_pos_out_of_bound(event.position().x(), event.position().y()):
            self.text("DEBUG: Clique ignorado (Fora dos limites da imagem).")
            return

        # -------------------------------------------------------------------------
        # 2. CÁLCULO DE COORDENADAS (Unificado)
        # -------------------------------------------------------------------------
        # Calculamos aqui para usar em todos os blocos abaixo
        ex, ey = self.get_scaled_pos(event.position().x(), event.position().y())
        self.text(f"DEBUG: Posição escalada na imagem: x={ex:.1f}, y={ey:.1f}")

        modifiers = QApplication.keyboardModifiers()

        # -------------------------------------------------------------------------
        # 3. LÓGICA YOLO (Prioridade Alta - Overlay)
        # -------------------------------------------------------------------------
        # Verifica se clicou numa sugestão amarela
        if self.controller.show_yolo_suggestions:
             box_idx = self.controller.get_box_index_at(ex, ey)
             
             if box_idx != -1:
                 self.text(f"DEBUG: Clique INTERCEPTADO por caixa YOLO (Index: {box_idx})")
                 
                 # Clique Direito: Modo de Edição (Mover/Esticar)
                 if event.button() == Qt.MouseButton.RightButton:
                     self.text("DEBUG: Ação YOLO: Editar (Transformar em RubberBand)")
                     rect = self.controller.get_yolo_box_rect(box_idx)
                     
                     # Converter coords da imagem -> tela para desenhar o rubberband visualmente
                     x1, y1 = self.image_pos_to_pixel_pos(rect[0], rect[1])
                     x2, y2 = self.image_pos_to_pixel_pos(rect[2], rect[3])
                     
                     self.origin_mouse_pos = QPoint(int(x1), int(y1))
                     self.rubber_band.setGeometry(QRect(QPoint(int(x1), int(y1)), QPoint(int(x2), int(y2))).normalized())
                     self.rubber_band.show()
                     
                     # Troca para ferramenta bbox para permitir o "release" do mouse terminar o desenho
                     self.controller.selection_tool = 'bbox' 
                     self.controller.remove_yolo_candidate(box_idx)
                     return # Encerra aqui

                 # Clique Esquerdo: Aceitar
                 elif event.button() == Qt.MouseButton.LeftButton:
                     self.text("DEBUG: Ação YOLO: Aceitar candidato")
                     self.controller.accept_yolo_candidate(box_idx)
                     return # Encerra aqui

        # -------------------------------------------------------------------------
        # 4. LÓGICA BBOX MANUAL (Ferramenta Selecionada)
        # -------------------------------------------------------------------------
        # Só entra aqui se NÃO clicou numa caixa YOLO (devido aos returns acima)
        # E se não estiver segurando CTRL (pois CTRL força o modo 'Pick')
        if (self.controller.selection_tool == 'bbox' and 
            self.controller.app_mode == 'annotation' and 
            event.button() == Qt.MouseButton.LeftButton and
            modifiers != Qt.KeyboardModifier.ControlModifier):
            
            self.text("DEBUG: Iniciando BBox Manual (RubberBand)")
            self.origin_mouse_pos = event.position().toPoint()
            self.rubber_band.setGeometry(QRect(self.origin_mouse_pos, QSize()))
            self.rubber_band.show()
            return # Encerra aqui para não gerar clique pontual

        # -------------------------------------------------------------------------
        # 5. LÓGICA PADRÃO (Clique Pontual / Pick / Interação)
        # -------------------------------------------------------------------------
        action = None

        # Ctrl + Clique Esquerdo = Pick (Selecionar objeto clicado)
        if (modifiers == Qt.KeyboardModifier.ControlModifier and 
            event.button() == Qt.MouseButton.LeftButton):
            action = 'pick'
            self.text("DEBUG: Ação definida: PICK")
        
        # Botão Esquerdo = Clique Positivo (Add)
        elif event.button() == Qt.MouseButton.LeftButton:
            action = 'left'
            self.text("DEBUG: Ação definida: LEFT (Add)")
        
        # Botão Direito = Clique Negativo (Remove)
        elif event.button() == Qt.MouseButton.RightButton:
            action = 'right'
            self.text("DEBUG: Ação definida: RIGHT (Remove)")
        
        # Botão do Meio = Trocar visualização overlay
        elif event.button() == Qt.MouseButton.MiddleButton:
            action = 'middle'
            self.text("DEBUG: Ação definida: MIDDLE (Vis)")

        if action is None:
            self.text("DEBUG: Nenhuma ação mapeada, ignorando.")
            return
        
        # Executa a ação final
        self.text(f"DEBUG: Enviando clique para controller (Ação: {action})")
        self.click_fn(action, ex, ey)

    def on_mouse_motion(self, event):
        if not self.rubber_band.isHidden():
            self.rubber_band.setGeometry(QRect(self.origin_mouse_pos, event.position().toPoint()).normalized())
            return

        ex, ey = self.get_scaled_pos(event.position().x(), event.position().y())
        
        # Se estiver em modo de visualização, mostra tooltip
        if self.controller.app_mode == 'view':
            info_text = self.controller.get_object_info_at(ex, ey)
            if info_text:
                # Mostra o tooltip perto do mouse, mas sem piscar demais
                # O QToolTip padrão do Qt lida bem com chamadas repetidas se o texto for o mesmo
                QToolTip.showText(event.globalPosition().toPoint(), info_text, self.main_canvas)
            else:
                QToolTip.hideText()
        else:
            # Comportamento original (arrastar cliques, etc)
            self.on_mouse_motion_xy(ex, ey)

    def on_mouse_release(self, event):
        # Finaliza o desenho da BBox
        if not self.rubber_band.isHidden():
            self.rubber_band.hide()
            
            # Pega o retângulo desenhado em coordenadas do widget (pixel da tela)
            rect_widget = self.rubber_band.geometry()
            
            # Precisamos converter o TopLeft e o BottomRight para coordenadas da imagem real
            x1, y1 = self.get_scaled_pos(rect_widget.left(), rect_widget.top())
            x2, y2 = self.get_scaled_pos(rect_widget.right(), rect_widget.bottom())
            
            # Chama o controlador para processar a caixa
            self.controller.on_bbox_complete(x1, y1, x2, y2)

    def on_play_video(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText('Play video')
        else:
            self.timer.start(1000 // 30)
            self.play_button.setText('Stop video')

    def open_file(self, prompt):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self,
                                                   prompt,
                                                   "",
                                                   "Image files (*)",
                                                   options=options)
        return file_name

    def set_object_color(self, object_id: int):
        r, g, b = davis_palette_np[object_id]
        rgb = f'rgb({r},{g},{b})'
        self.object_color.setStyleSheet('QLabel {background: ' + rgb + ';}')
        self.object_color.setText(f'{object_id}')

    def progressbar_update(self, progress: float):
        self.progressbar.setValue(int(progress * 100))
        self.process_events()
