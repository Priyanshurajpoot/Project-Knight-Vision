import cv2
import numpy as np
import chess
import chess.engine
import chess.pgn
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QTextEdit, QGroupBox, QProgressBar, QSlider,
    QCheckBox, QSpinBox, QTabWidget, QSplitter, QFrame, QGridLayout,
    QComboBox, QLineEdit, QFileDialog, QMessageBox, QStatusBar, QMenuBar,
    QMenu, QAction, QDialog, QFormLayout, QDialogButtonBox
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QSettings, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor, QIcon
import os
import sys
import threading
import queue
import platform
import json
from datetime import datetime
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
STOCKFISH_PATHS = {
    "Windows": ["stockfish.exe", "stockfish\\stockfish.exe", "engines\\stockfish.exe"],
    "Darwin": ["stockfish", "/usr/local/bin/stockfish", "/opt/homebrew/bin/stockfish"],
    "Linux": ["stockfish", "/usr/bin/stockfish", "/usr/local/bin/stockfish"]
}

WARPED_SIZE = (480, 480)
SQUARE_SIZE = WARPED_SIZE[0] // 8
ANALYSIS_TIME = 2.0
FRAME_SKIP = 1
APP_NAME = "ChessVision Pro"
APP_VERSION = "2.0"

# Piece mapping for standard chess notation
PIECE_MAPPING = {
    'white_pawn': 'P', 'white_knight': 'N', 'white_bishop': 'B', 
    'white_rook': 'R', 'white_queen': 'Q', 'white_king': 'K',
    'black_pawn': 'p', 'black_knight': 'n', 'black_bishop': 'b', 
    'black_rook': 'r', 'black_queen': 'q', 'black_king': 'k'
}

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.setFixedSize(400, 300)
        self.setup_ui()
        self.load_settings()

    def setup_ui(self):
        layout = QFormLayout(self)
        
        # Stockfish path
        self.stockfish_path = QLineEdit()
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_stockfish)
        stockfish_layout = QHBoxLayout()
        stockfish_layout.addWidget(self.stockfish_path)
        stockfish_layout.addWidget(browse_button)
        layout.addRow("Stockfish Path:", stockfish_layout)
        
        # Analysis time
        self.analysis_time = QSpinBox()
        self.analysis_time.setRange(1, 10)
        self.analysis_time.setValue(2)
        self.analysis_time.setSuffix(" seconds")
        layout.addRow("Analysis Time:", self.analysis_time)
        
        # Camera settings
        self.camera_index = QSpinBox()
        self.camera_index.setRange(0, 5)
        layout.addRow("Camera Index:", self.camera_index)
        
        # Detection sensitivity
        self.detection_sensitivity = QSlider(Qt.Horizontal)
        self.detection_sensitivity.setRange(1, 10)
        self.detection_sensitivity.setValue(5)
        layout.addRow("Detection Sensitivity:", self.detection_sensitivity)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def browse_stockfish(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Stockfish Executable", "", 
            "Executable Files (*.exe);;All Files (*)"
        )
        if file_path:
            self.stockfish_path.setText(file_path)

    def load_settings(self):
        settings = QSettings()
        self.stockfish_path.setText(settings.value("stockfish_path", ""))
        self.analysis_time.setValue(int(settings.value("analysis_time", 2)))
        self.camera_index.setValue(int(settings.value("camera_index", 0)))
        self.detection_sensitivity.setValue(int(settings.value("detection_sensitivity", 5)))

    def save_settings(self):
        settings = QSettings()
        settings.setValue("stockfish_path", self.stockfish_path.text())
        settings.setValue("analysis_time", self.analysis_time.value())
        settings.setValue("camera_index", self.camera_index.value())
        settings.setValue("detection_sensitivity", self.detection_sensitivity.value())

class AnalysisThread(QThread):
    analysis_complete = pyqtSignal(object, float, list, str)
    
    def __init__(self, engine, fen, analysis_time):
        super().__init__()
        self.engine = engine
        self.fen = fen
        self.analysis_time = analysis_time

    def run(self):
        try:
            if not self.engine:
                self.analysis_complete.emit(None, 0.0, [], "Engine not available")
                return

            board = chess.Board(self.fen)
            if not board.is_valid():
                self.analysis_complete.emit(None, 0.0, [], "Invalid position")
                return

            # Get best move
            result = self.engine.play(board, chess.engine.Limit(time=self.analysis_time))
            
            # Get detailed analysis
            info = self.engine.analyse(board, chess.engine.Limit(time=self.analysis_time), multipv=3)
            
            eval_score = 0.0
            pv_moves = []
            
            if isinstance(info, list) and len(info) > 0:
                main_line = info[0]
                score = main_line.get('score')
                if score:
                    relative_score = score.relative
                    if relative_score.is_mate():
                        eval_score = 1000 if relative_score.mate() > 0 else -1000
                    else:
                        eval_score = relative_score.score() / 100.0
                
                pv = main_line.get('pv', [])
                pv_moves = [move.uci() for move in pv[:5]]
            
            self.analysis_complete.emit(result.move, eval_score, pv_moves, "")
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            self.analysis_complete.emit(None, 0.0, [], str(e))

class ChessVisionCore:
    def __init__(self):
        self.engine = None
        self.cap = None
        self.board_corners = None
        self.board_locked = False
        self.current_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        self.frame_count = 0
        self.detection_sensitivity = 5
        self.camera_index = 0
        self.analysis_time = 2.0
        
    def find_stockfish(self):
        """Find Stockfish engine on system"""
        system = platform.system()
        paths = STOCKFISH_PATHS.get(system, ["stockfish"])
        
        # Check settings first
        settings = QSettings()
        custom_path = settings.value("stockfish_path", "")
        if custom_path and os.path.exists(custom_path):
            return custom_path
        
        # Check standard locations
        for path in paths:
            if os.path.exists(path):
                return path
            
            # Check in PATH
            try:
                import subprocess
                subprocess.run([path, "--help"], capture_output=True, timeout=5)
                return path
            except:
                continue
        
        return None

    def init_engine(self):
        """Initialize Stockfish engine"""
        stockfish_path = self.find_stockfish()
        if not stockfish_path:
            logger.error("Stockfish not found. Please install Stockfish and set path in settings.")
            return False
        
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            self.engine.configure({"Threads": 2, "Hash": 128})
            logger.info(f"Stockfish engine initialized: {stockfish_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize engine: {e}")
            return False

    def init_camera(self):
        """Initialize camera with robust settings"""
        settings = QSettings()
        self.camera_index = int(settings.value("camera_index", 0))
        
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            # Try different backends
            for backend in [cv2.CAP_DSHOW, cv2.CAP_V4L2, cv2.CAP_ANY]:
                self.cap = cv2.VideoCapture(self.camera_index, backend)
                if self.cap.isOpened():
                    break
        
        if self.cap.isOpened():
            # Set optimal camera settings
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            return True
        
        return False

    def detect_chessboard(self, frame):
        """Improved chessboard detection using multiple methods"""
        if self.board_locked and self.board_corners is not None:
            return self.board_corners
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Chessboard corner detection
        ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)
        if ret:
            # Extract outer corners to form board boundary
            corners = corners.reshape(-1, 2)
            # Get approximate board corners from inner corners
            top_left = corners[0]
            top_right = corners[6]
            bottom_left = corners[42]
            bottom_right = corners[48]
            
            # Expand to get full board
            board_corners = np.array([
                top_left - (top_right - top_left) * 0.1,
                top_right + (top_right - top_left) * 0.1,
                bottom_right + (bottom_right - bottom_left) * 0.1,
                bottom_left - (bottom_right - bottom_left) * 0.1
            ], dtype=np.float32)
            
            self.board_corners = board_corners.reshape(4, 1, 2).astype(np.int32)
            return self.board_corners
        
        # Method 2: Contour-based detection (fallback)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        min_area = 50000 * (self.detection_sensitivity / 10.0)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                if len(approx) == 4:
                    self.board_corners = approx
                    return self.board_corners
        
        return None

    def warp_perspective(self, frame, corners):
        """Warp board to standard view"""
        if corners is None:
            return None
            
        # Order points: top-left, top-right, bottom-right, bottom-left
        pts = corners.reshape(4, 2).astype(np.float32)
        
        # Calculate center and sort points
        center = np.mean(pts, axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        
        # Reorder: start from top-left, go clockwise
        ordered_pts = pts[sorted_indices]
        
        # Destination points for warping
        dst = np.array([
            [0, 0],
            [WARPED_SIZE[0], 0],
            [WARPED_SIZE[0], WARPED_SIZE[1]],
            [0, WARPED_SIZE[1]]
        ], dtype=np.float32)
        
        # Get perspective transform and warp
        matrix = cv2.getPerspectiveTransform(ordered_pts, dst)
        warped = cv2.warpPerspective(frame, matrix, WARPED_SIZE)
        
        return warped

    def analyze_board_colors(self, warped):
        """Analyze board to determine piece colors using color clustering"""
        if warped is None:
            return None
            
        # Convert to different color spaces for better analysis
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
        
        board_state = [['.' for _ in range(8)] for _ in range(8)]
        
        for row in range(8):
            for col in range(8):
                # Extract square
                y1 = row * SQUARE_SIZE
                y2 = y1 + SQUARE_SIZE
                x1 = col * SQUARE_SIZE
                x2 = x1 + SQUARE_SIZE
                
                square_bgr = warped[y1:y2, x1:x2]
                square_hsv = hsv[y1:y2, x1:x2]
                square_lab = lab[y1:y2, x1:x2]
                
                # Detect if square has a piece
                piece_type = self.detect_piece_in_square(square_bgr, square_hsv, square_lab, row, col)
                board_state[row][col] = piece_type
        
        return board_state

    def detect_piece_in_square(self, square_bgr, square_hsv, square_lab, row, col):
        """Detect piece in individual square using multiple features"""
        # Check if square is likely empty
        gray = cv2.cvtColor(square_bgr, cv2.COLOR_BGR2GRAY)
        
        # Calculate variance to detect pieces (pieces create more texture)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if variance < 50:  # Likely empty square
            return '.'
        
        # Color analysis for piece detection
        mask_center = np.zeros(square_bgr.shape[:2], dtype=np.uint8)
        cv2.circle(mask_center, (SQUARE_SIZE//2, SQUARE_SIZE//2), SQUARE_SIZE//3, 255, -1)
        
        # Get average color of center region
        mean_color = cv2.mean(square_bgr, mask_center)
        brightness = (mean_color[0] + mean_color[1] + mean_color[2]) / 3
        
        # Determine if piece is white or black based on brightness
        is_white = brightness > 100
        
        # Simple piece type detection based on shape analysis
        # This is a simplified approach - in production, you'd use trained models
        piece_type = self.classify_piece_shape(gray, mask_center)
        
        if piece_type == 'unknown':
            # Default to pawn if we can't classify
            return 'P' if is_white else 'p'
        
        # Map to appropriate piece
        piece_map = {
            'pawn': 'P' if is_white else 'p',
            'rook': 'R' if is_white else 'r',
            'knight': 'N' if is_white else 'n',
            'bishop': 'B' if is_white else 'b',
            'queen': 'Q' if is_white else 'q',
            'king': 'K' if is_white else 'k'
        }
        
        return piece_map.get(piece_type, 'P' if is_white else 'p')

    def classify_piece_shape(self, gray, mask):
        """Basic piece shape classification"""
        # Find contours in the masked region
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 'unknown'
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Basic shape analysis
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        if area < 100:
            return 'unknown'
        
        # Calculate shape metrics
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Simple classification based on shape
        if circularity > 0.7:
            return 'pawn'  # Round shapes are likely pawns
        else:
            return 'unknown'  # More complex shapes need better analysis

    def board_to_fen(self, board_state, turn='w'):
        """Convert board state to FEN notation"""
        fen_rows = []
        
        # Process board from rank 8 to rank 1 (top to bottom visually)
        for row in range(8):
            fen_row = ''
            empty_count = 0
            
            for col in range(8):
                piece = board_state[row][col]
                
                if piece == '.':
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen_row += str(empty_count)
                        empty_count = 0
                    fen_row += piece
            
            if empty_count > 0:
                fen_row += str(empty_count)
            
            fen_rows.append(fen_row)
        
        # Combine with game state info
        board_fen = '/'.join(fen_rows)
        full_fen = f"{board_fen} {turn} KQkq - 0 1"
        
        return full_fen

    def process_frame(self, frame):
        """Main frame processing pipeline"""
        self.frame_count += 1
        
        # Skip frames for performance
        if self.frame_count % (FRAME_SKIP + 1) != 0:
            return frame, None, None, None
        
        # Detect chessboard
        corners = self.detect_chessboard(frame)
        
        if corners is not None:
            # Draw detected board
            cv2.drawContours(frame, [corners], -1, (0, 255, 0), 3)
            
            # Warp perspective
            warped = self.warp_perspective(frame, corners)
            
            if warped is not None:
                # Analyze board
                board_state = self.analyze_board_colors(warped)
                
                if board_state:
                    fen = self.board_to_fen(board_state)
                    self.current_fen = fen
                    return frame, corners, warped, fen
        
        return frame, corners, None, None

    def cleanup(self):
        """Clean up resources"""
        if self.engine:
            self.engine.quit()
        if self.cap:
            self.cap.release()

class ChessVisionMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        QApplication.setOrganizationName("ChessVision")
        QApplication.setApplicationName("ChessVision Pro")
        
        self.core = ChessVisionCore()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.running = False
        self.analysis_thread = None
        
        self.init_ui()
        self.init_engine()
        self.apply_styling()

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setGeometry(100, 100, 1400, 900)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Central widget with splitter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Create main splitter
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Video and controls
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Analysis and info
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter sizes
        splitter.setSizes([800, 600])

    def create_menu_bar(self):
        """Create application menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        save_fen_action = QAction('Save FEN...', self)
        save_fen_action.triggered.connect(self.save_fen)
        file_menu.addAction(save_fen_action)
        
        save_pgn_action = QAction('Save PGN...', self)
        save_pgn_action.triggered.connect(self.save_pgn)
        file_menu.addAction(save_pgn_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Settings menu
        settings_menu = menubar.addMenu('Settings')
        
        preferences_action = QAction('Preferences...', self)
        preferences_action.triggered.connect(self.show_preferences)
        settings_menu.addAction(preferences_action)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        about_action = QAction('About...', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def create_left_panel(self):
        """Create left panel with video feed and controls"""
        panel = QFrame()
        layout = QVBoxLayout(panel)
        
        # Video feed group
        video_group = QGroupBox("Live Feed")
        video_layout = QVBoxLayout(video_group)
        
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("border: 2px solid #cccccc; background-color: #000000;")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("No Camera Feed")
        video_layout.addWidget(self.video_label)
        
        layout.addWidget(video_group)
        
        # Controls group
        controls_group = QGroupBox("Camera Controls")
        controls_layout = QGridLayout(controls_group)
        
        # Main controls
        self.start_button = QPushButton("Start Camera")
        self.start_button.clicked.connect(self.toggle_camera)
        controls_layout.addWidget(self.start_button, 0, 0)
        
        self.calibrate_button = QPushButton("Reset Detection")
        self.calibrate_button.clicked.connect(self.reset_detection)
        controls_layout.addWidget(self.calibrate_button, 0, 1)
        
        # Lock board checkbox
        self.lock_board_cb = QCheckBox("Lock Board")
        self.lock_board_cb.stateChanged.connect(self.toggle_board_lock)
        controls_layout.addWidget(self.lock_board_cb, 1, 0)
        
        # Detection sensitivity
        controls_layout.addWidget(QLabel("Sensitivity:"), 1, 1)
        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setRange(1, 10)
        self.sensitivity_slider.setValue(5)
        self.sensitivity_slider.valueChanged.connect(self.update_sensitivity)
        controls_layout.addWidget(self.sensitivity_slider, 2, 0, 1, 2)
        
        layout.addWidget(controls_group)
        
        # Status group
        status_group = QGroupBox("Detection Status")
        status_layout = QVBoxLayout(status_group)
        
        self.detection_status = QLabel("Waiting for camera...")
        status_layout.addWidget(self.detection_status)
        
        self.fps_label = QLabel("FPS: 0")
        status_layout.addWidget(self.fps_label)
        
        layout.addWidget(status_group)
        
        return panel

    def create_right_panel(self):
        """Create right panel with analysis tabs"""
        panel = QFrame()
        layout = QVBoxLayout(panel)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Analysis tab
        self.create_analysis_tab()
        
        # Position tab
        self.create_position_tab()
        
        # History tab
        self.create_history_tab()
        
        return panel

    def create_analysis_tab(self):
        """Create analysis tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Engine status
        engine_group = QGroupBox("Engine Status")
        engine_layout = QVBoxLayout(engine_group)
        
        self.engine_status = QLabel("Initializing engine...")
        engine_layout.addWidget(self.engine_status)
        
        layout.addWidget(engine_group)
        
        # Analysis results
        analysis_group = QGroupBox("Analysis")
        analysis_layout = QVBoxLayout(analysis_group)
        
        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setMaximumHeight(200)
        analysis_layout.addWidget(self.analysis_text)
        
        # Best move display
        move_group = QGroupBox("Best Move")
        move_layout = QVBoxLayout(move_group)
        
        self.best_move_label = QLabel("--")
        self.best_move_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.best_move_label.setAlignment(Qt.AlignCenter)
        move_layout.addWidget(self.best_move_label)
        
        self.evaluation_label = QLabel("Evaluation: --")
        self.evaluation_label.setAlignment(Qt.AlignCenter)
        move_layout.addWidget(self.evaluation_label)
        
        layout.addWidget(analysis_group)
        layout.addWidget(move_group)
        
        # Analysis progress
        self.analysis_progress = QProgressBar()
        self.analysis_progress.setVisible(False)
        layout.addWidget(self.analysis_progress)
        
        self.tab_widget.addTab(tab, "Analysis")

    def create_position_tab(self):
        """Create position information tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # FEN display
        fen_group = QGroupBox("Position (FEN)")
        fen_layout = QVBoxLayout(fen_group)
        
        self.fen_text = QTextEdit()
        self.fen_text.setMaximumHeight(100)
        self.fen_text.setPlainText("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        fen_layout.addWidget(self.fen_text)
        
        layout.addWidget(fen_group)
        
        # Position info
        info_group = QGroupBox("Position Information")
        info_layout = QFormLayout(info_group)
        
        self.turn_label = QLabel("White")
        info_layout.addRow("Turn:", self.turn_label)
        
        self.material_label = QLabel("Equal")
        info_layout.addRow("Material:", self.material_label)
        
        layout.addWidget(info_group)
        
        # Manual input
        manual_group = QGroupBox("Manual Input")
        manual_layout = QVBoxLayout(manual_group)
        
        self.manual_fen = QLineEdit()
        self.manual_fen.setPlaceholderText("Enter FEN notation...")
        manual_layout.addWidget(self.manual_fen)
        
        load_fen_btn = QPushButton("Load FEN")
        load_fen_btn.clicked.connect(self.load_manual_fen)
        manual_layout.addWidget(load_fen_btn)
        
        layout.addWidget(manual_group)
        
        self.tab_widget.addTab(tab, "Position")

    def create_history_tab(self):
        """Create move history tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Move history
        history_group = QGroupBox("Move History")
        history_layout = QVBoxLayout(history_group)
        
        self.history_text = QTextEdit()
        self.history_text.setReadOnly(True)
        history_layout.addWidget(self.history_text)
        
        layout.addWidget(history_group)
        
        # Export controls
        export_group = QGroupBox("Export")
        export_layout = QHBoxLayout(export_group)
        
        export_pgn_btn = QPushButton("Export PGN")
        export_pgn_btn.clicked.connect(self.save_pgn)
        export_layout.addWidget(export_pgn_btn)
        
        clear_history_btn = QPushButton("Clear History")
        clear_history_btn.clicked.connect(self.clear_history)
        export_layout.addWidget(clear_history_btn)
        
        layout.addWidget(export_group)
        
        self.tab_widget.addTab(tab, "History")

    def apply_styling(self):
        """Apply modern styling to the application"""
        style = """
        QMainWindow {
            background-color: #f5f5f5;
        }
        
        QGroupBox {
            font-weight: bold;
            border: 2px solid #cccccc;
            border-radius: 8px;
            margin: 5px;
            padding-top: 15px;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
            color: #2c3e50;
        }
        
        QPushButton {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
        }
        
        QPushButton:hover {
            background-color: #2980b9;
        }
        
        QPushButton:pressed {
            background-color: #21618c;
        }
        
        QPushButton:disabled {
            background-color: #95a5a6;
        }
        
        QTextEdit {
            border: 1px solid #bdc3c7;
            border-radius: 4px;
            padding: 5px;
            font-family: 'Courier New', monospace;
        }
        
        QLabel {
            color: #2c3e50;
        }
        
        QTabWidget::pane {
            border: 1px solid #bdc3c7;
            border-radius: 4px;
        }
        
        QTabBar::tab {
            background-color: #ecf0f1;
            padding: 8px 16px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        
        QTabBar::tab:selected {
            background-color: #3498db;
            color: white;
        }
        
        QProgressBar {
            border: 1px solid #bdc3c7;
            border-radius: 4px;
            text-align: center;
        }
        
        QProgressBar::chunk {
            background-color: #2ecc71;
            border-radius: 3px;
        }
        """
        self.setStyleSheet(style)

    def init_engine(self):
        """Initialize chess engine"""
        if self.core.init_engine():
            self.engine_status.setText("✓ Stockfish ready")
            self.engine_status.setStyleSheet("color: green;")
        else:
            self.engine_status.setText("✗ Engine not available")
            self.engine_status.setStyleSheet("color: red;")

    @pyqtSlot()
    def toggle_camera(self):
        """Start/stop camera feed"""
        if not self.running:
            if self.core.init_camera():
                self.timer.start(33)  # ~30 FPS
                self.running = True
                self.start_button.setText("Stop Camera")
                self.detection_status.setText("Camera started")
                self.status_bar.showMessage("Camera running")
            else:
                QMessageBox.critical(self, "Error", "Failed to initialize camera")
        else:
            self.timer.stop()
            self.running = False
            self.start_button.setText("Start Camera")
            self.detection_status.setText("Camera stopped")
            self.status_bar.showMessage("Camera stopped")
            self.video_label.setText("No Camera Feed")

    @pyqtSlot()
    def reset_detection(self):
        """Reset board detection"""
        self.core.board_locked = False
        self.core.board_corners = None
        self.lock_board_cb.setChecked(False)
        self.detection_status.setText("Detection reset")

    @pyqtSlot(int)
    def toggle_board_lock(self, state):
        """Toggle board lock"""
        self.core.board_locked = state == Qt.Checked

    @pyqtSlot(int)
    def update_sensitivity(self, value):
        """Update detection sensitivity"""
        self.core.detection_sensitivity = value

    @pyqtSlot()
    def update_frame(self):
        """Update video frame"""
        if not self.core.cap or not self.core.cap.isOpened():
            return

        ret, frame = self.core.cap.read()
        if not ret:
            return

        # Process frame
        processed_frame, corners, warped, fen = self.core.process_frame(frame)
        
        # Update detection status
        if corners is not None:
            self.detection_status.setText("✓ Board detected")
            self.detection_status.setStyleSheet("color: green;")
            if self.core.board_locked:
                self.lock_board_cb.setChecked(True)
        else:
            self.detection_status.setText("✗ No board detected")
            self.detection_status.setStyleSheet("color: red;")

        # Display frame
        if processed_frame is not None:
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Scale to fit label
            scaled_image = qt_image.scaled(
                self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.video_label.setPixmap(QPixmap.fromImage(scaled_image))

        # Update FEN if available
        if fen:
            self.fen_text.setPlainText(fen)
            self.update_position_info(fen)
            
            # Start analysis if engine is available
            if self.core.engine and (self.analysis_thread is None or not self.analysis_thread.isRunning()):
                self.start_analysis(fen)

    def update_position_info(self, fen):
        """Update position information display"""
        try:
            board = chess.Board(fen)
            
            # Update turn
            self.turn_label.setText("White" if board.turn else "Black")
            
            # Calculate material balance
            material_balance = self.calculate_material_balance(board)
            if material_balance > 0:
                self.material_label.setText(f"White +{material_balance}")
            elif material_balance < 0:
                self.material_label.setText(f"Black +{-material_balance}")
            else:
                self.material_label.setText("Equal")
                
        except Exception as e:
            logger.error(f"Error updating position info: {e}")

    def calculate_material_balance(self, board):
        """Calculate material balance"""
        piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
        }
        
        white_material = 0
        black_material = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    white_material += value
                else:
                    black_material += value
        
        return white_material - black_material

    def start_analysis(self, fen):
        """Start position analysis in background thread"""
        self.analysis_progress.setVisible(True)
        self.analysis_progress.setRange(0, 0)  # Indeterminate progress
        
        settings = QSettings()
        analysis_time = float(settings.value("analysis_time", 2.0))
        
        self.analysis_thread = AnalysisThread(self.core.engine, fen, analysis_time)
        self.analysis_thread.analysis_complete.connect(self.on_analysis_complete)
        self.analysis_thread.start()

    @pyqtSlot(object, float, list, str)
    def on_analysis_complete(self, best_move, evaluation, pv_moves, error):
        """Handle completed analysis"""
        self.analysis_progress.setVisible(False)
        
        if error:
            self.analysis_text.append(f"Analysis error: {error}")
            return
        
        if best_move:
            # Update best move display
            self.best_move_label.setText(best_move.uci())
            
            # Format evaluation
            if abs(evaluation) > 900:
                eval_text = f"Mate in {int(1000 - abs(evaluation))}"
            else:
                eval_text = f"{evaluation:+.2f}"
            
            self.evaluation_label.setText(f"Evaluation: {eval_text}")
            
            # Update analysis text
            analysis_text = f"Best move: {best_move.uci()}\n"
            analysis_text += f"Evaluation: {eval_text}\n"
            
            if pv_moves:
                analysis_text += f"Principal variation: {' '.join(pv_moves[:5])}\n"
            
            self.analysis_text.append(analysis_text)
            
            # Add to history
            timestamp = datetime.now().strftime("%H:%M:%S")
            history_entry = f"[{timestamp}] {best_move.uci()} ({eval_text})"
            self.history_text.append(history_entry)
            
            # Scroll to bottom
            self.analysis_text.moveCursor(self.analysis_text.textCursor().End)
            self.history_text.moveCursor(self.history_text.textCursor().End)

    @pyqtSlot()
    def load_manual_fen(self):
        """Load manually entered FEN"""
        fen = self.manual_fen.text().strip()
        if fen:
            try:
                # Validate FEN
                board = chess.Board(fen)
                self.fen_text.setPlainText(fen)
                self.core.current_fen = fen
                self.update_position_info(fen)
                self.start_analysis(fen)
                self.manual_fen.clear()
            except Exception as e:
                QMessageBox.warning(self, "Invalid FEN", f"The entered FEN is invalid:\n{e}")

    @pyqtSlot()
    def save_fen(self):
        """Save current FEN to file"""
        if self.core.current_fen:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save FEN", f"position_{datetime.now().strftime('%Y%m%d_%H%M%S')}.fen",
                "FEN Files (*.fen);;Text Files (*.txt);;All Files (*)"
            )
            if file_path:
                try:
                    with open(file_path, 'w') as f:
                        f.write(self.core.current_fen)
                    QMessageBox.information(self, "Success", f"FEN saved to {file_path}")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to save FEN:\n{e}")

    @pyqtSlot()
    def save_pgn(self):
        """Save analysis as PGN"""
        if not self.history_text.toPlainText():
            QMessageBox.information(self, "No Data", "No analysis history to save.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save PGN", f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pgn",
            "PGN Files (*.pgn);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(f'[Event "ChessVision Pro Analysis"]\n')
                    f.write(f'[Date "{datetime.now().strftime("%Y.%m.%d")}"]\n')
                    f.write(f'[White "Human"]\n')
                    f.write(f'[Black "Analysis"]\n')
                    f.write(f'[Result "*"]\n\n')
                    
                    # Add analysis as comments
                    history = self.history_text.toPlainText()
                    for line in history.split('\n'):
                        if line.strip():
                            f.write(f"; {line}\n")
                    
                    f.write("\n*\n")
                
                QMessageBox.information(self, "Success", f"Analysis saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save PGN:\n{e}")

    @pyqtSlot()
    def clear_history(self):
        """Clear analysis history"""
        reply = QMessageBox.question(
            self, "Clear History", "Are you sure you want to clear the analysis history?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.history_text.clear()
            self.analysis_text.clear()

    @pyqtSlot()
    def show_preferences(self):
        """Show preferences dialog"""
        dialog = SettingsDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            dialog.save_settings()
            
            # Update core settings
            settings = QSettings()
            self.core.analysis_time = float(settings.value("analysis_time", 2.0))
            self.core.detection_sensitivity = int(settings.value("detection_sensitivity", 5))
            
            # Reinitialize engine if path changed
            old_engine = self.core.engine
            if self.core.init_engine():
                if old_engine:
                    old_engine.quit()
                self.engine_status.setText("✓ Stockfish ready")
                self.engine_status.setStyleSheet("color: green;")

    @pyqtSlot()
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self, "About ChessVision Pro",
            f"<h3>{APP_NAME} v{APP_VERSION}</h3>"
            "<p>Professional chess position analysis using computer vision.</p>"
            "<p>Features:</p>"
            "<ul>"
            "<li>Real-time board detection</li>"
            "<li>Automatic piece recognition</li>"
            "<li>Stockfish engine analysis</li>"
            "<li>FEN and PGN export</li>"
            "</ul>"
            "<p><b>Requirements:</b> Stockfish engine, webcam</p>"
            "<p>© 2024 ChessVision Pro</p>"
        )

    def closeEvent(self, event):
        """Handle application close"""
        if self.running:
            self.timer.stop()
        
        # Stop analysis thread
        if self.analysis_thread and self.analysis_thread.isRunning():
            self.analysis_thread.terminate()
            self.analysis_thread.wait()
        
        # Cleanup core resources
        self.core.cleanup()
        
        # Save settings
        settings = QSettings()
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())
        
        event.accept()

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationVersion(APP_VERSION)
    
    # Set application icon (if available)
    icon_path = "icon.png"
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    
    # Apply application-wide styling
    app.setStyle('Fusion')
    
    # Create and show main window
    window = ChessVisionMainWindow()
    
    # Restore geometry
    settings = QSettings()
    geometry = settings.value("geometry")
    if geometry:
        window.restoreGeometry(geometry)
    
    window_state = settings.value("windowState")
    if window_state:
        window.restoreState(window_state)
    
    window.show()
    
    # Handle system exit
    try:
        sys.exit(app.exec_())
    except SystemExit:
        pass

if __name__ == "__main__":
    main()