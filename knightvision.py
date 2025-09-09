import cv2
import numpy as np
import chess
import chess.engine
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QTextEdit, QHBoxLayout)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import os
import sys
import threading
import queue
import platform
import pytesseract
from datetime import datetime

# Constants
STOCKFISH_PATH = "stockfish\stockfish-windows-x86-64-avx2.exe" if platform.system() == "Windows" else "stockfish"
WARPED_SIZE = (400, 400)
SQUARE_SIZE = WARPED_SIZE[0] // 8
PIECE_TEMPLATES = {}
ANALYSIS_TIME = 2.0  # seconds
FRAME_SKIP = 2  # Process every 3rd frame for performance
APP_NAME = "ChessVision Pro"

# Piece mapping: template key to FEN char
PIECE_MAPPING = {
    'wp': 'P', 'wn': 'N', 'wb': 'B', 'wr': 'R', 'wq': 'Q', 'wk': 'K',
    'bp': 'p', 'bn': 'n', 'bb': 'b', 'br': 'r', 'bq': 'q', 'bk': 'k'
}

class ChessVisionPro:
    def __init__(self):
        self.cap = None
        self.engine = None
        self.locked_contour = None
        self.board_locked = False
        self.current_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        self.result_queue = queue.Queue()
        self.last_move = None
        self.last_eval = 0.0
        self.init_engine()
        self.load_templates()
        self.setup_camera()

    def init_engine(self):
        """Initialize Stockfish engine with multi-PV for advanced analysis."""
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
            self.engine.configure({"Threads": 2, "MultiPV": 3})  # Top 3 moves
            print(f"{APP_NAME}: Stockfish engine initialized.")
        except Exception as e:
            print(f"{APP_NAME}: Error initializing Stockfish: {e}. Download from stockfishchess.org and update STOCKFISH_PATH.")
            self.engine = None

    def load_templates(self):
        """Load or generate templates for piece recognition."""
        template_dir = "templates"
        os.makedirs(template_dir, exist_ok=True)
        piece_keys = list(PIECE_MAPPING.keys())
        
        for key in piece_keys:
            template_path = os.path.join(template_dir, f"{key}.png")
            if os.path.exists(template_path):
                template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                if template is not None and template.shape[:2] == (40, 40):
                    PIECE_TEMPLATES[key] = template
            else:
                print(f"{APP_NAME}: Warning: Missing template {template_path}. Using placeholder.")
                template = np.zeros((40, 40), dtype=np.uint8)
                if 'w' in key:
                    cv2.circle(template, (20, 20), 15, 255, -1)
                else:
                    cv2.rectangle(template, (5, 5), (35, 35), 255, -1)
                PIECE_TEMPLATES[key] = template
                cv2.imwrite(template_path, template)
        
        print(f"{APP_NAME}: Loaded {len(PIECE_TEMPLATES)}/{len(piece_keys)} templates.")

    def setup_camera(self):
        """Initialize camera with fallback options."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print(f"{APP_NAME}: Camera not accessible. Trying default settings.")
            self.cap = cv2.VideoCapture(0, cv2.CAP_ANY)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def preprocess_image(self, frame):
        """Preprocess with adaptive thresholding for robustness."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        edges = cv2.Canny(thresh, 50, 150)
        return edges, thresh

    def detect_chessboard(self, edges):
        """Detect and lock the chessboard contour."""
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        if not self.board_locked:
            max_area = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 10000:
                    peri = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                    if len(approx) == 4 and area > max_area:
                        max_area = area
                        self.locked_contour = approx
            if self.locked_contour is not None:
                self.board_locked = True
        return self.locked_contour

    def warp_chessboard(self, frame, board_contour):
        """Warp the detected board to a square view."""
        pts = board_contour.reshape(4, 2).astype(np.float32)
        s = pts.sum(axis=1)
        rect = np.zeros((4, 2), dtype="float32")
        rect[0] = pts[np.argmin(s)]  # TL
        rect[2] = pts[np.argmax(s)]  # BR
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # TR
        rect[3] = pts[np.argmax(diff)]  # BL
        
        dst = np.array([[0, 0], [WARPED_SIZE[0], 0], 
                        [WARPED_SIZE[0], WARPED_SIZE[1]], [0, WARPED_SIZE[1]]], dtype="float32")
        
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(frame, M, WARPED_SIZE)
        return warped

    def recognize_pieces(self, warped):
        """Recognize pieces with template matching and confidence scoring."""
        squares = self.extract_squares(warped)
        board_state = [['.' for _ in range(8)] for _ in range(8)]
        
        for row in range(8):
            for col in range(8):
                square = squares[row][col]
                gray_square = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
                gray_square = cv2.resize(gray_square, (40, 40))
                
                best_match = None
                best_score = 0.65
                
                for piece_key, template in PIECE_TEMPLATES.items():
                    res = cv2.matchTemplate(gray_square, template, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(res)
                    if max_val > best_score:
                        best_score = max_val
                        best_match = piece_key
                
                if best_match and best_score > 0.7:
                    board_state[row][col] = PIECE_MAPPING[best_match]
        
        return board_state

    def extract_squares(self, warped):
        """Divide warped board into 64 squares."""
        squares = []
        for row in range(8):
            row_squares = []
            for col in range(8):
                y1 = row * SQUARE_SIZE
                y2 = y1 + SQUARE_SIZE
                x1 = col * SQUARE_SIZE
                x2 = x1 + SQUARE_SIZE
                square = warped[y1:y2, x1:x2]
                row_squares.append(square)
            squares.append(row_squares)
        return squares

    def generate_fen(self, board_state):
        """Generate FEN with OCR-based turn detection."""
        fen_rows = []
        for row in board_state[::-1]:
            empty_count = 0
            row_str = ''
            for cell in row:
                if cell == '.':
                    empty_count += 1
                else:
                    if empty_count > 0:
                        row_str += str(empty_count)
                        empty_count = 0
                    row_str += cell
            if empty_count > 0:
                row_str += str(empty_count)
            fen_rows.append(row_str)
        
        turn = 'w'
        try:
            ocr_text = pytesseract.image_to_string(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY))
            if "black" in ocr_text.lower() or "black's" in ocr_text.lower():
                turn = 'b'
        except Exception:
            print(f"{APP_NAME}: OCR failed, defaulting to white's turn.")
        
        fen = '/'.join(fen_rows) + f' {turn} KQkq - 0 1'
        return fen

    def analyze_position(self, fen):
        """Analyze FEN with Stockfish, returning top moves."""
        if self.engine is None:
            return None, 0.0, [], "Engine not available"
        
        try:
            board = chess.Board(fen)
            if not board.is_valid():
                return None, 0.0, [], "Invalid FEN"
            
            result = self.engine.play(board, chess.engine.Limit(time=ANALYSIS_TIME))
            info = self.engine.analyse(board, chess.engine.Limit(time=1.0))
            score = info.get('score', {}).relative
            eval_score = score.score(mate_score=10000) / 100.0 if score else 0.0
            pv = [move.uci() for move in info.get('pv', [])[:3]]
            
            return result.move, eval_score, pv, ""
        except Exception as e:
            return None, 0.0, [], f"Analysis error: {e}"

    def process_frame(self):
        """Process camera frame with frame skipping."""
        if self.cap.get(cv2.CAP_PROP_POS_FRAMES) % (FRAME_SKIP + 1) != 0:
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        edges, _ = self.preprocess_image(frame)
        board_contour = self.detect_chessboard(edges)
        
        if board_contour is not None:
            warped = self.warp_chessboard(frame, board_contour)
            board_state = self.recognize_pieces(warped)
            fen = self.generate_fen(board_state)
            self.current_fen = fen
            
            threading.Thread(target=lambda: self.result_queue.put(self.analyze_position(fen)), daemon=True).start()
        
        return frame, board_contour, warped if board_contour is not None else None

    def export_pgn(self):
        """Export current position and move to PGN file."""
        if self.last_move:
            board = chess.Board(self.current_fen)
            game = chess.pgn.Game()
            game.headers["Event"] = "ChessVision Pro Analysis"
            game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
            game.headers["White"] = "Player"
            game.headers["Black"] = "Stockfish"
            node = game.add_variation(self.last_move)
            node.comment = f"Evaluation: {self.last_eval:+.2f}"
            
            with open("chessvision_analysis.pgn", "a") as f:
                print(game, file=f, end="\n\n")
            print(f"{APP_NAME}: Exported to chessvision_analysis.pgn")
        else:
            print(f"{APP_NAME}: No move to export. Analyze a position first.")

class ChessVisionProWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.solver = ChessVisionPro()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(APP_NAME)
        self.setGeometry(100, 100, 1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Video Feed
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label)

        # Control Panel
        control_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Camera")
        self.start_button.clicked.connect(self.start_camera)
        self.calibrate_button = QPushButton("Calibrate Board")
        self.calibrate_button.clicked.connect(self.calibrate_board)
        self.export_button = QPushButton("Export PGN")
        self.export_button.clicked.connect(self.solver.export_pgn)
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.calibrate_button)
        control_layout.addWidget(self.export_button)
        layout.addLayout(control_layout)

        # Analysis Output
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        layout.addWidget(self.output_text)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.running = False

    def start_camera(self):
        if not self.running:
            self.timer.start(33)  # ~30 FPS
            self.running = True
            self.start_button.setText("Stop Camera")
        else:
            self.timer.stop()
            self.running = False
            self.start_button.setText("Start Camera")

    def calibrate_board(self):
        self.solver.board_locked = False
        self.solver.locked_contour = None
        self.output_text.append(f"{APP_NAME}: Calibration reset. Please align the board.")

    def update_frame(self):
        if self.running:
            frame_data = self.solver.process_frame()
            if frame_data:
                frame, board_contour, warped = frame_data
                if board_contour is not None:
                    cv2.drawContours(frame, [board_contour], -1, (0, 255, 0), 3)
                
                # Display frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                bytes_per_line = ch * w
                image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                scaled_image = image.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.video_label.setPixmap(QPixmap.fromImage(scaled_image))

                # Update analysis
                if not self.solver.result_queue.empty():
                    move, eval_score, pv, error = self.solver.result_queue.get()
                    if move:
                        self.solver.last_move = move
                        self.solver.last_eval = eval_score
                        self.output_text.append(f"{APP_NAME}: FEN: {self.solver.current_fen}")
                        self.output_text.append(f"{APP_NAME}: Best Move: {move.uci()} (Eval: {eval_score:+.2f})")
                        self.output_text.append(f"{APP_NAME}: Top Moves: {', '.join(pv)}")
                    elif error:
                        self.output_text.append(f"{APP_NAME}: Error: {error}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChessVisionProWindow()
    window.show()
    try:
        sys.exit(app.exec_())
    finally:
        if window.solver.engine:
            window.solver.engine.quit()
        if window.solver.cap:
            window.solver.cap.release()