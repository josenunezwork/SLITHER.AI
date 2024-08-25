from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QBrush, QColor, QPen, QRadialGradient
from PyQt5.QtCore import Qt, QTimer, QRect
import math

class GameWidget(QWidget):
    def __init__(self, game):
        super().__init__()
        self.game = game
        self.radar_animation_timer = QTimer(self)
        self.radar_animation_timer.timeout.connect(self.update_radar)
        self.radar_animation_timer.start(50)
        self.radar_phase = 0
        self.max_radar_radius = 50 

    def draw_game(self, painter):
        for snake in self.game.snakes:
            if not snake.is_alive:
                continue
            color = QColor(snake.color) if isinstance(snake.color, str) else QColor(*snake.color)
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.NoPen)
            for segment in snake.segments:
                painter.drawEllipse(segment[0] - snake.segment_size // 2, segment[1] - snake.segment_size // 2, snake.segment_size, snake.segment_size)
        for snake in self.game.snakes:
            if snake.is_alive:
                self.draw_radar(painter, snake)

        painter.setBrush(QBrush(Qt.green))
        for food in self.game.food:
            painter.drawEllipse(food[0] - 3, food[1] - 3, 6, 6)

    def draw_radar(self, painter, snake):
        head_x, head_y = snake.segments[0]
        
        current_radius = int(abs(math.sin(self.radar_phase)) * self.max_radar_radius)
        
        gradient = QRadialGradient(head_x, head_y, current_radius)
        gradient.setColorAt(0, QColor(*snake.color, 100))
        gradient.setColorAt(1, QColor(*snake.color, 0))
        
        painter.setBrush(gradient)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QRect(int(head_x - current_radius), int(head_y - current_radius), current_radius * 2, current_radius * 2))
    
    def update_radar(self):
        self.radar_phase = (self.radar_phase + 0.2) % (2 * math.pi)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        self.draw_game(painter)
