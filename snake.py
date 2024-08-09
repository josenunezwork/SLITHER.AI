from game_config import GameConfig

class Snake:
    def __init__(self, id, color, start_pos, segment_size, game_width, game_height):
        self.id = id
        self.color = color
        self.segments = [start_pos]
        self.direction = (1, 0)
        self.is_alive = True
        self.length = 1
        self.segment_size = segment_size
        self.game_width = game_width
        self.game_height = game_height
        self.respawn_timer = 0

    @property
    def head(self):
        return self.segments[0]  

    def move(self):
        new_head = (
            (self.head[0] + self.direction[0] * self.segment_size) % self.game_width,
            (self.head[1] + self.direction[1] * self.segment_size) % self.game_height
        )
        self.segments.insert(0, new_head)
        if len(self.segments) > self.length:
            self.segments.pop()

    def grow(self, amount=1):
        self.length += amount

    def change_direction(self, new_direction):
        if (new_direction[0] * -1, new_direction[1] * -1) != self.direction:
            self.direction = new_direction

    def check_self_collision(self):
        return self.head in self.segments[1:]

    def die(self):
        self.is_alive = False
        self.respawn_timer = GameConfig.FRAME_RATE

    def respawn(self, new_pos):
        self.segments = [new_pos]
        self.direction = (1, 0)  # Reset direction to right
        self.is_alive = True
        self.length = 1
        self.respawn_timer = 0