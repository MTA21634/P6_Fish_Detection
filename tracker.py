# A class for a single object tracker
class tracker:
    def __init__(self, initial_object, initial_frame, ID):
        self.positions = {initial_frame: initial_object}
        self.id = ID

    def update(self, detection, frame):
        self.positions[frame] = detection

    def get_by_frame(self, frame):
        return self.positions[frame]

    def get_id(self):
        return self.id
