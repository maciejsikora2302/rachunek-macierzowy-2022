class OpCount():
    def __init__(self, start):
        self.count = start
    def add(self, number):
        self.count += number
    def get(self):
        return self.count

counter = OpCount(0)