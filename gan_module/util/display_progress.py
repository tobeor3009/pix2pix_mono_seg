import progressbar


class ProgressBar:
    def __init__(self, total_length):
        self.bar = progressbar.ProgressBar(maxval=total_length).start()

    def update(self, batch_i):
        self.bar.update(batch_i)
