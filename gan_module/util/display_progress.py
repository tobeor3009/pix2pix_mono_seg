import progressbar


class ProgressBar:
    def __init__(self, total_length):
        widgets = [
            ' [', progressbar.Counter(format='%(value)02d/%(max_val)d'), '] ',
            progressbar.Bar(),
            ' (', progressbar.ETA(), ') ',
        ]
        self.bar = progressbar.ProgressBar(
            widgets=widgets,
            max_val=total_length
        ).start()

    def update(self, batch_i):
        self.bar.update(batch_i)
