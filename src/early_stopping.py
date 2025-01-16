class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss
                improved.
            delta (float): Minimum change in the monitored quantity to qualify
                as an improvement.
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.stop = False
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss  # We negate the loss because we're minimizing

        if self.stop:
            return

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best_score = score
            self.counter = 0
