import matplotlib.pyplot as plt


class MetricTracker:
    def __init__(self):
        self.train_loss = []
        self.val_loss = []

        self.train_acc = []
        self.val_acc = []

    def append(self, val, categ):
        if categ == 'train_loss':
            self.train_loss.append(val)
        elif categ == 'val_loss':
            self.val_loss.append(val)
        elif categ == 'train_acc':
            self.train_acc.append(val)
        elif categ == 'val_acc':
            self.val_acc.append(val)

    def plot_loss(self):
        plt.figure()
        plt.title('Loss')
        plt.plot(self.train_loss, label='Train Loss')
        plt.plot(self.val_loss, label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def plot_acc(self):
        plt.figure()
        plt.title('Accuracy')
        plt.plot(self.train_acc, label='Train Acc')
        plt.plot(self.val_acc, label='Val Acc')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()
