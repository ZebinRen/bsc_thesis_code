import matplotlib.pyplot as plt
import numpy as np
import os

def plot_train(train_loss, train_acc, val_loss, val_acc, path, name, save = False):
    '''
    Plot the train_loss, train_acc, val_loss, val_acc curve
    '''
    fig, ax1 = plt.subplots()

    epochs = list(range(len(train_loss)))

    #plot loss
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')

    #Plot accruacy
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy')

    ax1.plot(epochs, train_loss, label='train loss')
    ax1.plot(epochs, val_loss, label='validation loss')
    ax2.plot(epochs, train_acc, label='train accuracy', linestyle='dashed')
    ax2.plot(epochs, val_acc, label = 'validation accuracy', linestyle='dashed')

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])


    fig.legend(loc='lower center', ncol=4)

    if save:
        save_path_w = os.path.join(path, 'w_'+name)
        save_path_t = os.path.join(path, 't_'+name)
        plt.savefig(save_path_t, transparent=True, pad_inches=1.5)
        plt.savefig(save_path_w, transparent=False, pad_inches=1.5)
    else:
        plt.show()
    