import torch
import gc
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
from torch.utils.tensorboard import SummaryWriter

def get_callbacks(validation_loader, model, logdir):
    return get_image_callbacks(validation_loader, model, logdir) + \
           get_model_callbacks(model, logdir) + \
           get_plotting_callbacks(logdir)

def get_image_callbacks(validation_loader, model, logdir):
    # Get a batch of validation data
    inputs, targets = next(iter(validation_loader))
    noisy = inputs if not isinstance(inputs, list) else inputs[0]
    target = targets if not isinstance(targets, list) else targets[1]

    frames, M, N = target.shape[1:-1]  # (25, 176, 132)

    def log_images(epoch, logs):
        model.eval()
        with torch.no_grad():
            prediction = model(inputs)[1]

        # Creates a file writer for the log directory.
        writer = SummaryWriter(logdir)

        # Using the writer, log the reshaped image.
        def process(x):
            return x / torch.max(x)

        writer.add_image("Validation predict xy", process(torch.abs(prediction[:, frames // 2])), epoch)
        writer.add_image("Validation noisy xy", process(torch.abs(noisy[:, frames // 2])), epoch)
        writer.add_image("Validation target xy", process(torch.abs(target[:, frames // 2])), epoch)
        writer.add_image("Validation predict xt", process(torch.abs(prediction[:, :, M // 2])), epoch)
        writer.add_image("Validation noisy xt", process(torch.abs(noisy[:, :, M // 2])), epoch)
        writer.add_image("Validation target xt", process(torch.abs(target[:, :, M // 2])), epoch)
        writer.add_image("Validation predict yt", process(torch.abs(prediction[:, :, :, N // 2])), epoch)
        writer.add_image("Validation noisy yt", process(torch.abs(noisy[:, :, :, N // 2])), epoch)
        writer.add_image("Validation target yt", process(torch.abs(target[:, :, :, N // 2])), epoch)

        writer.close()

    class GCCallback:
        def __call__(self, epoch, logs=None):
            gc.collect()

    gc_callback = GCCallback()
    img_callback = lambda epoch, logs=None: log_images(epoch, logs)
    tensorboard_callback = lambda epoch, logs=None: SummaryWriter(logdir).close()

    return [img_callback, gc_callback, tensorboard_callback]

def get_model_callbacks(model, logdir):
    # Save the model's weights
    def save_model_weights(epoch, logs=None):
        torch.save(model.state_dict(), os.path.join(logdir, f'weights{epoch:03d}.pt'))

    def optimizer_checkpoint_callback(epoch, logs=None):
        opt_weights = model.optimizer.state_dict()
        with open(f'{logdir}/optimizer.pkl', 'wb') as f:
            pickle.dump(opt_weights, f)

    cp_callback = lambda epoch, logs=None: save_model_weights(epoch, logs)
    opt_cp_callback = lambda epoch, logs=None: optimizer_checkpoint_callback(epoch, logs)

    return [cp_callback, opt_cp_callback]

class LossCallback:
    def __init__(self, logdir):
        self.logdir = logdir
        self.csv_path = f'{self.logdir}/loss.csv'
        self.keys = ['loss', 'val_loss',
                     'crop_loss_rmse', 'val_crop_loss_rmse',
                     'crop_loss_abs_mse', 'val_crop_loss_abs_mse',
                     'crop_loss_abs_mae', 'val_crop_loss_abs_mae',
                     ]

    def on_train_begin(self, logs=None):
        if os.path.exists(self.csv_path):
            self.df = pd.read_csv(self.csv_path)
        else:
            self.df = pd.DataFrame(columns=['epoch'] + self.keys)

    def on_epoch_end(self, epoch, logs):
        # create the loss dict and update dataframe
        update_dict = {'epoch': epoch}
        for key in self.keys:
            update_dict[key] = logs.get(key)
        self.df = self.df.append(update_dict, ignore_index=True)

        # save csv
        self.df.to_csv(self.csv_path, index=False)

        # Plot train & val loss
        plt.figure()
        x = np.arange(0, len(self.df))
        plt.plot(x, self.df['loss'], label="train_loss")
        plt.plot(x, self.df['val_loss'], label="val_loss")
        plt.title(f"Training/Validation Loss Epoch {epoch}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f'{self.logdir}/loss.png')
        plt.close()

def get_plotting_callbacks(logdir):
    loss_callback = LossCallback(logdir)
    return [loss_callback]
