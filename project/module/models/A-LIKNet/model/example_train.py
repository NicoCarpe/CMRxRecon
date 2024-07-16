import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import utils
from A_LIKNet_model import A_LIKNet

# specify GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def create_random_real_tensor(shape):
    return torch.randn(shape, dtype=torch.float32)

def create_random_complex_tensor(shape):
    return torch.complex(torch.randn(shape, dtype=torch.float32), torch.randn(shape, dtype=torch.float32))

def create_inputs(batch_size, nt, nx, ny, nc):
    masked_img = create_random_complex_tensor(shape=(batch_size, nt, nx, ny, 1))
    masked_kspace = create_random_complex_tensor(shape=(batch_size, nt, nx, ny, nc))
    mask = create_random_real_tensor(shape=(1, nt, 1, ny, 1))
    smaps = create_random_complex_tensor(shape=(1, 1, nx, ny, nc))

    kspace_label = create_random_complex_tensor(shape=(1, nt, nx, ny, nc))
    image_label = create_random_complex_tensor(shape=(batch_size, nt, nx, ny, 1))
    return [masked_img, masked_kspace, mask, smaps], [kspace_label, image_label]

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = A_LIKNet(num_iter=8).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = utils.get_loss('loss_complex_mae_all_dim')
    metrics = utils.get_metrics()

    # initialize model to print model summary
    inputs, targets = create_inputs(batch_size=1, nt=25, nx=192, ny=156, nc=15)
    inputs = [x.to(device) for x in inputs]
    targets = [x.to(device) for x in targets]

    model.train()
    start = time.time()
    outputs = model(*inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    end = time.time()
    
    print(f"Time taken for forward and backward pass: {end - start} seconds")
    print(model)

    # To print the model summary, we can use torchinfo
    from torchinfo import summary
    summary(model, input_data=inputs)
