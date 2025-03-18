import argparse
import time

import deepspeed
import torch
import torch.nn.functional as F
from deepspeed import comm as dist
from torchvision import datasets, transforms

from net import Net

deepspeed.init_distributed()


def train(args, model, train_loader, epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(model.local_rank), target.to(model.local_rank)
        output = model(data)
        loss = F.nll_loss(output, target)
        model.backward(loss)
        model.step()
        if dist.get_rank() == 0:
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, dist.get_world_size() * batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                if args.dry_run:
                    break


def test(model, device, test_loader):
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    if dist.get_rank() == 0:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    # Train
    parser.add_argument('--batch_size', default=64, type=int,
                        help='mini-batch size (default:64)')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--epochs', default=14, type=int,
                        help='number of total epochs (default: 14)')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    cuda_kwargs = {'num_workers': 1,
                   'pin_memory': True,
                   'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if dist.get_rank() != 0:
        # might be downloading mnist data, let rank 0 download first
        dist.barrier()

    dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)

    if dist.get_rank() == 0:
        # mnist data is downloaded, indicate other ranks can proceed
        dist.barrier()

    dataset2 = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    net = Net().to(device)
    parameters = filter(lambda p: p.requires_grad, net.parameters())

    model, optimizer, train_loader, _ = deepspeed.initialize(
        args=args, model=net, model_parameters=parameters, training_data=dataset1)

    total_time = 0.

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train(args, model, train_loader, epoch)
        total_time += time.time() - start
        test(net, device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    return total_time


if __name__ == '__main__':
    print(f'[{dist.get_rank()}] Total time elapsed: {main()} seconds')
