##############################################################################################
# Reference: https://spell.ml/blog/distributed-model-training-using-horovod-XvqEGRUAACgAa5th #
##############################################################################################

import argparse
import time

import horovod.torch as hvd
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed as dist
from torchvision import datasets, transforms

from net import Net

# Training settings
parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
parser.add_argument(
    "--batch-size",
    type=int,
    default=64,
    metavar="N",
    help="input batch size for training (default: 64)",
)
parser.add_argument(
    "--test-batch-size",
    type=int,
    default=1000,
    metavar="N",
    help="input batch size for testing (default: 1000)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=14,
    metavar="N",
    help="number of epochs to train (default: 14)",
)
parser.add_argument(
    "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.5,
    metavar="M",
    help="SGD momentum (default: 0.5)",
)
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disables CUDA training"
)
parser.add_argument(
    "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=10,
    metavar="N",
    help="how many batches to wait before logging training status",
)
parser.add_argument(
    "--fp16-allreduce",
    action="store_true",
    default=False,
    help="use fp16 compression during allreduce",
)
parser.add_argument(
    "--use-adasum",
    action="store_true",
    default=False,
    help="use adasum algorithm to do reduction",
)


def train(model, train_sampler, train_loader, args, optimizer, epoch):
    model.train()
    # Horovod: set epoch to sampler for shuffling.
    train_sampler.set_epoch(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if hvd.rank() == 0:
            if batch_idx % args.log_interval == 0:
                # Horovod: use train_sampler to determine the number of examples in
                # this worker's partition.
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        hvd.size() * batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )


def metric_average(val, name):
    if type(val) is not torch.Tensor:
        val = torch.tensor(val)
    avg_tensor = hvd.allreduce(val, name=name)
    return avg_tensor.item()


def test(model, test_sampler, test_loader, args):
    model.eval()
    test_loss = 0.0
    test_accuracy = 0.0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, reduction="sum").item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        test_accuracy += pred.eq(target.data.view_as(pred)).cpu().float().sum()

    # Horovod: use test_sampler to determine the number of examples in
    # this worker's partition.
    test_loss /= len(test_sampler)
    test_accuracy /= len(test_sampler)

    # Horovod: average metric values across workers.
    test_loss = metric_average(test_loss, "avg_loss")
    test_accuracy = metric_average(test_accuracy, "avg_accuracy")

    # Horovod: print output only on first rank.
    if hvd.rank() == 0:
        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n".format(
                test_loss, 100.0 * test_accuracy
            )
        )


def main():
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Horovod: initialize library.
    hvd.init()
    torch.manual_seed(args.seed)

    if args.cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)

    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(1)

    kwargs = {"num_workers": 1, "pin_memory": True} if args.cuda else {}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (
        kwargs.get("num_workers", 0) > 0
        and hasattr(mp, "_supports_context")
        and mp._supports_context
        and "forkserver" in mp.get_all_start_methods()
    ):
        kwargs["multiprocessing_context"] = "forkserver"

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    if hvd.rank() != 0:
        # might be downloading mnist data, let rank 0 download first
        hvd.barrier()

    # train_dataset = datasets.MNIST('data-%d' % hvd.rank(), train=True, download=True, transform=transform)
    train_dataset = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )

    if hvd.rank() == 0:
        # mnist data is downloaded, indicate other ranks can proceed
        hvd.barrier()

    # Horovod: use DistributedSampler to partition the training data.
    train_sampler = dist.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank()
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs
    )

    # test_dataset = datasets.MNIST('data-%d' % hvd.rank(), train=False, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, transform=transform)
    # Horovod: use DistributedSampler to partition the test data.
    test_sampler = dist.DistributedSampler(
        test_dataset, num_replicas=hvd.size(), rank=hvd.rank()
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.test_batch_size, sampler=test_sampler, **kwargs
    )

    model = Net()

    # By default, Adasum doesn't need scaling up learning rate.
    lr_scaler = hvd.size() if not args.use_adasum else 1

    if args.cuda:
        # Move model to GPU.
        model.cuda()
        # If using GPU Adasum allreduce, scale learning rate by local_size.
        if args.use_adasum and hvd.nccl_built():
            lr_scaler = hvd.local_size()

    # Horovod: scale learning rate by lr_scaler.
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr * lr_scaler, momentum=args.momentum
    )

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(
        optimizer,
        named_parameters=model.named_parameters(),
        compression=compression,
        op=hvd.Adasum if args.use_adasum else hvd.Average,
    )

    total_time = 0.0

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train(model, train_sampler, train_loader, args, optimizer, epoch)
        total_time += time.time() - start
        test(model, test_sampler, test_loader, args)

    return hvd.rank(), total_time


if __name__ == "__main__":
    rk, tt = main()
    print(f"[{rk}] Total time elapsed: {tt} seconds")
