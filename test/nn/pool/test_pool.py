from typing import Optional

import torch
from torch import Tensor

from torch_geometric.nn import (
    fps,
    knn,
    knn_graph,
    nearest,
    radius,
    radius_graph,
)
from torch_geometric.testing import onlyFullTest, withPackage


@onlyFullTest
@withPackage('pyg_lib')
def test_radius_graph_jit():
    class Net(torch.nn.Module):
        def forward(self, x: Tensor, batch: Optional[Tensor] = None) -> Tensor:
            return radius_graph(x, r=2.5, batch=batch, loop=False)

    x = torch.tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=torch.float)
    batch = torch.tensor([0, 0, 0, 0])

    model = Net()
    jit = torch.jit.script(model)
    assert model(x, batch).size() == jit(x, batch).size()

    batch = torch.tensor([0, 1, 0, 1])  # Not sorted.
    assert model(x, batch).tolist() == jit(x, batch).tolist()


def _rand_batch(sizes):
    batch = torch.arange(len(sizes)).repeat_interleave(torch.tensor(sizes))
    return batch[torch.randperm(batch.numel())]  # Not sorted.


def _brute_force_knn(x, y, k, batch_x, batch_y, exclude_self=False):
    dist = torch.cdist(y, x)
    dist[batch_y.view(-1, 1) != batch_x.view(1, -1)] = float('inf')
    if exclude_self:
        dist.fill_diagonal_(float('inf'))
    out = set()
    for i in range(y.size(0)):
        num = min(k, int(dist[i].isfinite().sum()))
        for j in dist[i].topk(num, largest=False).indices.tolist():
            out.add((i, j))
    return out


def _to_set(index):
    return {tuple(pair) for pair in index.t().tolist()}


@withPackage('pyg_lib')
def test_unsorted_batch_example():
    # Example from https://github.com/pyg-team/pytorch_geometric/issues/10254
    x = torch.tensor([[0., 1.], [1., 1.], [10., 10.], [10., 11.]])
    batch = torch.tensor([0, 1, 0, 0])  # Not sorted.

    edge_index = radius_graph(x, r=2.0, batch=batch)
    assert _to_set(edge_index) == {(3, 2), (2, 3)}

    edge_index = knn_graph(x, k=1, batch=batch)
    assert _to_set(edge_index) == {(2, 0), (3, 2), (2, 3)}


@withPackage('pyg_lib')
def test_unsorted_batch():
    torch.manual_seed(12345)
    x, batch_x = torch.randn(20, 3), _rand_batch([7, 6, 7])
    y, batch_y = torch.randn(9, 3), _rand_batch([3, 3, 3])

    # knn: (y, x) pairs.
    out = knn(x, y, 2, batch_x, batch_y)
    assert _to_set(out) == _brute_force_knn(x, y, 2, batch_x, batch_y)

    # knn_graph: (source, target) pairs.
    out = knn_graph(x, k=2, batch=batch_x)
    expected = _brute_force_knn(x, x, 2, batch_x, batch_x, exclude_self=True)
    assert _to_set(out.flip([0])) == expected

    # radius: (y, x) pairs.
    out = radius(x, y, 1.5, batch_x, batch_y)
    dist = torch.cdist(y, x)
    mask = (dist < 1.5) & (batch_y.view(-1, 1) == batch_x.view(1, -1))
    assert _to_set(out) == _to_set(mask.nonzero().t())

    # radius_graph: (source, target) pairs.
    out = radius_graph(x, r=1.5, batch=batch_x)
    dist = torch.cdist(x, x)
    mask = (dist < 1.5) & (batch_x.view(-1, 1) == batch_x.view(1, -1))
    mask.fill_diagonal_(False)
    assert _to_set(out) == _to_set(mask.nonzero().t())

    # nearest: For each element in `x`, the index of the closest `y`.
    out = nearest(x, y, batch_x, batch_y)
    dist = torch.cdist(x, y)
    dist[batch_x.view(-1, 1) != batch_y.view(1, -1)] = float('inf')
    assert out.tolist() == dist.argmin(dim=1).tolist()

    # fps: Equal to running `fps` on every example separately.
    out = fps(x, batch_x, ratio=0.5, random_start=False)
    expected = set()
    for i in range(3):
        idx = (batch_x == i).nonzero().view(-1)
        expected |= set(idx[fps(x[idx], ratio=0.5,
                                random_start=False)].tolist())
    assert set(out.tolist()) == expected


@withPackage('pyg_lib')
def test_sorted_batch():
    x = torch.tensor([[0., 1.], [10., 10.], [10., 11.], [1., 1.]])
    batch = torch.tensor([0, 0, 0, 1])  # Sorted, with a single-node example.

    edge_index = radius_graph(x, r=2.0, batch=batch)
    assert sorted(edge_index.t().tolist()) == [[1, 2], [2, 1]]

    edge_index = knn_graph(x, k=1, batch=batch)
    assert sorted(edge_index.t().tolist()) == [[1, 0], [1, 2], [2, 1]]
