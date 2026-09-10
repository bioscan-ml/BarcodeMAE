r"""
Data samplers.
"""

import math

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler


__all__ = [
    "BalancedSampler",
    "DistributedBalancedSampler",
    "KClassMSampleSampler",
    "DistributedKClassMSampleSampler",
    "get_dataset_labels",
    "get_metadata",
]


def get_dataset_labels(dataset, column=None):
    r"""
    Get the class labels within a :class:`torch.utils.data.Dataset` object.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The dataset object.

    Returns
    -------
    array_like or None
        The class labels for each sample.
    """
    if isinstance(dataset, torch.utils.data.Subset):
        # For a dataset subset, we need to get the full set of labels from the
        # interior subset and then reduce them down to just the labels we have
        # in the subset.
        labels = get_dataset_labels(dataset.dataset, column=column)
        if labels is None:
            return labels
        return np.array(labels)[dataset.indices]

    if isinstance(dataset, torch.utils.data.ConcatDataset):
        # For a concat dataset, we need to get the labels from each of the
        # interior datasets and then concatenate them together.
        labels = [get_dataset_labels(d, column=column) for d in dataset.datasets]
        for labels_i in labels:
            if labels_i is None:
                return None
        return np.concatenate(labels)

    if column is None:
        column = "age_group"
    labels = None
    if hasattr(dataset, "targets"):
        # MNIST, CIFAR, ImageFolder, etc
        labels = dataset.targets
    elif hasattr(dataset, "labels"):
        # STL10, SVHN
        labels = dataset.labels
    elif hasattr(dataset, "_labels"):
        # Flowers102
        labels = dataset._labels
    elif hasattr(dataset, "metadata") and column in dataset.metadata:
        # HiddenDataset, IMDbWikiFaces
        labels = dataset.metadata[column].values

    return labels


def get_metadata(dataset):
    """
    Get the metadata for a dataset.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The dataset object.

    Returns
    -------
    pd.DataFrame or None
        The metadata for the dataset.
    """
    if isinstance(dataset, torch.utils.data.Subset):
        # For a dataset subset, we need to get the full set of labels from the
        # interior subset and then reduce them down to just the labels we have
        # in the subset.
        metadata = get_metadata(dataset.dataset)
        if metadata is None:
            return metadata
        return metadata.iloc[dataset.indices]

    if isinstance(dataset, torch.utils.data.ConcatDataset):
        # For a concat dataset, we need to get the labels from each of the
        # interior datasets and then concatenate them together.
        metadata = [get_metadata(d) for d in dataset.datasets]
        for metadata_i in metadata:
            if metadata_i is None:
                return None
        return pd.concat(metadata)

    if hasattr(dataset, "metadata"):
        return dataset.metadata

    return None


class BalancedSampler(Sampler):
    r"""
    Balanced sampler that presents each class in the dataset equally often.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset that will be sampled from.
    num_samples : int, optional
        Number of samples to draw from the dataset. If not specified, the
        number of samples is equal to the size of the largest class times the
        number of classes, to ensure each sample is presented once.
    shuffle : bool, default=True
        Whether to present the samples in a random order.
    seed : int, optional
        Random seed to use for the generator that shuffles the samples.
        If not specified, a random seed is generated from the global RNG
        each time the iterator is instantiated.
    cycle : bool, default=True
        Whether to cycle through the samples in a locally balanced way.
        If ``True``, each batch will be balanced. If ``False``, each batch
        will be unbalanced, but the dataset as a whole will be balanced.
    """

    def __init__(self, dataset, num_samples=None, shuffle=True, seed=None, cycle=True, column=None):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.cycle = cycle
        if not self.shuffle and not self.cycle:
            raise ValueError("If shuffle is False, cycle must be True.")

        if self.seed is None:
            self.generator = None
        else:
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)

        labels = get_dataset_labels(dataset, column=column)
        if labels is None:
            raise ValueError("Dataset must have labels to perform balanced sampling.")
        # Use numpy to convert non-numeric objects into an ID number
        labels = np.unique(labels, return_inverse=True)[1]
        # Use torch for the rest
        labels = torch.as_tensor(labels)
        u_labels = torch.unique(labels)
        self.label2indices = {}
        max_num_per_label = 0
        for label in u_labels:
            self.label2indices[label] = torch.nonzero(labels == label).squeeze(-1)
            max_num_per_label = max(max_num_per_label, len(self.label2indices[label]))
        if num_samples is None:
            num_samples = max_num_per_label * len(u_labels)
        self.num_samples = num_samples

    def get_generator(self):
        if self.generator is not None:
            return self.generator
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator()
        generator.manual_seed(seed)
        return generator

    def get_indices(self, num_samples=None):
        if num_samples is None:
            num_samples = self.num_samples
        num_classes = len(self.label2indices)
        num_per_class = int(math.ceil(num_samples / num_classes))
        indices = torch.empty((num_classes, num_per_class), dtype=torch.long)
        if self.shuffle:
            generator = self.get_generator()

        if not self.shuffle:
            # Fill indices with repetitions of the indices of each class
            # in the order they were in the dataset originally.
            for i_label, idx in enumerate(self.label2indices.values()):
                L = len(idx)
                n_rep = int(math.ceil(num_per_class / L))
                indices[i_label] = idx.repeat(n_rep)[:num_per_class]
            # Flatten the indices into a vector.
            indices = indices.transpose(0, 1).reshape(-1)

        elif not self.cycle:
            # Fill indices with repetitions of the indices of each class
            # in the order they were in the dataset originally, except the last
            # repetition which we order at random so the samples which are
            # dropped are randomly selected.
            for i_label, idx in enumerate(self.label2indices.values()):
                L = len(idx)
                n_rep = int(math.ceil(num_per_class / L))
                indices[i_label, : (n_rep - 1) * L] = idx.repeat(n_rep - 1)
                indices[i_label, (n_rep - 1) * L :] = idx[torch.randperm(L, generator=generator)][
                    : num_per_class - (n_rep - 1) * L
                ]
            # Flatten the indices into a vector.
            indices = indices.transpose(0, 1).reshape(-1)
            # Shuffle the indices of everything, so there is only balance on
            # aggregate but no balance within minibatches.
            indices = indices[torch.randperm(len(indices), generator=generator)]

        else:
            # Fill indices with repetitions of the indices of each class
            # in the order they were in the dataset originally, except the last
            # repetition which we order at random so the samples which are
            # dropped are randomly selected.
            for i_label, idx in enumerate(self.label2indices.values()):
                L = len(idx)
                n_rep = int(math.ceil(num_per_class / L))
                for i_rep in range(n_rep - 1):
                    # Shuffle every time we add a new repetition of the indices
                    # so each class has its own "inner epoch" where samples all
                    # appear once in a random order before we start repeating
                    # samples.
                    indices[i_label, i_rep * L : (i_rep + 1) * L] = idx[torch.randperm(L, generator=generator)]
                indices[i_label, (n_rep - 1) * L :] = idx[torch.randperm(L, generator=generator)][
                    : num_per_class - (n_rep - 1) * L
                ]
            # Flip the axes over before shuffling further.
            indices = indices.transpose(0, 1)
            # Now each column is a class, and each row is the i-th sample to
            # appear within the class. By walking along a row, we see each
            # sample once before progressing to next row where there is another
            # appearance of each sample. We want to shuffle the order of the
            # classes within each row, so they appear in a random order, but
            # no class gets ahead of the others by more than one sample.
            order = torch.argsort(torch.rand(indices.shape), dim=-1)
            indices = torch.gather(indices, dim=-1, index=order)
            # Flatten the indices into a vector.
            indices = indices.reshape(-1)

        # Trim down to the desired number of samples.
        indices = indices[:num_samples]
        return indices

    def __iter__(self):
        indices = self.get_indices()
        assert len(indices) == self.num_samples
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples


class DistributedBalancedSampler(BalancedSampler):
    r"""
    Balanced sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`DistributedBalancedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size and that any instance of it always
        returns the same elements in the same order.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset that will be sampled from.
    num_samples : int, optional
        Number of samples to draw from the dataset across all replicas.
        If not specified, the number of samples is equal to the size of the
        largest class times the number of classes, to ensure each sample is
        presented once.
    shuffle : bool, default=True
        Whether to present the samples in a random order.
    cycle : bool, default=True
        Whether to cycle through the samples in a locally balanced way.
        If ``True``, each batch will be balanced. If ``False``, each batch
        will be unbalanced, but the dataset as a whole will be balanced.
    seed : int, default=0
        Random seed to use for the generator that shuffles the samples.
        This seed will be offset by the epoch number, which is incremented each
        time the iterator is instantiated. The epoch number can be manually
        set using :meth:`set_epoch`.
    num_replicas : int, optional
        Number of processes participating in distributed training. By default,
        :attr:`world_size` is retrieved from the current distributed group.
    rank : int, optional
        Rank of the current process within :attr:`num_replicas`. By default,
        :attr:`rank` is retrieved from the current distributed group.
    drop_last : bool, default=False
        If ``True``, then the sampler will drop the remainder ``num_samples``
        to make it evenly divisible across the number of replicas.
        If ``False``, the sampler will add extra indices to ``num_samples`` to
        make it evenly divisible across the replicas.
    """

    def __init__(self, *args, num_replicas=None, rank=None, drop_last=False, seed=0, **kwargs):
        super().__init__(*args, seed=seed, **kwargs)
        self.generator = None

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # From our call to super init, num_samples is currently the total
        # number of samples per epoch across all GPUs.
        if self.drop_last:
            # Round down to nearest available length that is evenly divisible
            # by the number of GPUs. This is to ensure each rank receives the
            # same amount of data when using this Sampler.
            self.num_samples_per_gpu = self.num_samples // self.num_replicas
        else:
            # Round up to a number of samples that is evenly divisible by the
            # number of GPUs instead.
            self.num_samples_per_gpu = math.ceil(self.num_samples / self.num_replicas)

        self.num_samples = self.num_samples_per_gpu * self.num_replicas
        self.total_size = self.num_samples
        self.seed = seed

    def get_generator(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        return generator

    def __iter__(self):
        indices = self.get_indices()
        indices = indices[self.rank : self.num_samples : self.num_replicas]
        assert len(indices) == self.num_samples_per_gpu
        self.epoch = self.epoch + 1
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples_per_gpu

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this
        ensures all replicas use a different random ordering for each epoch.

        Parameters
        ----------
        epoch : int
            Epoch number.
        """
        self.epoch = epoch


class KClassMSampleSampler(Sampler):
    r"""
    Hybrid sampler for self-supervised + contrastive pretraining.

    Each batch of size ``batch_size`` contains two parts:

    1. **Balanced labeled block** (k * m indices): k taxonomy classes are chosen
       at random and m samples are drawn per class.  This guarantees
       k * C(m, 2) positive pairs for the auxiliary taxonomy loss.

    2. **Random fill** (batch_size - k * m indices): remaining slots are filled
       with uniform random samples from the *full* dataset, including those
       with unknown labels (-1).  These samples contribute to the MAE / self-
       supervised loss just like any other sample.

    The taxonomy loss (``compute_taxonomy_classification_loss``) already filters
    out -1-labelled entries via ``create_taxonomy_pairs``, so no special handling
    is needed in the training loop.

    Parameters
    ----------
    labels : array-like of shape (N,)
        Integer class label for each sample.  Values < 0 mean "no label";
        these samples are excluded from the balanced block but can appear in
        the random fill.
    k : int
        Number of distinct classes in the balanced block per batch.
    m : int
        Samples per class in the balanced block.  k * m <= batch_size.
    batch_size : int
        Total number of samples per batch (= batch_size_per_gpu).
    num_batches : int, optional
        Batches per epoch.  The default depends on ``cover_full_dataset``.
    cover_full_dataset : bool, default=True
        Controls how ``num_batches`` is chosen when it is not set explicitly.

        ``True``  — set ``num_batches = ceil(N / n_random)`` so that the fill
        pool cycles through **all** N dataset indices at least once per epoch.
        This guarantees every sample (including unlabelled ones) is seen in the
        fill each epoch, at the cost of a longer epoch when ``n_random`` is
        small relative to ``N``.

        ``False`` — set ``num_batches = ceil(N / batch_size)``, matching the
        standard epoch length.  Only a fraction ``n_random / batch_size`` of
        the dataset appears in the fill; the rest must be covered by the
        labeled block or will be missed entirely for that epoch.
    shuffle : bool, default=True
        Randomly select classes and samples each epoch.
    seed : int, optional
        Fixed random seed.  If None a fresh seed is drawn on each __iter__.
    """

    def __init__(
        self, labels, k, m, batch_size, num_batches=None, cover_full_dataset=True, shuffle=True, seed=None
    ):
        if k * m > batch_size:
            raise ValueError(
                f"k * m = {k} * {m} = {k * m} must be <= batch_size={batch_size}."
            )
        self.k = k
        self.m = m
        self.batch_size = batch_size
        self.n_labeled = k * m
        self.n_random = batch_size - k * m
        self.shuffle = shuffle
        self.seed = seed

        labels = np.asarray(labels)
        N = len(labels)
        self.N = N

        # Per-class index lists (valid labels only, for the balanced block)
        valid_mask = labels >= 0
        valid_global = np.where(valid_mask)[0]
        valid_labels = labels[valid_mask]

        unique_labels = np.unique(valid_labels)
        self.label2indices = {}
        for lbl in unique_labels:
            idx = torch.from_numpy(valid_global[valid_labels == lbl])
            self.label2indices[int(lbl)] = idx
        self._class_keys = list(self.label2indices.keys())
        self.num_classes = len(self._class_keys)

        if k > self.num_classes:
            raise ValueError(
                f"k={k} > num_classes={self.num_classes}. "
                "Reduce --k-classes or use a dataset with more distinct classes."
            )

        if num_batches is None:
            if self.n_random > 0 and cover_full_dataset:
                # Enough batches so the fill pool covers all N samples at least once
                num_batches = math.ceil(N / self.n_random)
            else:
                # Standard: one pass through the dataset by batch count
                num_batches = math.ceil(N / batch_size)
        self.num_batches = num_batches
        self.num_samples = num_batches * batch_size

        self._fixed_generator = None
        if seed is not None:
            self._fixed_generator = torch.Generator()
            self._fixed_generator.manual_seed(seed)

    def _make_generator(self):
        if self._fixed_generator is not None:
            return self._fixed_generator
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        g = torch.Generator()
        g.manual_seed(seed)
        return g

    def get_indices(self):
        r"""
        Build all indices for one epoch as a flat tensor of shape (num_samples,).

        Within each consecutive block of ``batch_size`` indices:

        - First ``k * m`` are the balanced labeled samples
          (m indices per class, k classes, all valid labels ≥ 0).
        - Remaining ``batch_size - k * m`` are the *fill* slots drawn from a
          shuffled permutation of **all** N dataset indices (including -1-labelled
          samples).  The permutation is cycled as needed so every sample appears
          in the fill at least once per epoch, with no sample repeated until the
          full dataset has been exhausted.
        """
        g = self._make_generator()

        # Pre-build the fill pool as one or more full shuffled permutations so
        # that every sample is guaranteed to appear in the fill at least once.
        # Using a permutation (without replacement within each cycle) ensures
        # unlabelled samples are not skipped.
        if self.n_random > 0:
            total_fill = self.num_batches * self.n_random
            fill_parts = []
            remaining = total_fill
            while remaining > 0:
                if self.shuffle:
                    perm = torch.randperm(self.N, generator=g)
                else:
                    perm = torch.arange(self.N)
                take = min(remaining, self.N)
                fill_parts.append(perm[:take])
                remaining -= take
            fill_pool = torch.cat(fill_parts)  # shape (total_fill,)
        else:
            fill_pool = None

        all_blocks = []
        fill_offset = 0

        for _ in range(self.num_batches):
            # --- balanced labeled block ---
            if self.shuffle:
                chosen = torch.randperm(self.num_classes, generator=g)[: self.k]
            else:
                chosen = torch.arange(self.k)

            labeled_part = []
            for ci in chosen:
                key = self._class_keys[ci.item()]
                idx = self.label2indices[key]
                n = len(idx)
                if n >= self.m:
                    perm = torch.randperm(n, generator=g)[: self.m] if self.shuffle else torch.arange(self.m)
                    labeled_part.append(idx[perm])
                else:
                    # Oversample with replacement for small classes
                    rep = torch.randint(n, (self.m,), generator=g) if self.shuffle else torch.arange(self.m) % n
                    labeled_part.append(idx[rep])

            # --- fill from pre-shuffled pool, excluding already-selected labeled indices ---
            if fill_pool is not None:
                labeled_set = {idx.item() for t in labeled_part for idx in t}
                fill_slice = fill_pool[fill_offset : fill_offset + self.n_random]
                fill_offset += self.n_random

                fill_list = [x.item() for x in fill_slice if x.item() not in labeled_set]

                # Patch any slots removed due to overlap with the labeled block
                if len(fill_list) < self.n_random:
                    fill_set = set(fill_list)
                    for c in torch.randperm(self.N, generator=g).tolist():
                        if c not in labeled_set and c not in fill_set:
                            fill_list.append(c)
                            fill_set.add(c)
                            if len(fill_list) == self.n_random:
                                break

                fill = torch.tensor(fill_list, dtype=torch.long)
                all_blocks.append(torch.cat(labeled_part + [fill]))
            else:
                all_blocks.append(torch.cat(labeled_part))

        return torch.cat(all_blocks)  # shape (num_samples,)

    def __iter__(self):
        return iter(self.get_indices().tolist())

    def __len__(self) -> int:
        return self.num_samples


class DistributedKClassMSampleSampler(KClassMSampleSampler):
    r"""
    Distributed version of KClassMSampleSampler for DDP training.

    Each rank independently samples its own balanced batches from the full
    class pool.  Ranks use different seeds so they draw different samples,
    but every local batch on every GPU is guaranteed to be class-balanced.

    Compatible with set_epoch() so the training loop in pretraining.py keeps
    working unchanged.

    Parameters
    ----------
    labels : array-like of shape (N,)
        Integer class label for each sample.  Values < 0 are excluded from
        the balanced block but may appear in the random fill.
    k : int
        Number of classes per batch (balanced block).
    m : int
        Samples per class per batch.
    batch_size : int
        Total samples per local batch (= batch_size_per_gpu).
    num_replicas : int, optional
        World size.  Auto-detected if not given.
    rank : int, optional
        This process's rank.  Auto-detected if not given.
    seed : int, default=0
        Base seed.  Actual seed = seed + epoch * 100003 + rank.
    drop_last : bool, default=False
        Unused; kept for API compatibility with DistributedSampler.
    **kwargs
        Forwarded to KClassMSampleSampler (num_batches, shuffle, etc.).
    """

    def __init__(
        self, labels, k, m, batch_size, num_replicas=None, rank=None, seed=0, drop_last=False, **kwargs
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires the distributed package to be available.")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires the distributed package to be available.")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, expected [0, {num_replicas - 1}].")

        # Compute per-rank num_batches before calling super().__init__.
        # Each rank gets 1/world_size of the total batches so that collectively
        # all ranks together cover N samples once per epoch.
        n_random = batch_size - k * m
        if "num_batches" not in kwargs or kwargs["num_batches"] is None:
            cover = kwargs.get("cover_full_dataset", True)
            N = len(labels)
            if n_random > 0 and cover:
                total_batches = math.ceil(N / n_random)
            else:
                total_batches = math.ceil(N / batch_size)
            kwargs["num_batches"] = math.ceil(total_batches / num_replicas)

        super().__init__(labels, k, m, batch_size, seed=seed, **kwargs)
        self._fixed_generator = None

        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.base_seed = seed

    def _make_labeled_generator(self):
        """Rank-specific seed so each GPU draws different labeled samples."""
        g = torch.Generator()
        g.manual_seed(self.base_seed + self.epoch * 100003 + self.rank)
        return g

    def _make_fill_generator(self):
        """Shared seed across all ranks so the fill permutation is coordinated."""
        g = torch.Generator()
        g.manual_seed(self.base_seed + self.epoch * 100003)
        return g

    def get_indices(self):
        r"""
        Build indices for one epoch on this rank.

        Labeled block: rank-specific seed → each GPU draws different balanced
        samples, maximising variety of positive pairs across GPUs.

        Fill pool: shared seed across all ranks → one global permutation of all
        N indices is generated, then each rank takes every ``num_replicas``-th
        element (interleaved striding).  This guarantees non-overlapping fill
        coverage: collectively the ranks cover every sample exactly once per
        epoch with no duplicates across GPUs.
        """
        g_labeled = self._make_labeled_generator()
        g_fill = self._make_fill_generator()

        # --- Build this rank's slice of the shared fill pool ---
        if self.n_random > 0:
            total_fill = self.num_batches * self.n_random * self.num_replicas
            fill_parts = []
            remaining = total_fill
            while remaining > 0:
                perm = torch.randperm(self.N, generator=g_fill)
                take = min(remaining, self.N)
                fill_parts.append(perm[:take])
                remaining -= take
            global_fill = torch.cat(fill_parts)           # (total_fill,)
            # Each rank takes its interleaved slice
            rank_fill = global_fill[self.rank :: self.num_replicas]  # (num_batches * n_random,)
        else:
            rank_fill = None

        all_blocks = []
        fill_offset = 0

        for _ in range(self.num_batches):
            # Labeled block — rank-specific
            if self.shuffle:
                chosen = torch.randperm(self.num_classes, generator=g_labeled)[: self.k]
            else:
                chosen = torch.arange(self.k)

            labeled_part = []
            for ci in chosen:
                key = self._class_keys[ci.item()]
                idx = self.label2indices[key]
                n = len(idx)
                if n >= self.m:
                    perm = torch.randperm(n, generator=g_labeled)[: self.m] if self.shuffle else torch.arange(self.m)
                    labeled_part.append(idx[perm])
                else:
                    rep = (
                        torch.randint(n, (self.m,), generator=g_labeled) if self.shuffle else torch.arange(self.m) % n
                    )
                    labeled_part.append(idx[rep])

            if rank_fill is not None:
                labeled_set = {idx.item() for t in labeled_part for idx in t}
                fill_slice = rank_fill[fill_offset : fill_offset + self.n_random]
                fill_offset += self.n_random

                fill_list = [x.item() for x in fill_slice if x.item() not in labeled_set]

                # Patch any slots removed due to overlap with the labeled block
                if len(fill_list) < self.n_random:
                    fill_set = set(fill_list)
                    for c in torch.randperm(self.N, generator=g_labeled).tolist():
                        if c not in labeled_set and c not in fill_set:
                            fill_list.append(c)
                            fill_set.add(c)
                            if len(fill_list) == self.n_random:
                                break

                fill = torch.tensor(fill_list, dtype=torch.long)
                all_blocks.append(torch.cat(labeled_part + [fill]))
            else:
                all_blocks.append(torch.cat(labeled_part))

        return torch.cat(all_blocks)

    def __iter__(self):
        indices = self.get_indices()
        self.epoch += 1
        return iter(indices.tolist())

    def set_epoch(self, epoch: int) -> None:
        r"""
        Set epoch for reproducible per-epoch shuffling.  Called automatically
        by pretraining.py at the start of each epoch.
        """
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_samples