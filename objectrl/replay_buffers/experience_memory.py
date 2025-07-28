# -----------------------------------------------------------------------------------
# ObjectRL: An Object-Oriented Reinforcement Learning Codebase
# Copyright (C) 2025 ADIN Lab

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------------
import gc
import warnings
from collections.abc import Iterator

import torch
from tensordict import TensorDict
from torchrl.data import (
    LazyMemmapStorage,
    LazyTensorStorage,
    TensorDictReplayBuffer,
)


class ReplayBuffer:
    """
    A fixed-size replay buffer to store and sample experience tuples for reinforcement learning.

    This buffer uses either lazy memory mapping on the cpu or a lazy tensor on the gpu
    and TorchRL's TensorDictReplayBuffer for efficient storage and retrieval.
    """

    def __init__(
        self,
        device: torch.device,
        storing_device: torch.device,
        buffer_size: int,
        print_gc_warning: bool = False,
    ) -> None:
        """
        Initialize the ReplayBuffer.

        Args:
            device (torch.device): The device to move sampled batches to (usually the training device).
            storing_device (torch.device): The device to store the experience buffer (e.g., CPU or gpu).
                Unless memory is a concern, use `cuda` to improve runtime.
            buffer_size (int): Maximum number of experience tuples the buffer can hold.
            print_gc_warning (bool): Print a warning that the garbage collector is run

        Returns:
            None
        """
        self.device = device
        self.storing_device = storing_device
        self.buffer_size = buffer_size
        self.print_gc_warning = print_gc_warning
        self.reset()

    def _get_storage(
        self, buffer_size: int, device: torch.device
    ) -> LazyMemmapStorage | LazyTensorStorage:
        # Use LazyMemmap on CPU and LazyTensorStorage on GPU
        if device.type == "cpu":
            return LazyMemmapStorage(buffer_size, device=device)
        elif device.type == "cuda":
            return LazyTensorStorage(buffer_size, device=device)
        else:
            raise NotImplementedError(
                f"No storage support for device type: {device.type}"
            )

    def reset(self, buffer_size: int | None = None) -> None:
        """
        Reset the buffer, optionally resizing it.

        Args:
            buffer_size (int, optional): If provided, sets a new maximum buffer size.
        Returns:
            None
        """

        if buffer_size is not None:
            self.buffer_size = buffer_size

        self.data_size = 0
        self.pointer = 0

        storage = self._get_storage(self.buffer_size, self.storing_device)

        # Initialize TensorDictReplayBuffer with the storage
        self.memory = TensorDictReplayBuffer(storage=storage)

    def add(self, experience: TensorDict) -> None:
        """
        Add a single experience to the buffer.

        Args:
            experience (TensorDict): A single experience entry, usually a dictionary of tensors.
        Returns:
            None
        """
        try:
            self.memory.add(experience)
        except OSError:
            warnings.warn(
                "Failed to add experience to replay buffer, triggering a manual garbage collection",
                stacklevel=2,
            )
            gc.collect()
            self.memory.add(experience)

        self.data_size = min(self.data_size + 1, self.buffer_size)
        self.pointer = (self.pointer + 1) % self.buffer_size

    def add_batch(self, batch: TensorDict) -> None:
        """
        Add a batch of experiences to memory.

        Args:
            batch: A TensorDict containing the batch of experiences to be added.
        """
        try:
            self.memory.extend(batch)
        except OSError:
            warnings.warn(
                "Failed to add experience to replay buffer, triggering a manual garbage collection",
                stacklevel=2,
            )
            gc.collect()
            self.memory.extend(batch)

        self.data_size = min(self.data_size + len(batch), self.buffer_size)
        self.pointer = (self.pointer + len(batch)) % self.buffer_size

    def sample_batch(self, batch_size: int) -> TensorDict:
        """
        Randomly sample a batch of experiences from the buffer.

        Args:
            batch_size (int): The number of samples to draw.
        Returns:
            TensorDict: A batch of randomly sampled experiences moved to the working device.
        """
        batch = self.memory.sample(batch_size).to(self.device).clone()
        return batch

    def sample_random(self, batch_size: int) -> TensorDict:
        """
        Alias for `sample_batch`.

        Args:
            batch_size (int): The number of samples to draw.
        Returns:
            TensorDict: A batch of randomly sampled experiences.
        """
        # Same as sample_batch for the new implementation
        return self.sample_batch(batch_size)

    def sample_by_index(self, indices: list | torch.Tensor | range) -> TensorDict:
        """
        Sample specific experiences by index.

        Args:
            indices (Union[list, torch.Tensor, range]): Indices of experiences to sample.
        Returns:
            TensorDict: A batch of selected experiences moved to the working device.
        """

        if isinstance(indices, range):
            indices = torch.tensor(list(indices), device=self.storing_device)
        elif isinstance(indices, list):
            indices = torch.tensor(indices, device=self.storing_device)
        return self.memory.storage[indices].to(self.device).clone()

    def sample_by_index_fields(
        self, indices: list | torch.Tensor | range, fields: list
    ) -> TensorDict:
        """
        Sample specific fields of selected experiences by index.

        Args:
            indices (Union[list, torch.Tensor, range]): Indices of experiences to sample.
            fields (List[str]): Names of fields to retrieve (e.g., ['obs', 'action']).
        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor]]: If one field, returns a tensor.
                If multiple fields, returns a tuple of tensors.
        """
        if isinstance(indices, range):
            indices = torch.tensor(list(indices), device=self.storing_device)
        elif isinstance(indices, list):
            indices = torch.tensor(indices, device=self.storing_device)
        return self.memory.storage[indices].select(fields).to(self.device).clone()

    def sample_all(self) -> TensorDict:
        """
        Sample all experiences currently stored in the buffer.

        Returns:
            TensorDict: All stored experiences up to `data_size`.
        """
        return self.sample_by_index(range(self.data_size))

    def __len__(self) -> int:
        """
        Get the number of experiences currently stored in the buffer.

        Args:
            None
        Returns:
            int: The current number of stored experiences.
        """
        return self.data_size

    @property
    def size(self) -> int:
        """
        Property alias for the current number of stored experiences.

        Args:
            None
        Returns:
            int: Current buffer size (number of items stored).
        """
        return self.data_size

    def save(self, path: str) -> None:
        """
        Save memory metadata to a file.

        Args:
            path (str): Path to save the metadata file.
        Returns:
            None
        """
        metadata = {
            "buffer_size": self.buffer_size,
            "data_size": self.data_size,
            "pointer": self.pointer,
        }
        torch.save(metadata, path + ".metadata")

    def load(self, path: str) -> None:
        """
        Load buffer metadata from disk (does not restore experience data).

        Args:
            path (str): The file path (excluding file extension) to load metadata from.
        Returns:
            None
        """
        metadata = torch.load(path + ".metadata")
        self.buffer_size = metadata["buffer_size"]
        self.data_size = metadata["data_size"]
        self.pointer = metadata["pointer"]

    def create_epoch_iterator(self, batch_size: int, n_epochs: int = 1) -> Iterator:
        """
        Create an iterator that yields batches from the buffer for multiple epochs.

        Args:
            batch_size (int): Number of samples per batch.
            n_epochs (int): Number of epochs to iterate through the data.
        Returns:
            Iterator: An iterator that yields batches of experience.
        """
        total_samples = self.data_size

        def batch_generator():
            for _ in range(n_epochs):
                # Create indices for the entire dataset
                indices = torch.arange(total_samples, device=self.storing_device)

                # Yield batches
                for i in range(0, total_samples, batch_size):
                    batch_indices = indices[i : min(i + batch_size, total_samples)]
                    yield self.sample_by_index(batch_indices)

        self.epoch_iterator = batch_generator()
        return self.epoch_iterator

    def get_next_batch(self, batch_size: int) -> TensorDict:
        """
        Retrieve the next batch from the current epoch iterator.
        Falls back to random sampling if the iterator is not initialized.

        Args:
            batch_size (int): Batch size to use for fallback random sampling.
        Returns:
            TensorDict: A batch of experience data.
        """

        if self.epoch_iterator is not None:
            return next(self.epoch_iterator)
        return self.sample_batch(batch_size)

    def calculate_num_batches(self, batch_size: int) -> int:
        """
        Calculate how many full batches can be drawn from the current buffer content.

        Args:
            batch_size (int): Number of samples per batch.
        Returns:
            int: Total number of batches possible with current data size.
        """
        return (self.data_size + batch_size - 1) // batch_size

    def get_steps_and_iterator(
        self, n_epochs: int, max_iter: int, batch_size: int
    ) -> int:
        """
        Compute total training steps and initialize an internal batch iterator.

        Args:
            n_epochs (int): Number of training epochs. If > 0, iterator will be used.
            max_iter (int): Number of learning updates to perform (used if n_epochs = 0).
            batch_size (int): Number of samples per training step.
        Returns:
            int: Total number of training steps.
        """
        if n_epochs > 0:
            n_batches = self.calculate_num_batches(batch_size)
            n_steps = n_epochs * n_batches
            # Initialize the internal iterator
            self.create_epoch_iterator(batch_size, n_epochs)
        else:
            n_steps = max_iter
            self.epoch_iterator = None

        return n_steps
