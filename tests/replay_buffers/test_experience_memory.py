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

import pytest
import torch
from unittest.mock import MagicMock, patch
from objectrl.replay_buffers.experience_memory import ReplayBuffer


# Helper to create dummy TensorDict-like object for add and add_batch
class DummyTensorDict:
    def __init__(self, length=1):
        self.length = length

    def __len__(self):
        return self.length


@pytest.fixture
def mock_storage():
    # Mock for LazyMemmapStorage instance
    return MagicMock(name="LazyMemmapStorageInstance")


@pytest.fixture
def mock_replay_buffer(mock_storage):
    # Mock for TensorDictReplayBuffer instance
    mock_buf = MagicMock(name="TensorDictReplayBufferInstance")
    mock_buf.storage = MagicMock(name="Storage")
    mock_buf.storage.__getitem__.return_value = torch.tensor([1, 2, 3])
    mock_buf.sample.return_value = torch.tensor([1, 2, 3])
    return mock_buf


@pytest.fixture
def replay_buffer_patch(mock_storage, mock_replay_buffer):
    with (
        patch(
            "objectrl.replay_buffers.experience_memory.LazyMemmapStorage",
            return_value=mock_storage,
        ) as lazy_mock,
        patch(
            "objectrl.replay_buffers.experience_memory.TensorDictReplayBuffer",
            return_value=mock_replay_buffer,
        ) as tdrb_mock,
    ):
        yield lazy_mock, tdrb_mock, mock_storage, mock_replay_buffer


def test_init_and_reset(replay_buffer_patch):
    lazy_mock, tdrb_mock, mock_storage, mock_replay_buffer = replay_buffer_patch

    device = torch.device("cpu")
    storing_device = torch.device("cpu")
    buffer_size = 100

    buf = ReplayBuffer(device, storing_device, buffer_size)

    # Check init calls
    lazy_mock.assert_called_once_with(buffer_size, device=storing_device)
    tdrb_mock.assert_called_once_with(storage=mock_storage)

    # After reset, data_size and pointer reset
    assert buf.data_size == 0
    assert buf.pointer == 0
    assert buf.buffer_size == buffer_size


def test_add_and_add_batch_updates_data_and_pointer(replay_buffer_patch):
    _, _, _, mock_replay_buffer = replay_buffer_patch

    buf = ReplayBuffer(torch.device("cpu"), torch.device("cpu"), 5)

    # Patch memory with mocked TensorDictReplayBuffer
    buf.memory = mock_replay_buffer

    # Add a single experience
    exp = DummyTensorDict()
    buf.add(exp)
    mock_replay_buffer.add.assert_called_once_with(exp)
    assert buf.data_size == 1
    assert buf.pointer == 1

    # Add batch of 3 experiences
    batch = DummyTensorDict(length=3)
    mock_replay_buffer.extend.reset_mock()
    buf.add_batch(batch)
    mock_replay_buffer.extend.assert_called_once_with(batch)
    # data_size capped at buffer_size=5
    assert buf.data_size == 4
    # pointer should be (1 + 3) % 5 = 4
    assert buf.pointer == 4


def test_sample_batch_and_sample_random_calls_memory_sample(replay_buffer_patch):
    _, _, _, mock_replay_buffer = replay_buffer_patch

    buf = ReplayBuffer(torch.device("cpu"), torch.device("cpu"), 10)
    buf.memory = mock_replay_buffer

    mock_sample_return = MagicMock()
    mock_sample_return.to.return_value = "sampled_batch"
    mock_replay_buffer.sample.return_value = mock_sample_return

    batch = buf.sample_batch(3)
    assert batch == "sampled_batch"
    mock_replay_buffer.sample.assert_called_once_with(3)

    # sample_random is alias
    batch2 = buf.sample_random(2)
    assert batch2 == "sampled_batch"
    assert mock_replay_buffer.sample.call_count == 2  # Called again


def test_sample_by_index_and_fields(replay_buffer_patch):
    _, _, _, mock_replay_buffer = replay_buffer_patch

    buf = ReplayBuffer(torch.device("cpu"), torch.device("cpu"), 10)
    buf.memory = mock_replay_buffer

    # Prepare fake storage return value and its select method
    dummy_samples = MagicMock()
    dummy_samples.select.return_value.to.return_value = "selected_fields"
    mock_replay_buffer.storage.__getitem__.return_value = dummy_samples

    # sample_by_index with list
    indices = [0, 1, 2]
    result = buf.sample_by_index(indices)
    assert result == dummy_samples.to.return_value

    # sample_by_index with range (converted to tensor)
    result = buf.sample_by_index(range(3))
    assert result == dummy_samples.to.return_value

    # sample_by_index_fields with multiple fields
    fields = ["obs", "action"]
    result = buf.sample_by_index_fields(indices, fields)
    dummy_samples.select.assert_called_once_with(fields)
    assert result == "selected_fields"


def test_sample_all_calls_sample_by_index(replay_buffer_patch):
    _, _, _, _ = replay_buffer_patch

    buf = ReplayBuffer(torch.device("cpu"), torch.device("cpu"), 10)
    buf.data_size = 5

    with patch.object(
        buf, "sample_by_index", return_value="all_samples"
    ) as mock_sample:
        result = buf.sample_all()
        mock_sample.assert_called_once_with(range(5))
        assert result == "all_samples"


def test_len_and_size_property(replay_buffer_patch):
    _, _, _, _ = replay_buffer_patch

    buf = ReplayBuffer(torch.device("cpu"), torch.device("cpu"), 10)
    buf.data_size = 7

    assert len(buf) == 7
    assert buf.size == 7


def test_save_and_load(tmp_path, replay_buffer_patch):
    _, _, _, _ = replay_buffer_patch

    buf = ReplayBuffer(torch.device("cpu"), torch.device("cpu"), 10)
    buf.buffer_size = 10
    buf.data_size = 6
    buf.pointer = 3

    path = tmp_path / "buffer"

    # Save metadata
    buf.save(str(path))

    # Check file exists and content
    metadata_path = str(path) + ".metadata"
    loaded = torch.load(metadata_path)
    assert loaded["buffer_size"] == 10
    assert loaded["data_size"] == 6
    assert loaded["pointer"] == 3

    # Reset values and load
    buf.buffer_size = 0
    buf.data_size = 0
    buf.pointer = 0
    buf.load(str(path))
    assert buf.buffer_size == 10
    assert buf.data_size == 6
    assert buf.pointer == 3


def test_epoch_iterator_and_get_next_batch(replay_buffer_patch):
    _, _, _, mock_replay_buffer = replay_buffer_patch

    buf = ReplayBuffer(torch.device("cpu"), torch.device("cpu"), 10)
    buf.memory = mock_replay_buffer
    buf.data_size = 10

    # Setup sample_by_index to return a tensor dict batch
    batch_mock = MagicMock()
    buf.sample_by_index = MagicMock(return_value=batch_mock)

    it = buf.create_epoch_iterator(batch_size=3, n_epochs=2)
    # Should yield batch twice per epoch (10/3 rounds up to 4 batches)
    batches = list(it)
    expected_batches = 4 * 2
    assert len(batches) == expected_batches
    for b in batches:
        assert b == batch_mock

    # Assign iterator and get next batch
    buf.epoch_iterator = iter([batch_mock])
    next_batch = buf.get_next_batch(3)
    assert next_batch == batch_mock

    # If no iterator, fallback to sample_batch
    buf.epoch_iterator = None
    buf.sample_batch = MagicMock(return_value="random_batch")
    batch = buf.get_next_batch(3)
    assert batch == "random_batch"
    buf.sample_batch.assert_called_once_with(3)


def test_calculate_num_batches_and_get_steps_and_iterator(replay_buffer_patch):
    _, _, _, _ = replay_buffer_patch

    buf = ReplayBuffer(torch.device("cpu"), torch.device("cpu"), 10)
    buf.data_size = 25

    # batch size 10 => 3 batches (25 items)
    n_batches = buf.calculate_num_batches(10)
    assert n_batches == 3

    # n_epochs > 0 initializes iterator and calculates n_steps
    n_steps = buf.get_steps_and_iterator(n_epochs=2, max_iter=100, batch_size=10)
    assert n_steps == 2 * n_batches
    assert buf.epoch_iterator is not None

    # n_epochs == 0 disables iterator and returns max_iter
    n_steps = buf.get_steps_and_iterator(n_epochs=0, max_iter=42, batch_size=10)
    assert n_steps == 42
    assert buf.epoch_iterator is None
