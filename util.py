import os
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

# Format datetime to avoid characters not allowed in Windows file names
revision = os.environ.get("REVISION") or datetime.now().strftime("%m%d")
message = os.environ.get('MESSAGE') or "default_message"
tensorboard_dir = os.environ.get('TENSORBOARD_DIR') or os.path.join("tmp", "tensorboard")
flush_secs = 10

def get_writers(name, ds, n_updates, batch_size, test_batch_size, z_size, bin_threshold, learning_rate):

    revision_name = f"{revision}_{ds}_n{n_updates}_b{batch_size}_tb{test_batch_size}_z{z_size}_t{bin_threshold}_lr{learning_rate}"

    train_path = os.path.join(tensorboard_dir, name, 'tensorboard', revision_name, 'train', message)
    test_path = os.path.join(tensorboard_dir, name, 'tensorboard', revision_name, 'test', message)
    
    train_writer = SummaryWriter(train_path, flush_secs=flush_secs)
    test_writer = SummaryWriter(test_path, flush_secs=flush_secs)
    
    return train_writer, test_writer
