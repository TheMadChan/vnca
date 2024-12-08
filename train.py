import tqdm
import os

from modules.model import Model


def train(model: Model, n_updates=int(1e6), eval_interval=1000):

    best = float("inf")

    os.makedirs("models", exist_ok=True)

    base_file_name = f"n{model.n_updates}_b{model.batch_size}_tb{model.test_batch_size}_z{model.z_size}_t{model.bin_threshold}_lr{model.learning_rate}_beta{model.beta}_aug{model.augment}"

    for i in tqdm.tqdm(range(n_updates)):
        model.train_batch()
        if (i + 1) % eval_interval == 0:
            loss = model.eval_batch()
            model.save(os.path.join("models", f"latest_{base_file_name}.pt"))
            if loss < best:
                best = loss
                model.save(os.path.join("models",f"best_{base_file_name}.pt"))
