"""Memory-optimized active learning with Optuna hyperparameter search.

The script performs round-based active learning with BADGE and supports global
or class-wise acquisition, random-acquisition baselines, and ten-seed repeated
experiments. Hyperparameters can be selected once with ``--weight_search`` and
reused by subsequent runs.

Inputs:
    Memory-mapped training, validation, and test arrays.

Outputs:
    Per-round model weights, checkpoints, experiment configuration files,
    per-round metrics, and aggregated ten-seed learning curves.

Common arguments:
    --rounds --initial_k --select_k --select_way --select_model --seed
    --ten_times --save_dir --weight_search --n_trials --param_path

Examples:
    Search hyperparameters:
        python active_learning_optuna_3.8.py --weight_search --select_way class \
            --rounds 20 --seed 42 --save_dir ./exp_class

    Load existing hyperparameters:
        python active_learning_optuna_3.8.py --select_way class --rounds 20 \
            --seed 42 --save_dir ./exp_class \
            --param_path al_optuna/Round1/best_model_run1.pth
"""
import os
os.environ['MPLBACKEND'] = 'Agg'  # Configure the non-interactive backend before importing Matplotlib.
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
import json
from copy import deepcopy
import gc
import psutil
import optuna
import warnings
warnings.filterwarnings('ignore')

DEFAULT_OPTUNA_DIR = "al_optuna"
DEFAULT_OPTUNA_PARAM_PATH = os.path.join(
    DEFAULT_OPTUNA_DIR, "Round1", "best_model_run1.pth"
)

# ============================ Memory Monitoring ============================
def print_memory_usage(message=""):
    """Print the resident memory used by the current process."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"{message} - Memory usage: {mem_info.rss / 1024 ** 2:.2f} MB")

# =============================== Utilities ================================
def evaluate(model, data_loader, criterion, device, val=False):
    """
    Evaluate model on a DataLoader. Returns accuracy, precision (macro), recall (macro), f1 (macro), preds, labels.
    """
    model.eval()
    preds_all, labels_all = [], []
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y.argmax(dim=1))
            total_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y.argmax(dim=1)).sum().item()
            total += y.size(0)
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(y.argmax(dim=1).cpu().numpy())

    acc = correct / total if total > 0 else 0.0
    precision = precision_score(labels_all, preds_all, average='macro', zero_division=0) if labels_all else 0.0
    recall = recall_score(labels_all, preds_all, average='macro', zero_division=0) if labels_all else 0.0
    f1 = f1_score(labels_all, preds_all, average='macro', zero_division=0) if labels_all else 0.0
    loss = total_loss / total if total > 0 else 0.0

    if not val:
        print(f"Eval Acc: {acc:.4f}  Precision(macro): {precision:.4f}  Recall(macro): {recall:.4f}  F1(macro): {f1:.4f}  Loss: {loss:.4f}")

    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return acc, precision, recall, f1, preds_all, labels_all


class EarlyStopping:
    def __init__(self, patience=10, delta=0.001):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_acc):
        if self.best_score is None:
            self.best_score = val_acc
        elif val_acc < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_acc
            self.counter = 0
        return self.early_stop


# ======================= Memory-Optimized BADGE Query ======================
def badge_distance_batch(X1, X2, mu):
    """Compute BADGE distances in batches to limit peak memory usage.

    Args:
        X1: Tuple containing modified probabilities and their squared norms.
        X2: Tuple containing feature embeddings and their squared norms.
        mu: Current center represented as
            ``((p_c, ||p_c||^2), (h_c, ||h_c||^2))``.

    Returns:
        A one-dimensional array of distances from all samples to ``mu``.
    """
    Y1, Y2 = mu
    X1_vec, X1_norm_square = X1
    X2_vec, X2_norm_square = X2
    Y1_vec, Y1_norm_square = Y1
    Y2_vec, Y2_norm_square = Y2
    
    # Process samples in batches to limit peak memory usage.
    batch_size = 10000
    n_samples = len(X1_norm_square)
    distances = np.zeros(n_samples, dtype=np.float32)
    
    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        
        X1_batch = X1_vec[i:end]
        X2_batch = X2_vec[i:end]
        X1_norm_batch = X1_norm_square[i:end]
        X2_norm_batch = X2_norm_square[i:end]
        
        # Compute squared distances.
        dist_sq = (
            X1_norm_batch * X2_norm_batch
            + Y1_norm_square * Y2_norm_square
            - 2.0 * (X1_batch @ Y1_vec) * (X2_batch @ Y2_vec)
        )
        
        # Clamp small negative values introduced by floating-point error.
        dist_sq = np.clip(dist_sq, a_min=0.0, a_max=None)
        distances[i:end] = np.sqrt(dist_sq)
    
    return distances


def badge_init_centers_optimized(X1, X2, chosen, chosen_list, mu, D2):
    """Select and register the next center for BADGE k-means++ seeding."""
    if len(chosen) == 0:
        # Use the sample with the largest gradient norm as the first center.
        probs_mod, prob_norms_square = X1
        embs, emb_norms_square = X2
        ind = np.argmax(prob_norms_square * emb_norms_square)
        mu = [((probs_mod[ind], prob_norms_square[ind]),
               (embs[ind], emb_norms_square[ind]))]
        D2 = badge_distance_batch(X1, X2, mu[0]).astype(np.float32)
        D2[ind] = 0.0
    else:
        # Update each sample's distance to its nearest selected center.
        newD = badge_distance_batch(X1, X2, mu[-1]).astype(np.float32)
        D2 = np.minimum(D2, newD)
        D2[chosen_list] = 0.0
        
        # Sample the next center from the squared-distance distribution.
        D2_sq = D2 ** 2
        total = np.sum(D2_sq)
        if total > 0:
            probs = D2_sq / total
            ind = np.random.choice(len(probs), p=probs)
        else:
            # Fall back to a random unselected sample when all distances are zero.
            available = np.setdiff1d(np.arange(len(D2)), chosen_list)
            ind = np.random.choice(available) if len(available) > 0 else 0
        
        probs_mod, prob_norms_square = X1
        embs, emb_norms_square = X2
        mu.append(((probs_mod[ind], prob_norms_square[ind]),
                   (embs[ind], emb_norms_square[ind])))
    
    chosen.add(ind)
    chosen_list.append(ind)
    
    # Periodically release unused host and device memory.
    if len(chosen) % 10 == 0:
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return chosen, chosen_list, mu, D2


def badge_selection_optimized(train_xrd, model, candidate_indices, k, device, batch_size=256):
    """Select candidate samples with the memory-optimized BADGE routine."""
    if len(candidate_indices) == 0:
        return []
    
    k = min(k, len(candidate_indices))
    
    model.eval()
    
    # Partition candidate indices into manageable extraction batches.
    candidate_batches = []
    for i in range(0, len(candidate_indices), 5000):  # Use smaller candidate chunks to reduce peak memory usage.
        end_idx = min(i + 5000, len(candidate_indices))
        batch_indices = candidate_indices[i:end_idx]
        candidate_batches.append(batch_indices)
    
    embs_list = []
    probs_list = []
    
    with torch.no_grad():
        for batch_indices in tqdm(candidate_batches, desc="Extracting features", leave=False):
            cand_np = train_xrd[batch_indices]
            cand_tensor = torch.tensor(cand_np, dtype=torch.float32).permute(0, 2, 1)
            
            # Use a smaller inference batch within each candidate chunk.
            sub_batch_size = min(batch_size, 64)  # Cap the inference batch size at 64.
            dataset = TensorDataset(cand_tensor, torch.zeros(len(batch_indices)))
            loader = DataLoader(dataset, batch_size=sub_batch_size, shuffle=False)
            
            for x_batch, _ in loader:
                x_batch = x_batch.to(device)
                logits, feats = model(x_batch, return_features=True)
                probs = F.softmax(logits, dim=1)
                
                embs_list.append(feats.cpu().numpy())
                probs_list.append(probs.cpu().numpy())
            
            # Release temporary objects after each candidate chunk.
            del cand_tensor, dataset, loader
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Concatenate extracted features and probabilities.
    embs = np.vstack(embs_list)
    probs = np.vstack(probs_list)
    
    # Normalize feature embeddings.
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    
    # Release intermediate lists before BADGE selection.
    del embs_list, probs_list
    gc.collect()
    
    # Construct the BADGE probability and feature components.
    m = embs.shape[0]
    emb_norms_square = np.sum(embs ** 2, axis=-1).astype(np.float32)
    max_inds = np.argmax(probs, axis=-1)
    
    probs_mod = -1.0 * probs
    probs_mod[np.arange(m), max_inds] += 1.0
    probs_mod = probs_mod / (np.linalg.norm(probs_mod, axis=1, keepdims=True) + 1e-8)
    prob_norms_square = np.sum(probs_mod ** 2, axis=-1).astype(np.float32)
    
    # Select centers with k-means++ initialization.
    mu = None
    D2 = None
    chosen = set()
    chosen_list = []
    
    for i in range(k):
        chosen, chosen_list, mu, D2 = badge_init_centers_optimized(
            (probs_mod, prob_norms_square),
            (embs, emb_norms_square),
            chosen, chosen_list, mu, D2
        )
    
    # Map candidate-local indices back to global training indices.
    selected_global_indices = [candidate_indices[i] for i in chosen_list]
    
    # Release large arrays after selection.
    del embs, probs, probs_mod, emb_norms_square, prob_norms_square
    gc.collect()
    
    return selected_global_indices


# ======================== Memory-Mapped XRD Dataset ========================
class MemmapXRDDataset(Dataset):
    """Expose memory-mapped XRD arrays through the PyTorch Dataset API."""
    def __init__(self, xrd_memmap, label_memmap, indices=None):
        """Initialize a dataset backed by memory-mapped arrays.

        Args:
            xrd_memmap: Memory-mapped XRD feature array.
            label_memmap: Memory-mapped one-hot label array.
            indices: Optional sample indices. All samples are used when omitted.
        """
        self.xrd_memmap = xrd_memmap
        self.label_memmap = label_memmap
        self.indices = indices if indices is not None else range(len(xrd_memmap))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Resolve the dataset-relative index to the source-array index.
        actual_idx = self.indices[idx]
        
        # Read and reshape one XRD sample from the memory-mapped array.
        x_1d_series = self.xrd_memmap[actual_idx].flatten().astype(np.float32)
        x_1d_tensor = torch.from_numpy(x_1d_series).unsqueeze(0)
        
        # Read the corresponding one-hot label.
        y = torch.tensor(self.label_memmap[actual_idx], dtype=torch.float32)
        
        return x_1d_tensor, y
    
    def subset(self, new_indices):
        """Create an indexed subset that shares the same memory-mapped arrays."""
        return MemmapXRDDataset(self.xrd_memmap, self.label_memmap, new_indices)


# ================================ CNN Model ================================
class CustomDropout(nn.Module):
    def __init__(self, rate):
        super(CustomDropout, self).__init__()
        self.rate = rate

    def forward(self, x):
        return F.dropout(x, self.rate, training=self.training)


class XRDNet(nn.Module):
    def __init__(self, n_phases, conv_params, n_dense, dropout_rate=0.7):
        super(XRDNet, self).__init__()
        self.conv_layers = nn.Sequential()
        in_channels = 1
        
        for i, params in enumerate(conv_params):
            self.conv_layers.add_module(
                f"conv_{i}",
                nn.Conv1d(in_channels, params["out_channels"],
                          kernel_size=params["kernel_size"],
                          padding=params["padding"])
            )
            self.conv_layers.add_module(f"relu_{i}", nn.ReLU())
            self.conv_layers.add_module(
                f"pool_{i}",
                nn.MaxPool1d(kernel_size=params["pool_size"], stride=2)
            )
            in_channels = params["out_channels"]

        self.flatten = nn.Flatten()
        
        # Infer the flattened convolutional feature size dynamically.
        dummy = torch.zeros(1, 1, 4501)
        flattened_size = torch.flatten(self.conv_layers(dummy), start_dim=1).size(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, n_dense[0]),
            nn.BatchNorm1d(n_dense[0]),
            nn.ReLU(),
            CustomDropout(dropout_rate),
            nn.Linear(n_dense[0], n_dense[1]),
            nn.BatchNorm1d(n_dense[1]),
            nn.ReLU(),
            CustomDropout(dropout_rate),
            nn.Linear(n_dense[1], n_phases)
        )
    
    def extra_features(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        return x
    
    def forward(self, x, return_features=False):
        feats = self.extra_features(x)
        logits = self.classifier(feats)
        if return_features:
            return logits, feats
        return logits


# ============================= Optuna Objective ============================
def objective(trial, train_data_memmap, train_label_memmap, current_indices,
              val_set, device, n_phases, num_epochs=10):
    """Evaluate one Optuna trial on the current labeled subset.

    The objective trains an ``XRDNet`` using memory-mapped data and returns the
    best validation accuracy observed during the trial.
    """
    # 1. Sample hyperparameters from the configured search space.
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    n_conv_layers = trial.suggest_int("n_conv_layers", 3, 4)
    
    conv_params = []
    for i in range(n_conv_layers):
        conv_params.append({
            "out_channels": trial.suggest_categorical(f"out_ch_{i}", [32, 64, 128]),
            "kernel_size": trial.suggest_int(f"ksize_{i}", 10, 40, step=5),
            "pool_size": trial.suggest_int(f"pool_{i}", 2, 3),
            "padding": trial.suggest_int(f"pad_{i}", 5, 20)
        })
    
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    dropout = trial.suggest_float("dropout_rate", 0.3, 0.6)
    dense_units = trial.suggest_categorical("dense_units", [[256, 128], [512, 256], [1024, 512]])
    
    # 2. Build data loaders backed by memory-mapped arrays.
    train_dataset = MemmapXRDDataset(train_data_memmap, train_label_memmap, current_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    # 3. Construct the trial model.
    model = XRDNet(
        n_phases=n_phases,
        conv_params=conv_params,
        n_dense=dense_units,
        dropout_rate=dropout
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # 4. Train the trial model with early stopping.
    best_val_acc = 0.0
    early_stopping = EarlyStopping(patience=3, delta=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y.argmax(dim=1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Evaluate on the validation set.
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_correct += (model(x).argmax(dim=1) == y.argmax(dim=1)).sum().item()
        val_acc = val_correct / len(val_set)
        
        best_val_acc = max(best_val_acc, val_acc)
        
        if early_stopping(val_acc):
            break
    
    # 5. Release trial-specific objects.
    del model, optimizer, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return best_val_acc


# ========================== Active Learning Workflow =======================
def class_active_learning_optimized(
    train_data_memmap, 
    train_label_memmap, 
    val_set, 
    test_set, 
    initial_k=10, 
    select_k=5,
    num_rounds=5, 
    num_classes=177, 
    num_epochs=5,
    save_dir="./active_optuna_output",
    seed=42,
    select_way="global",
    select_model="active",
    weight_search=False,
    n_trials=30,
    param_path=None
):
    """Run the memory-optimized active learning workflow.

    Hyperparameters are either searched with Optuna or loaded from a saved
    parameter checkpoint. Each acquisition round rebuilds and retrains the
    classifier from scratch.
    """
    os.makedirs(save_dir, exist_ok=True)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if select_way not in ["global", "class"]:
        raise ValueError(f"select_way must be 'global' or 'class', got: {select_way}")
    if select_model not in ["active", "random"]:
        raise ValueError(f"select_model must be 'active' or 'random', got: {select_model}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    
    print_memory_usage("Before active learning")
    print(f"Random seed: {seed}, selection mode: {select_way}, selection model: {select_model}")
    if weight_search:
        print(f"Hyperparameter search enabled, Optuna trials: {n_trials}")
    
    # Store selected indices in a set for efficient membership checks.
    selected_global_set = set()
    
    # Determine the total training-pool size.
    total_samples = len(train_data_memmap)
    print(f"Total number of samples: {total_samples}")
    
    # Group all training indices by class.
    print("Building the class-to-index mapping...")
    class_indices = {cls: [] for cls in range(num_classes)}
    
    # Build the class index map in batches.
    batch_size = 50000
    for start in tqdm(range(0, total_samples, batch_size), desc="Building index mapping"):
        end = min(start + batch_size, total_samples)
        batch_indices = range(start, end)
        
        # Read one label batch and convert one-hot labels to class IDs.
        batch_labels = train_label_memmap[start:end]
        batch_classes = np.argmax(batch_labels, axis=1)
        
        for idx_rel, cls in enumerate(batch_classes):
            actual_idx = start + idx_rel
            class_indices[cls].append(actual_idx)
    
    print_memory_usage("Index mapping completed")
    
    # Select up to ``initial_k`` initial samples from each class.
    initial_selected = []
    for cls in tqdm(range(num_classes), desc="Initial selection"):
        if len(class_indices[cls]) > 0:
            available = class_indices[cls]
            n_select = min(initial_k, len(available))
            selected = np.random.choice(available, size=n_select, replace=False)
            
            for idx in selected:
                initial_selected.append(idx)
                selected_global_set.add(idx)
    
    print(f"Initially selected {len(initial_selected)} samples")
    print_memory_usage("Initial selection completed")
    
    # ===================== Hyperparameter Search or Loading ====================
    # Resolve hyperparameters through either search or checkpoint loading.
    if weight_search:
        # Run Optuna when hyperparameter search is enabled.
        al_optuna_dir = DEFAULT_OPTUNA_DIR
        os.makedirs(al_optuna_dir, exist_ok=True)
        
        study = optuna.create_study(
            direction="maximize",
            storage=f"sqlite:///{al_optuna_dir}/optuna_round1_3.5.db",
            study_name="al_optimization_3.5",
            load_if_exists=True,
        )
        
        complete_trials = [t for t in study.trials if t.state.is_finished()]
        remaining = max(0, n_trials - len(complete_trials))
        
        if remaining > 0:
            print(f"Starting Optuna hyperparameter search; remaining trials: {remaining}")
            study.optimize(
                lambda trial: objective(
                    trial, train_data_memmap, train_label_memmap,
                    list(selected_global_set), val_set, device, num_classes
                ),
                n_trials=remaining
            )
        else:
            print(f"Optuna search already completed with {len(complete_trials)} trials")
        
        best_params = study.best_trial.params
        print(f"Best validation accuracy: {study.best_trial.value:.4f}")
        
        # Save the full Optuna trial table.
        trials_df = study.trials_dataframe()
        trials_df.to_csv(f"{al_optuna_dir}/1optuna_hyperparameter_trials.csv", index=False)
        
        # Reconstruct and save the selected model configuration.
        conv_params = []
        for i in range(best_params["n_conv_layers"]):
            conv_params.append({
                "out_channels": best_params[f"out_ch_{i}"],
                "kernel_size": best_params[f"ksize_{i}"],
                "pool_size": best_params[f"pool_{i}"],
                "padding": best_params[f"pad_{i}"]
            })
        
        temp_model = XRDNet(
            n_phases=num_classes,
            conv_params=conv_params,
            n_dense=best_params["dense_units"],
            dropout_rate=best_params["dropout_rate"]
        ).to(device)
        
        os.makedirs(f"{al_optuna_dir}/Round1", exist_ok=True)
        torch.save({
            'state_dict': temp_model.state_dict(),
            'best_params': best_params,
            'pytorch_version': torch.__version__
        }, DEFAULT_OPTUNA_PARAM_PATH)
        
        parameter_source = DEFAULT_OPTUNA_PARAM_PATH
        print(f"Optuna search completed; the best hyperparameters were saved to {al_optuna_dir}/")
        del temp_model
        gc.collect()
        
    else:
        # Load existing hyperparameters when search is disabled.
        load_path = param_path or DEFAULT_OPTUNA_PARAM_PATH
        print(f"Loading existing hyperparameters from: {load_path}")
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Hyperparameter file not found: {load_path}\n"
                                    f"Run with --weight_search first or provide a valid --param_path")
        
        params = torch.load(load_path, map_location=device)
        best_params = params['best_params']
        parameter_source = load_path
        print(f"Loaded hyperparameters: {best_params}")
    
    # Rebuild convolution settings for per-round model initialization.
    conv_params = []
    for i in range(best_params["n_conv_layers"]):
        conv_params.append({
            "out_channels": best_params[f"out_ch_{i}"],
            "kernel_size": best_params[f"ksize_{i}"],
            "pool_size": best_params[f"pool_{i}"],
            "padding": best_params[f"pad_{i}"]
        })

    # Save the resolved experiment configuration for reproducibility.
    experiment_config = {
        'seed': seed,
        'num_rounds': num_rounds,
        'initial_k': initial_k,
        'select_k': select_k,
        'num_classes': num_classes,
        'num_epochs': num_epochs,
        'select_way': select_way,
        'select_model': select_model,
        'weight_search': weight_search,
        'n_trials': n_trials,
        'param_source': parameter_source,
        'best_params': best_params,
        'conv_params': conv_params,
        'input_length': 4501,
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'device': str(device)
    }
    config_path = os.path.join(save_dir, 'experiment_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(experiment_config, f, ensure_ascii=False, indent=2)
    print(f"Experiment configuration saved to: {config_path}")

    # Run the acquisition-and-training rounds.
    val_metrics_list = []
    test_metrics_list = []
    
    for rnd in range(num_rounds):
        print(f"\n=== Round {rnd + 1}/{num_rounds} ===")
        print_memory_usage(f"Start of round {rnd + 1}")
        
        # Build the labeled training subset for the current round.
        current_indices = list(selected_global_set)
        train_dataset = MemmapXRDDataset(train_data_memmap, train_label_memmap, current_indices)
        
        # Create training, validation, and test data loaders.
        train_loader = DataLoader(
            train_dataset, 
            batch_size=best_params["batch_size"], 
            shuffle=True,
            num_workers=0,  # Disable worker processes to avoid duplicated memory mappings.
            pin_memory=False,
            drop_last=True
        )
        
        val_loader = DataLoader(val_set, batch_size=best_params["batch_size"], shuffle=False)
        test_loader = DataLoader(test_set, batch_size=best_params["batch_size"], shuffle=False)
        
        # Reinitialize the model and optimizer from scratch each round.
        torch.manual_seed(seed + rnd)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + rnd)

        model = XRDNet(
            n_phases=num_classes,
            conv_params=conv_params,
            n_dense=best_params["dense_units"],
            dropout_rate=best_params["dropout_rate"]
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])

        # Initialize per-round training state.
        early_stopping = EarlyStopping(patience=10)
        best_val_acc = 0.0
        best_model_wts = None
        
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for x, y in tqdm(train_loader, desc=f"Training epoch {epoch+1}", leave=False):
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y.argmax(dim=1))
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * x.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == y.argmax(dim=1)).sum().item()
                total += y.size(0)
            
            train_acc = correct / total if total > 0 else 0.0
            train_loss = epoch_loss / total if total > 0 else 0.0
            
            # Evaluate on the validation set.
            model.eval()
            val_correct = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    outputs = model(x)
                    val_correct += (outputs.argmax(dim=1) == y.argmax(dim=1)).sum().item()
            
            val_acc = val_correct / len(val_set) if len(val_set) > 0 else 0
            
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
            
            # Track early stopping and retain the best validation weights.
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_wts = deepcopy(model.state_dict())
            
            if early_stopping(val_acc):
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Restore the best validation weights when available.
        if best_model_wts:
            model.load_state_dict(best_model_wts)

        # Compute complete validation and test metrics.
        val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(model, val_loader, criterion, device, val=True)
        test_acc, test_prec, test_rec, test_f1, test_preds, test_labels = evaluate(model, test_loader, criterion, device)

        val_metrics_list.append({
            'round': rnd + 1,
            'n_samples': len(current_indices),
            'val_acc': val_acc,
            'val_precision_macro': val_prec,
            'val_recall_macro': val_rec,
            'val_f1_macro': val_f1
        })

        test_metrics_list.append({
            'round': rnd + 1,
            'n_samples': len(current_indices),
            'test_acc': test_acc,
            'test_precision_macro': test_prec,
            'test_recall_macro': test_rec,
            'test_f1_macro': test_f1
        })

        print(f"Round {rnd+1} - Validation Acc/F1: {val_acc:.4f}/{val_f1:.4f}, Test Acc/F1: {test_acc:.4f}/{test_f1:.4f}")
        
        # Acquire new samples with BADGE or random selection.
        # Keep an empty acquisition list for the final round.
        new_indices = []
        if rnd < num_rounds - 1:  # Skip acquisition after the final training round.
            print("Starting sample selection...")

            if select_model == "active":
                # Apply the existing BADGE acquisition routine.
                if select_way == "global":
                    all_possible = set(range(total_samples))
                    remaining = list(all_possible - selected_global_set)

                    if not remaining:
                        print("No unlabeled samples remain; terminating active learning early")
                        break

                    total_to_select = min(num_classes * select_k, len(remaining))
                    if total_to_select > 0:
                        print(f"[global-active] Selecting {total_to_select} new samples from {len(remaining)} candidates")
                        new_indices = badge_selection_optimized(
                            train_data_memmap,
                            model,
                            remaining,
                            total_to_select,
                            device,
                            batch_size=min(best_params["batch_size"], 64)
                        )
                else:
                    print(f"[class-active] Selecting up to {select_k} samples per class")
                    for cls in tqdm(range(num_classes), desc="Class-wise selection", leave=False):
                        cls_remaining = [idx for idx in class_indices[cls] if idx not in selected_global_set]
                        if not cls_remaining:
                            continue

                        cls_to_select = min(select_k, len(cls_remaining))
                        cls_selected = badge_selection_optimized(
                            train_data_memmap,
                            model,
                            cls_remaining,
                            cls_to_select,
                            device,
                            batch_size=min(best_params["batch_size"], 64)
                        )
                        new_indices.extend(cls_selected)
            else:
                # Apply the random-acquisition baseline.
                if select_way == "global":
                    all_possible = set(range(total_samples))
                    remaining = list(all_possible - selected_global_set)

                    if not remaining:
                        print("No unlabeled samples remain; terminating active learning early")
                        break

                    total_to_select = min(num_classes * select_k, len(remaining))
                    if total_to_select > 0:
                        print(f"[global-random] Randomly selecting {total_to_select} new samples from {len(remaining)} candidates")
                        new_indices = list(np.random.choice(remaining, size=total_to_select, replace=False))
                else:
                    print(f"[class-random] Randomly selecting up to {select_k} samples per class")
                    for cls in tqdm(range(num_classes), desc="Class-wise selection", leave=False):
                        cls_remaining = [idx for idx in class_indices[cls] if idx not in selected_global_set]
                        if not cls_remaining:
                            continue
                        cls_to_select = min(select_k, len(cls_remaining))
                        if cls_to_select > 0:
                            cls_selected = list(np.random.choice(cls_remaining, size=cls_to_select, replace=False))
                            new_indices.extend(cls_selected)

            # Add newly acquired samples to the labeled pool.
            if new_indices:
                for idx in new_indices:
                    selected_global_set.add(idx)
                print(f"Selected {len(new_indices)} new samples")
            else:
                print("No new samples were selected")
        
        print_memory_usage(f"End of round {rnd + 1}")
        
        # Release unused memory before saving round artifacts.
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Save per-round artifacts.
        # 1. Save standalone model weights for the current round.
        round_model_path = os.path.join(save_dir, f"model_round_{rnd+1}.pth")
        torch.save(model.state_dict(), round_model_path)

        # 2. Save a complete checkpoint for the current round.
        # ``trained_indices`` matches the saved model; ``next_round_indices`` supports the next round or recovery.
        checkpoint_path = os.path.join(save_dir, f"checkpoint_round_{rnd+1}.pth")
        torch.save({
            'round': rnd + 1,
            'model_state_dict': model.state_dict(),
            'trained_indices': current_indices,
            'newly_selected_indices': new_indices,
            'next_round_indices': list(selected_global_set),
            'current_val_metrics': val_metrics_list[-1],
            'current_test_metrics': test_metrics_list[-1],
            'val_metrics_history': val_metrics_list,
            'test_metrics_history': test_metrics_list,
            'best_params': best_params,
            'conv_params': conv_params,
            'num_classes': num_classes,
            'seed': seed,
            'model_initialization_seed': seed + rnd,
            'initial_k': initial_k,
            'select_k': select_k,
            'select_way': select_way,
            'select_model': select_model
        }, checkpoint_path)
        print(f"Model weights saved to: {round_model_path}")
        print(f"Checkpoint saved to: {checkpoint_path}")
    
    # Save final metrics and plots.
    print("\nSaving final results...")
    
    # Save the per-round validation and test accuracy curves.
    test_accs = [m['test_acc'] for m in test_metrics_list]
    val_accs = [m['val_acc'] for m in val_metrics_list]

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(test_accs) + 1), test_accs, 'b-o', label='Test Accuracy')
    plt.plot(range(1, len(val_accs) + 1), val_accs, 'r--s', label='Validation Accuracy')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title('Active Learning Performance')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save complete per-round metrics as CSV files.
    if test_metrics_list:
        test_df = pd.DataFrame(test_metrics_list)
        test_df.to_csv(os.path.join(save_dir, "test_metrics_by_round.csv"), index=False)
    if val_metrics_list:
        val_df = pd.DataFrame(val_metrics_list)
        val_df.to_csv(os.path.join(save_dir, "val_metrics_by_round.csv"), index=False)
    
    print(f"Final results saved to: {save_dir}")
    
    return test_metrics_list[-1]['test_acc'] if test_metrics_list else 0.0


# ================================= CLI =====================================
def main():
    parser = argparse.ArgumentParser(description='Memory-efficient active learning with Optuna hyperparameter search')
    parser.add_argument("--rounds", type=int, default=20, help="Number of active learning rounds")
    parser.add_argument("--initial_k", type=int, default=15, help="Initial number of samples per class")
    parser.add_argument("--select_k", type=int, default=10, help="Number of samples selected per class in each round")
    parser.add_argument("--save_dir", type=str, default="./active_output_final", help="Directory for saving active learning results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--ten_times", action="store_true", help="Run active learning with 10 different random seeds")
    parser.add_argument(
        "--select_way",
        type=str,
        default="global",
        choices=["global", "class"],
        help="Selection mode: global for pool-wide selection, class for class-wise selection"
    )
    parser.add_argument(
        "--select_model",
        type=str,
        default="active",
        choices=["active", "random"],
        help="Selection model: active for BADGE acquisition, random for random sampling"
    )
    # Hyperparameter-search arguments.
    parser.add_argument("--weight_search", action="store_true",
                        help="Enable Optuna hyperparameter search and save results to al_optuna/")
    parser.add_argument("--n_trials", type=int, default=30,
                        help="Number of Optuna trials; used only with --weight_search")
    parser.add_argument("--param_path", type=str, default=None,
                        help="Path to saved hyperparameters when --weight_search is disabled. "
                             "Ignored when --weight_search is enabled. "
                             "Default: al_optuna/Round1/best_model_run1.pth")
    args = parser.parse_args()

    # Resolve command-line precedence for hyperparameter sources.
    if args.weight_search:
        effective_param_path = DEFAULT_OPTUNA_PARAM_PATH
        if args.param_path:
            print(
                "\n[Parameter priority] Both --weight_search and --param_path "
                "were provided. --weight_search takes precedence.\n"
                f"The searched hyperparameters will be saved to and subsequently "
                f"loaded from: {DEFAULT_OPTUNA_PARAM_PATH}\n"
                f"The supplied --param_path={args.param_path} is ignored for this run."
            )
    else:
        effective_param_path = args.param_path or DEFAULT_OPTUNA_PARAM_PATH

    print_memory_usage("Program start")
    
    # Load training data through memory mapping.
    print("Loading data...")
    
    def load_memmap_safe(path):
        """Load a NumPy array in memory-mapped read-only mode."""
        try:
            base = np.load(path, mmap_mode='r')
            # Recreate the mapping with the NumPy header offset and array shape.
            mm = np.memmap(path, dtype=base.dtype, mode='r', 
                         offset=base.offset, shape=base.shape)
            return mm
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            # Fall back to a raw memory map if NumPy loading fails.
            return np.memmap(path, mode='r')
    
    train_data = load_memmap_safe('../data_set/X_train.npy')
    train_labels = load_memmap_safe('../val_test/train_177(4108)_y.npy')
    
    # Load the smaller validation and test arrays.
    print("Loading validation and test sets...")
    val_samples = np.load('../data_set/X_val.npy', mmap_mode='r')
    val_labels = np.load('../val_test/val_test_177(4108)_y.npy', mmap_mode='r')
    test_samples = np.load('../data_set/X_test.npy', mmap_mode='r')
    test_labels = np.load('../val_test/val_test_177(4108)_y.npy', mmap_mode='r')
    
    print_memory_usage("Data loading completed")
    
    # Wrap validation and test arrays as datasets.
    val_dataset = MemmapXRDDataset(val_samples, val_labels)
    test_dataset = MemmapXRDDataset(test_samples, test_labels)
    
    # Run either one experiment or ten experiments with consecutive seeds.
    if args.ten_times:
        print("\nten_times=True: running 10 experiments with different random seeds")
        all_test_rounds = []
        all_val_rounds = []
        final_results = []

        for i in range(10):
            run_seed = args.seed + i
            run_save_dir = os.path.join(args.save_dir, f"seed_{run_seed}")
            print(f"\n{'=' * 80}")
            print(f"Starting experiment {i + 1}/10, seed={run_seed}, save_dir={run_save_dir}")
            print(f"{'=' * 80}")

            class_active_learning_optimized(
                train_data_memmap=train_data,
                train_label_memmap=train_labels,
                val_set=val_dataset,
                test_set=test_dataset,
                initial_k=args.initial_k,
                select_k=args.select_k,
                num_rounds=args.rounds,
                num_classes=177,
                num_epochs=100,
                save_dir=run_save_dir,
                seed=run_seed,
                select_way=args.select_way,
                select_model=args.select_model,
                weight_search=args.weight_search if i == 0 else False,  # Search only during the first repeated run.
                n_trials=args.n_trials,
                param_path=effective_param_path,
            )

            # Load the per-round metrics produced by the current seed.
            test_metrics_path = os.path.join(run_save_dir, "test_metrics_by_round.csv")
            val_metrics_path = os.path.join(run_save_dir, "val_metrics_by_round.csv")
            test_run_df = pd.read_csv(test_metrics_path)
            val_run_df = pd.read_csv(val_metrics_path)
            test_run_df["seed"] = run_seed
            val_run_df["seed"] = run_seed
            all_test_rounds.append(test_run_df)
            all_val_rounds.append(val_run_df)

            final_test_row = test_run_df.sort_values("round").iloc[-1]
            final_results.append({
                "seed": run_seed,
                "final_round": int(final_test_row["round"]),
                "final_n_samples": int(final_test_row["n_samples"]),
                "final_test_acc": float(final_test_row["test_acc"]),
                "final_test_precision_macro": float(final_test_row["test_precision_macro"]),
                "final_test_recall_macro": float(final_test_row["test_recall_macro"]),
                "final_test_f1_macro": float(final_test_row["test_f1_macro"])
            })

        os.makedirs(args.save_dir, exist_ok=True)

        # Save all raw per-round results across the ten seeds.
        all_test_df = pd.concat(all_test_rounds, ignore_index=True)
        all_val_df = pd.concat(all_val_rounds, ignore_index=True)
        all_test_df.to_csv(
            os.path.join(args.save_dir, "ten_times_test_metrics_all.csv"),
            index=False
        )
        all_val_df.to_csv(
            os.path.join(args.save_dir, "ten_times_val_metrics_all.csv"),
            index=False
        )

        # Aggregate test metrics by round using the mean and sample standard deviation.
        test_round_summary = (
            all_test_df
            .groupby("round", as_index=False)
            .agg(
                n_runs=("seed", "nunique"),
                n_samples_mean=("n_samples", "mean"),
                n_samples_std=("n_samples", "std"),
                test_acc_mean=("test_acc", "mean"),
                test_acc_std=("test_acc", "std"),
                test_precision_macro_mean=("test_precision_macro", "mean"),
                test_precision_macro_std=("test_precision_macro", "std"),
                test_recall_macro_mean=("test_recall_macro", "mean"),
                test_recall_macro_std=("test_recall_macro", "std"),
                test_f1_macro_mean=("test_f1_macro", "mean"),
                test_f1_macro_std=("test_f1_macro", "std")
            )
            .fillna(0.0)
        )
        test_round_summary.to_csv(
            os.path.join(args.save_dir, "ten_times_test_metrics_summary.csv"),
            index=False
        )

        # Aggregate validation metrics by round using the mean and sample standard deviation.
        val_round_summary = (
            all_val_df
            .groupby("round", as_index=False)
            .agg(
                n_runs=("seed", "nunique"),
                n_samples_mean=("n_samples", "mean"),
                n_samples_std=("n_samples", "std"),
                val_acc_mean=("val_acc", "mean"),
                val_acc_std=("val_acc", "std"),
                val_precision_macro_mean=("val_precision_macro", "mean"),
                val_precision_macro_std=("val_precision_macro", "std"),
                val_recall_macro_mean=("val_recall_macro", "mean"),
                val_recall_macro_std=("val_recall_macro", "std"),
                val_f1_macro_mean=("val_f1_macro", "mean"),
                val_f1_macro_std=("val_f1_macro", "std")
            )
            .fillna(0.0)
        )
        val_round_summary.to_csv(
            os.path.join(args.save_dir, "ten_times_val_metrics_summary.csv"),
            index=False
        )

        # Save final results for each seed and their aggregate statistics.
        final_results_df = pd.DataFrame(final_results)
        final_results_df.to_csv(
            os.path.join(args.save_dir, "ten_times_final_results.csv"),
            index=False
        )
        metric_columns = [
            "final_test_acc",
            "final_test_precision_macro",
            "final_test_recall_macro",
            "final_test_f1_macro"
        ]
        final_statistics = (
            final_results_df[metric_columns]
            .agg(["mean", "std"])
            .T
            .reset_index()
        )
        final_statistics.columns = ["metric", "mean", "std"]
        final_statistics.to_csv(
            os.path.join(args.save_dir, "ten_times_final_statistics.csv"),
            index=False
        )

        # Plot mean test accuracy with a one-standard-deviation band.
        rounds = test_round_summary["round"].to_numpy()
        acc_mean = test_round_summary["test_acc_mean"].to_numpy()
        acc_std = test_round_summary["test_acc_std"].to_numpy()
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, acc_mean, marker="o", label="Mean Test Accuracy")
        plt.fill_between(
            rounds,
            acc_mean - acc_std,
            acc_mean + acc_std,
            alpha=0.2,
            label="±1 SD"
        )
        plt.xlabel("Round")
        plt.ylabel("Test Accuracy")
        plt.title("Active Learning Test Accuracy Across 10 Seeds")
        plt.legend()
        plt.grid(True)
        plt.savefig(
            os.path.join(args.save_dir, "ten_times_test_accuracy_curve.png"),
            dpi=150,
            bbox_inches="tight"
        )
        plt.close()

        # Plot mean test macro-F1 with a one-standard-deviation band.
        f1_mean = test_round_summary["test_f1_macro_mean"].to_numpy()
        f1_std = test_round_summary["test_f1_macro_std"].to_numpy()
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, f1_mean, marker="o", label="Mean Test Macro-F1")
        plt.fill_between(
            rounds,
            f1_mean - f1_std,
            f1_mean + f1_std,
            alpha=0.2,
            label="±1 SD"
        )
        plt.xlabel("Round")
        plt.ylabel("Test Macro-F1")
        plt.title("Active Learning Test Macro-F1 Across 10 Seeds")
        plt.legend()
        plt.grid(True)
        plt.savefig(
            os.path.join(args.save_dir, "ten_times_test_macro_f1_curve.png"),
            dpi=150,
            bbox_inches="tight"
        )
        plt.close()

        print("\nAll 10 experiments completed")
        print("\nFinal result for each experiment:")
        print(final_results_df)
        print("\nMean and standard deviation of the final metrics:")
        print(final_statistics)
        print(f"\nSummary results saved to: {args.save_dir}")
    else:
        final_acc = class_active_learning_optimized(
            train_data_memmap=train_data,
            train_label_memmap=train_labels,
            val_set=val_dataset,
            test_set=test_dataset,
            initial_k=args.initial_k,
            select_k=args.select_k,
            num_rounds=args.rounds,
            num_classes=177,
            num_epochs=100,
            save_dir=args.save_dir,
            seed=args.seed,
            select_way=args.select_way,
            select_model=args.select_model,
            weight_search=args.weight_search,
            n_trials=args.n_trials,
            param_path=effective_param_path,
        )
        print(f"\nActive learning completed. Final test accuracy: {final_acc:.4f}")
    print_memory_usage("Program end")


if __name__ == "__main__":
    main()
