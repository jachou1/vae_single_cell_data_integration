import wandb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from umap import UMAP
from sklearn.model_selection import train_test_split
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.patches as mpatches
import pandas as pd
from sklearn.metrics import confusion_matrix
from io import BytesIO
import argparse
import seaborn as sns
import os
from collections import deque
import pickle

def softplus_np(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

# Newest model (2 hidden layers in encoder / decoders + BatchNorm added)
# Define VAE model with class probabilities output
class JointVAEwithClassification(nn.Module):
    def __init__(self, n_genes, n_classes, hidden_dim=200, latent_dim=20):
        super(JointVAEwithClassification, self).__init__()
        self.n_genes = n_genes
        self.n_classes = n_classes
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_dim, latent_dim * 2))

        self.decoder_scRNA = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_dim, n_genes))

        self.decoder_xenium = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_dim, n_genes))

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_dim, n_classes))

    def encode(self, data):
        mu_logstdev = self.encoder(data)
        mu, logstdev = mu_logstdev[..., :self.latent_dim], mu_logstdev[..., self.latent_dim:]
        # Take care of NaNs:
        if torch.isnan(logstdev).any():
            print("NaN values detected in scale, replacing them with a small value.")
            logstdev = torch.nan_to_num(logstdev, nan=1e-5)  # Replace NaNs
        # Clamp the tensors so that it's greater than 0
        logstdev = torch.clamp(logstdev, min= 1e-5)
        dist = torch.distributions.Normal(mu, torch.nn.functional.softplus(logstdev))
        return dist.rsample(), mu, logstdev

    def forward(self, data, modality_mask):
        z, mu, logstdev = self.encode(data)

        # Provide the labels on modality so that it'll affect the reconstruction array
        recon_scRNA = self.decoder_scRNA(z[modality_mask == 1])  # This will pick out the cells that have scRNA data
        recon_Xenium = self.decoder_xenium(z[modality_mask == 0])  # This will pick out the cells that have Xenium labels
        class_logits = self.classifier(z)
        return class_logits, z, mu, logstdev, recon_scRNA, recon_Xenium

class MultimodalDataset(Dataset):
    def __init__(self, cells, cell_type_labels, modality_labels, true_labels):
        self.cells = cells
        self.cell_type_labels = cell_type_labels
        self.modality_labels = modality_labels
        self.true_labels = true_labels

    def __getitem__(self, index):
        return (self.cells[index], self.cell_type_labels[index], self.modality_labels[index], self.true_labels[index])

    def __len__(self):
        return len(self.cells)

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, filename='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_loss = float('inf') # Initialize to a very large value for loss
        self.filename = filename

    def __call__(self, val_loss, model):
        score = -val_loss   # We want the loss to decrease, so we negate it to follow the same logic

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation accuracy increases.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.best_loss = val_loss
        torch.save(model.state_dict(), self.filename)

class ContrastiveSampler:
    """
    Iterator that samples from contrastive distribution.
    Input:
        - groupings: list of tensors, signifying the target clustering (Meaning; each tensor contains idxs corresponding to idxs of each celltype)
        - batch_size: the number of contrastive samples to generate each iteration
        - n_negatives: the number of negative samples to generate per anchor
    """

    def __init__(
            self,
            groupings,  ## list<tensor>
            batch_size: int,
            n_negatives: int = 1,
    ):
        self.samples = torch.cat(groupings)                                # Concatenate the celltype labels into 1 tensor
        self.group_sizes = torch.LongTensor([len(x) for x in groupings])   # Keeps track of how many cells of each type
        self.batch_size = batch_size                                       # Batch_size
        self.n_negatives = n_negatives
        self.c_sizes = torch.cumsum(self.group_sizes, dim=0)
        self.total_size = self.c_sizes[-1]
        lbound = torch.cat([torch.tensor([0]), self.c_sizes[:-1]])   # lower-bound?
        bounds = torch.stack([lbound, self.c_sizes]).T               # 2D tensor: [[0, 522], [522, 1000]]
        self.fbounds = bounds.type(torch.float64) / self.total_size  # 2D tensor: [[0, 0.5220], [0.5220, 1.000]]

    def __iter__(self):
        return self

    def __next__(self):
        anchor_idx = torch.randint(self.total_size, size=(self.batch_size,))
        groups = torch.searchsorted(self.c_sizes, anchor_idx, side="right")   # I'm not sure what this doing; returns
                                                        # indices wherein anchor_idx can be placed into self.c_sizes and self.c_sizes' order is preserved

        pos_range = self.fbounds[groups, 1] - self.fbounds[groups, 0]
        pos_offset = self.fbounds[groups, 0]

        neg_range = 1 + self.fbounds[groups, 0] - self.fbounds[groups, 1]
        neg_offset = self.fbounds[groups, 1]

        pos_raw = (
                pos_range * torch.rand(size=(self.batch_size,), dtype=torch.float64)
                + pos_offset
        )
        pos_idx = (self.total_size * pos_raw).long() % self.total_size

        neg_raw = neg_range.unsqueeze(1) * torch.rand(
            size=(self.batch_size, self.n_negatives), dtype=torch.float64
        ) + neg_offset.unsqueeze(1)
        neg_idx = (self.total_size * neg_raw).long() % self.total_size

        return (self.samples[anchor_idx], self.samples[pos_idx], self.samples[neg_idx])


def build_contrastive_sampler(cell_type_labels, batch_size):
    max_class = cell_type_labels.max()
    idxs = torch.arange(cell_type_labels.shape[0])
    groupings = [torch.Tensor(idxs[cell_type_labels == i]) for i in range(max_class + 1)]  # this generates the groupings of idxs of each celltype
    return ContrastiveSampler(groupings, batch_size)


def contrastive_loss(
        anchor_embed: torch.FloatTensor,
        pos_embed: torch.FloatTensor,
        neg_embed: torch.FloatTensor,
        con_temp: float = 1
):
    # anchor_embed = anchor_embed / torch.linalg.vector_norm(anchor_embed, dim=-1, keepdims=True).unsqueeze(1)  ## B x 1 x d
    # pos_embed = pos_embed / torch.linalg.vector_norm(pos_embed, dim=-1, keepdims=True).unsqueeze(1)  ## B x 1 x d
    # neg_embed = neg_embed / torch.linalg.vector_norm(neg_embed, dim=-1, keepdims=True).unsqueeze(1)  ## B x 1 x d
    anchor_embed = anchor_embed.unsqueeze(1)
    pos_embed = pos_embed.unsqueeze(1)
    neg_embed = neg_embed.unsqueeze(1)

    pos_preds = torch.sum(anchor_embed * pos_embed, dim=-1)  ## B x 1
    neg_preds = torch.sum(anchor_embed * neg_embed, dim=-1)  ## B x n_neg (B x 1, since n_neg = 1)

    all_preds = torch.cat((pos_preds, neg_preds), dim=-1) / con_temp

    ## cross entropy loss
    con_loss = torch.logsumexp(all_preds, dim=-1) - all_preds[:, 0]
    return con_loss

# Function to get UMAP embedding and plot with modality and cell-type labels
def umap_and_plot(z_embed: torch.FloatTensor, z_modality_labels: np.ndarray, z_cell_type_labels: np.ndarray, epoch: int, file_dir: str):
    # Convert tensor to np array
    if isinstance(z_embed, torch.Tensor):
        z_embed = z_embed.numpy()

    # Validate input dimensions
    assert z_embed.shape[0] == len(z_modality_labels) == len(
        z_cell_type_labels), "Length mismatch between embeddings and labels"

    xy = UMAP(random_state=16).fit_transform(z_embed)
    # Let's now plot the xy with modality labels and cell_type labels
    fig, axarr = plt.subplots(1, 2, figsize=(10 * 1.5, 5 * 1.5), sharex=True, sharey=True)
    scatter1 = axarr[0].scatter(xy[:, 0], xy[:, 1], s=2, c=z_modality_labels)
    axarr[0].set_title('Modalities')
    scatter2 = axarr[1].scatter(xy[:, 0], xy[:, 1], s=2, c=z_cell_type_labels, cmap='tab10')
    axarr[1].set_title('Cell types')

    # Add colorbars for reference
    legend_patches = [mpatches.Patch(color=scatter2.cmap(scatter2.norm(cell_label)), label=f'{cell_label}') for cell_label in np.unique(z_cell_type_labels)]
    axarr[1].legend(handles=legend_patches, title="Cell Types", loc = 'center left', bbox_to_anchor = (1, 0.5))

    # Save plot to BytesIO
    # buf = BytesIO()
    filename = os.path.join(file_dir, f'umap_plot_z_{epoch}.png')
    # Save the plot locally
    plt.savefig(filename, format='png')
    # buf.seek(0)  # Rewind the buffer to the beginning
    plt.close()
    # return filename

def calculate_accuracy(predictions, labels):
    """Calculate accuracy given predictions and true labels."""
    if len(labels) == 0:
        accuracy = 0
    else:
        _, preds = torch.max(predictions, 1)  # Get the predicted class with the highest probability
        correct = (preds == labels).sum().item()  # Count correct predictions
        accuracy = correct / len(labels)  # Calculate accuracy
    return accuracy

def compute_weighted_total_loss(loss_list, loss_histories, loss_weights):
    """
    loss_list: List of current losses (one for each of the 5 losses)
    loss_histories: A list of deque, each deque contains the past 100 loss values
    loss_weights: List of weights to apply to each normalized loss
    """
    normalized_losses = []

    for i, loss in enumerate(loss_list):
        # Convert scalar loss to tensor if it's not already
        if isinstance(loss, (int, float)):
            loss = torch.tensor(loss)  # Convert scalar to tensor
        # Update the loss history for the current loss type
        loss_histories[i].append(loss.item())

        # Calculate the average of the last 100 steps for this loss (return scalar value)
        avg_loss = sum(loss_histories[i]) / len(loss_histories[i])

        # Normalize the current loss by its average (normalized_loss is still a tensor)
        if avg_loss > 0:
            normalized_loss = loss / avg_loss
        else:
            normalized_loss = loss  # Avoid divide-by-zero errors

        normalized_losses.append(normalized_loss)

    # Compute the weighted total loss (normalized_total_loss is a tensor)
    normalized_total_loss = sum(w * norm_loss for w, norm_loss in zip(loss_weights, normalized_losses))

    return normalized_total_loss, normalized_losses # second return value is a list of the 5 normalized losses

def fit_vae(cells, cell_type_labels, modality_labels,
            test_pct=0.2, n_epochs=100,
            cell_type_label_pct=0.1,
            learning_rate=1e-3, batch_size=64, reconstruction_penalty=1, xenium_classifier_weight=1, classification_penalty=1,
            contrastive_penalty=1, kl_penalty=1, con_temp=1, model_filename='test', patience=5):
    # Hide the labels for lots of cells
    masked_cell_type_labels = np.copy(cell_type_labels)
    masked_cell_type_labels[np.random.choice(masked_cell_type_labels.shape[0],
                                             replace=False,
                                             size=int(
                                                 masked_cell_type_labels.shape[0] * (1 - cell_type_label_pct)))] = -1

    # Train/test split
    train_cells, test_cells, train_cell_types, test_cell_types, train_modalities, test_modalities, train_true_cell_types, test_true_cell_types, _, test_idx = train_test_split(
        cells, masked_cell_type_labels, modality_labels, cell_type_labels, np.arange(len(modality_labels)), test_size=test_pct, shuffle=True, stratify = cell_type_labels,
        random_state=42
    )

    # # Standardize the input data
    train_means = train_cells.mean(axis=0)
    train_stdevs = train_cells.std(axis=0)
    train_cells = (train_cells - train_means[None]) / train_stdevs[None]
    test_cells = (test_cells - train_means[None]) / train_stdevs[None]
    #
    # # If there are NaNs in the above, drop them:
    gene_idx = np.where(~np.isnan(train_cells).any(axis=0))[0]
    train_cells_dropped_nans = train_cells[:, ~np.isnan(train_cells).any(axis=0)]
    # print(f'{train_cells_dropped_nans.shape[1]}')
    test_cells = test_cells[:, ~np.isnan(train_cells).any(axis=0)]

    # Let's further split the train_cells to have the validation_set that we monitor accuracy during training
    (train_cell_final, validation_cells, train_cell_types_final, validation_cell_types, train_modalities_final, validation_modalities, train_true_cell_types_final, \
     validation_true_cell_types) = train_test_split(
        train_cells_dropped_nans, train_cell_types, train_modalities, train_true_cell_types, test_size=0.1, shuffle=True, stratify= train_true_cell_types,
    random_state=42)

    # Let's print out how many labels are missing per split
    print(f'Train set number of missing labels: {np.sum(train_cell_types_final == -1)}')
    print(f'Test set number of missing labels: {np.sum(test_cell_types == -1)}')
    print(f'Validation set number of missing labels: {np.sum(validation_cell_types == -1)}')

    # Create tensors for the data
    t_train_cells = torch.FloatTensor(train_cell_final)
    t_test_cells = torch.FloatTensor(test_cells)

    t_train_cell_types = torch.LongTensor(train_cell_types_final)
    t_test_cell_types = torch.LongTensor(test_cell_types)

    t_train_modalities = torch.LongTensor(train_modalities_final)
    t_test_modalities = torch.LongTensor(test_modalities)

    t_train_true_cell_types = torch.LongTensor(train_true_cell_types_final)
    t_test_true_cell_types = torch.LongTensor(test_true_cell_types)

    # Do the same for validation set
    t_validation_cells = torch.FloatTensor(validation_cells)
    t_validation_modalities = torch.LongTensor(validation_modalities)
    t_validation_cell_types = torch.LongTensor(validation_cell_types)
    t_validation_true_cell_types = torch.LongTensor(validation_true_cell_types)

    # Build the datasets and loaders
    train_data = MultimodalDataset(t_train_cells, t_train_cell_types, t_train_modalities, t_train_true_cell_types)
    test_data = MultimodalDataset(t_test_cells, t_test_cell_types, t_test_modalities, t_test_true_cell_types)
    validation_data = MultimodalDataset(t_validation_cells, t_validation_cell_types, t_validation_modalities, t_validation_true_cell_types)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_data, batch_size=batch_size)

    model = JointVAEwithClassification(train_cells_dropped_nans.shape[1], cell_type_labels.max() + 1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    contraster = build_contrastive_sampler(t_train_cell_types, batch_size)

    # Need to build a contrastive_sampler for validation_set too
    val_contraster = build_contrastive_sampler(t_validation_cell_types, batch_size)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create tensors / np arrays for saving the losses per epoch (maybe average across batches: right now, it's just summed)
    recon_scRNA_loss_arr = []
    recon_xenium_loss_arr = []
    kl_loss_arr = []
    cross_entropy_loss_arr = []
    scrna_ce_loss_array = []
    xenium_ce_loss_array = []
    cl_scrna_loss_arr = []
    cl_xenium_loss_arr = []
    total_loss_arr = []

    # Log model's weights and biases after every epoch
    wandb.watch(model, log="all")

    early_stopping = EarlyStopping(patience=patience, verbose=True, filename=f'{model_filename}/checkpoint.pth')
    print(f'filename: {model_filename}/checkpoint.pth')
    for epoch in range(n_epochs):
        # print(f'epoch: {epoch}')
        model.train()

        epoch_total_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_recon_xenium_loss = 0.0
        epoch_total_ce_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_con_loss = 0.0

        epoch_unscaled_total_loss = 0.0
        epoch_unscaled_recon_scrna_loss = 0.0
        epoch_unscaled_xenium_loss = 0.0
        epoch_unscaled_ce_loss = 0.0
        epoch_unscaled_con_loss = 0.0
        epoch_unscaled_kl_loss = 0.0

        # Create empty tensor to hold the learned z per mini-batch, then plot UMAP during training to see what's happening in RT
        tensor_z_epoch = torch.empty((0, *(model.latent_dim,)))
        # Get the modality labels for plots
        z_modality_labels_epoch = []
        # Get the cell-type labels for plots
        z_cell_type_labels_epoch = []
        # Get the ground-truth cell-type labels for plots
        z_cell_type_true_labels_epoch = []

        # Initialize deques to store the last 100 loss values for each loss type
        history_len = 100
        loss_histories = [deque(maxlen=history_len) for _ in range(5)]

        for (batch_data, batch_labels, batch_modalities, batch_true_labels), (anchors, pos_samples, neg_samples) in zip(train_dataloader,
                                                                                                     contraster):
            # Move data to GPU
            batch_data, batch_labels, batch_modalities, batch_true_labels = batch_data.to(device), batch_labels.to(
                device), batch_modalities.to(device), batch_true_labels.to(device)
            optimizer.zero_grad()

            # Forward pass
            class_logits, z, mu, logstdev, recon_scRNA, recon_Xenium = model(batch_data, batch_modalities)

            # Suggestion for Reconstruction Losses: sum across all recon losses, then divide by batch_size (batch_data.shape[0])
            recon_scRNA_loss = ((recon_scRNA - batch_data[batch_modalities == 1]) ** 2).mean()
            recon_Xenium_loss = ((recon_Xenium - batch_data[batch_modalities == 0]) ** 2).mean()
            # It's possible that for a given batch, the number of scRNA and Xenium data points varies such that
            # it'll put more weight towards one data modality more than the other
            # recon_loss = ((torch.sum((recon_scRNA - batch_data[batch_modalities == 1]) ** 2) + torch.sum((recon_Xenium - batch_data[batch_modalities == 0]) ** 2))
            #               / batch_data.shape[0])

            # Classification loss for labeled data
            # if (batch_labels > -1).sum() > 0:
            #     classification_loss = torch.nn.functional.cross_entropy(class_logits[batch_labels != -1],
            #                                                             batch_labels[batch_labels != -1])
            # else:
            #     classification_loss = 0

            # Let's split up the classification_loss into data modalities
            if (batch_labels > -1).sum() > 0:
                # Calculate the classification_loss for labeled scRNA data
                scrna_bool_mask = ((batch_labels != -1) & (batch_modalities == 1))
                if scrna_bool_mask.sum() > 0:
                    classifier_loss_scrna = torch.nn.functional.cross_entropy(class_logits[scrna_bool_mask],
                                                                            batch_labels[scrna_bool_mask])
                else:
                    classifier_loss_scrna = 0
                # Calculate the classification_loss for labeled Xenium data
                xenium_bool_mask = ((batch_labels != -1) & (batch_modalities == 0))
                if xenium_bool_mask.sum() > 0:
                    classifier_loss_xenium = torch.nn.functional.cross_entropy(class_logits[xenium_bool_mask],
                                                                            batch_labels[xenium_bool_mask])
                else: # if there are no labeled Xenium data points, set it to zero
                    classifier_loss_xenium = 0
                # Put more weight on the classifier_loss_xenium
                classification_loss = classifier_loss_scrna + xenium_classifier_weight * classifier_loss_xenium
            else: # if there are no labels at all within a mini-batch
                classification_loss = 0

            kl_loss = torch.distributions.kl_divergence(
                torch.distributions.Normal(mu, torch.nn.functional.softplus(logstdev)),
                torch.distributions.Normal(0, 1)
            ).mean()

            # Sample positive and negative contrastive pairs
            z_anchors, _, _ = model.encode(t_train_cells[anchors].to(device))
            z_pos, _, _ = model.encode(t_train_cells[pos_samples].to(device))
            z_neg, _, _ = model.encode(t_train_cells[neg_samples[:, 0]].to(device))

            # Contrastive learning penalties
            con_loss = contrastive_loss(z_anchors, z_pos, z_neg, con_temp=con_temp).mean()

            # Let's just calculate the unscaled total_loss
            total_loss = (reconstruction_penalty * (recon_Xenium_loss + recon_scRNA_loss)
                            + (kl_penalty * kl_loss)
                            + (contrastive_penalty * con_loss)
                            + classification_loss)

            # Combine losses into a list
            current_losses = [recon_Xenium_loss, recon_scRNA_loss, kl_loss, classification_loss, con_loss]

            # Compute the normalized weighted total loss
            normalized_total_loss, normalized_losses_list = compute_weighted_total_loss(current_losses, loss_histories,
                                  loss_weights=[reconstruction_penalty, reconstruction_penalty, kl_penalty, classification_penalty, contrastive_penalty])
            # Add up reconstruction losses w/ KL loss, classification losses
            normalized_total_loss.backward()
            optimizer.step()

            # Let's keep track of z in mini-batch
            tensor_z_epoch = torch.cat((tensor_z_epoch, z.cpu().detach()), dim=0)
            z_modality_labels_epoch.append(batch_modalities.cpu().numpy())
            z_cell_type_labels_epoch.append(batch_labels.cpu().numpy())   # A lot of these cells will be -1
            z_cell_type_true_labels_epoch.append(batch_true_labels.cpu().numpy())

            # Log mini-batch losses to wandb
            wandb.log({"Mini-batch Average Z-embedding std": z.std(dim = 0).mean(),    # log the mean of the std of z's per mini-batch
                       "Mini-batch loss": normalized_total_loss,  # per mini-batch
                       "Mini-batch Contrastive loss": normalized_losses_list[4],  # per mini-batch
                       # "Mini-batch reconstruction Loss": recon_loss,  # per mini-batch
                       "Mini-batch recon scRNA loss": normalized_losses_list[1], # per mini-batch
                       "Mini-batch recon Xenium loss": normalized_losses_list[0], # per mini-batch
                       "Mini-batch scRNA data points": batch_modalities.cpu()[batch_modalities.cpu() == 1].sum(), # length of scRNA data points
                       # "Reconstruction Loss (Xenium)": epoch_total_recon_xenium_loss,  # per mini-batch
                       "Mini-batch KL loss": normalized_losses_list[2],
                       "Mini-batch Classification loss": normalized_losses_list[3]})

            # Accumulate losses for the epoch
            epoch_total_loss += normalized_total_loss.item()
            epoch_recon_loss += (normalized_losses_list[1])
            epoch_recon_xenium_loss += normalized_losses_list[0]
            epoch_total_ce_loss += normalized_losses_list[3]
            epoch_kl_loss += normalized_losses_list[2]
            epoch_con_loss += normalized_losses_list[4]

            epoch_unscaled_total_loss += total_loss.item()
            epoch_unscaled_recon_scrna_loss += recon_scRNA_loss
            epoch_unscaled_xenium_loss += recon_Xenium_loss
            epoch_unscaled_ce_loss += classification_loss
            epoch_unscaled_con_loss += con_loss.item()
            epoch_unscaled_kl_loss += kl_loss.item()

        # Store summed losses
        total_loss_arr.append(epoch_total_loss)

        # At every 10 epochs, let's run UMAP on z_embedding and visualize how it changes during training
        if (epoch % 5) == 0:
            # UMAP the input and return plot
            z_modality_labels_epoch = np.concatenate(z_modality_labels_epoch)
            z_cell_type_labels_epoch = np.concatenate(z_cell_type_labels_epoch)
            z_cell_type_true_labels_epoch = np.concatenate(z_cell_type_true_labels_epoch)
            # umap_learned_z = umap_and_plot(tensor_z_epoch, z_modality_labels_epoch, z_cell_type_labels_epoch, epoch, model_filename)
            # wandb.log({umap_learned_z: wandb.Image(umap_learned_z,
            #                                        caption=f"UMAP Plot of Z {epoch}",
            #                                        )})
            umap_learned_z_celltype = umap_and_plot(tensor_z_epoch, z_modality_labels_epoch, z_cell_type_true_labels_epoch, epoch, model_filename)
            # print(type(umap_learned_z))
            # Log the plot to wandb

            # wandb.log({
            #            umap_learned_z_celltype: wandb.Image(umap_learned_z_celltype,
            #                                                 caption=f"UMAP Plot of Z {epoch} with unmasked labels")
            #            })
        # Let's check on whether we're over-fitting by checking accuracy on validation set
        # Validate the model
        model.eval()  # Set model to evaluation mode
        val_accuracy = 0.0
        sc_val_accuracy = 0.0
        xenium_val_accuracy = 0.0
        val_classification_loss = 0.0  # Unscaled loss
        val_classifier_loss_scrna = 0.0 # Unscaled loss
        val_classifier_loss_xenium = 0.0 # Unscaled loss
        val_recon_scrna = 0.0   # Unscaled loss
        val_recon_Xenium = 0.0  # Unscaled loss
        val_kl_loss = 0.0
        val_con_loss = 0.0
        val_total_loss = 0.0    # Unsaled loss

        with torch.no_grad():  # No need to compute gradients during validation
            for (inputs, cell_type_labels, modality_labels, true_cell_type_labels), (v_anchors, v_pos_samples, v_neg_samples) in zip(validation_dataloader, val_contraster):
                inputs, cell_type_labels, modality_labels, true_cell_type_labels = inputs.to(device), cell_type_labels.to(
                    device), modality_labels.to(device), true_cell_type_labels.to(device)
                v_class_logits, v_z, v_mu, v_logstdev, v_recon_scrna, v_recon_xenium = model(inputs, modality_labels)  # Forward pass
                # Get class_preds and calculate accuracy
                class_preds = torch.nn.functional.softmax(v_class_logits, dim=-1)

                bool_mask = true_cell_type_labels != -1
                val_accuracy += calculate_accuracy(class_preds[bool_mask], true_cell_type_labels[bool_mask]) * inputs[bool_mask].size(0)
                # Let's look at the validation accuracy across data modalities:
                # Possible that w/i a mini-batch, there are no Xenium or scRNA data points
                sc_bool_mask = (bool_mask) & (modality_labels == 1)
                sc_val_accuracy += calculate_accuracy(class_preds[sc_bool_mask], true_cell_type_labels[sc_bool_mask]) \
                                   * inputs[sc_bool_mask].size(0)
                xenium_bool_mask = (bool_mask) & (modality_labels == 0)
                xenium_val_accuracy += calculate_accuracy(class_preds[xenium_bool_mask], true_cell_type_labels[xenium_bool_mask]) \
                                   * inputs[xenium_bool_mask].size(0)
                print(f"Number of Validation mini-batch labels: {sum(true_cell_type_labels != -1)}")
                print(f"Number of Validation mini-batch labels scRNA: {sum(sc_bool_mask)}")
                print(f"Number of Validation mini-batch labels Xenium: {sum(xenium_bool_mask)}")
                # Rather than calculating 1 loss, let's calculate for the technologies separately
                # Then use the same weight as in for the training loop
                if (cell_type_labels > -1).sum() > 0:
                    # Calculate the classification_loss for labeled scRNA data
                    minibatch_val_classifier_loss_scrna = torch.nn.functional.cross_entropy(v_class_logits[sc_bool_mask],
                                                                              true_cell_type_labels[sc_bool_mask])
                    # Keep track of the classification accuracy as minibatch of val set is traversed
                    val_classifier_loss_scrna += minibatch_val_classifier_loss_scrna
                    # Calculate the classification_loss for labeled Xenium data
                    if xenium_bool_mask.sum() > 0:
                        minibatch_val_classifier_loss_xenium = torch.nn.functional.cross_entropy(v_class_logits[xenium_bool_mask],
                                                                                   true_cell_type_labels[xenium_bool_mask])
                        val_classifier_loss_xenium += minibatch_val_classifier_loss_xenium
                    else:
                        minibatch_val_classifier_loss_xenium = 0
                        val_classifier_loss_xenium += minibatch_val_classifier_loss_xenium
                    # Put more weight on the classifier_loss_xenium
                    val_classification_loss += minibatch_val_classifier_loss_scrna + xenium_classifier_weight * minibatch_val_classifier_loss_xenium
                else:
                    mini_batch_val_classification_loss = 0
                    val_classification_loss += mini_batch_val_classification_loss

                mini_batch_val_recon_scRNA_loss = ((v_recon_scrna - inputs[modality_labels == 1]) ** 2).mean()
                mini_batch_val_recon_Xenium_loss = ((v_recon_xenium - inputs[modality_labels == 0]) ** 2).mean()

                mini_batch_val_kl_loss = torch.distributions.kl_divergence(
                    torch.distributions.Normal(v_mu, torch.nn.functional.softplus(v_logstdev)),
                    torch.distributions.Normal(0, 1)
                ).mean()

                # Sample positive and negative contrastive pairs
                v_z_anchors, _, _ = model.encode(t_test_cells[v_anchors].to(device))
                v_z_pos, _, _ = model.encode(t_test_cells[v_pos_samples].to(device))
                v_z_neg, _, _ = model.encode(t_test_cells[v_neg_samples[:, 0]].to(device))

                # Contrastive learning penalties
                mini_batch_val_con_loss = contrastive_loss(v_z_anchors, v_z_pos, v_z_neg, con_temp=con_temp).mean()

                val_con_loss += mini_batch_val_con_loss
                val_kl_loss += mini_batch_val_kl_loss
                val_recon_Xenium += mini_batch_val_recon_Xenium_loss
                val_recon_scrna += mini_batch_val_recon_scRNA_loss

                # After the mini_batch losses are added across the entire validation set, weigh them according to get the total_loss
                val_total_loss = (reconstruction_penalty * (val_recon_Xenium + val_recon_scrna)
                              + val_kl_loss
                              + (contrastive_penalty * val_con_loss)
                              + (classification_penalty * val_classification_loss))

        print(f"Number of Validation labels: {np.sum(validation_true_cell_types != -1)}")
        print(f"Number of Validation labels scRNA: {np.sum((validation_modalities == 1) & (validation_true_cell_types != -1))}")
        print(f"Number of Validation Xenium: {np.sum((validation_modalities == 0) & (validation_true_cell_types != -1))}")

        val_accuracy /= np.sum(validation_true_cell_types != -1)       # Divide by the number of labeled data points, not all data points
        sc_val_accuracy /= np.sum((validation_modalities == 1) & (validation_true_cell_types != -1))         # Divide by the number of labeled single-cell data points
        xenium_val_accuracy /= np.sum((validation_modalities == 0) & (validation_true_cell_types != -1))     # Divide by the number of labeled Xenium data points


        # Log losses to wandb
        wandb.log({"Scaled Total loss": epoch_total_loss,  # per epoch
                   "Scaled Contrastive loss": epoch_con_loss,   # per epoch
                   "Scaled Reconstruction Loss (scRNA)": epoch_recon_loss,  # per epoch
                   "Scaled Reconstruction Loss (Xenium)": epoch_recon_xenium_loss,  # per epoch
                   "Scaled KL loss": epoch_kl_loss,
                   "Scaled Classification loss": epoch_total_ce_loss,

                   "Total loss": epoch_unscaled_total_loss, # per epoch
                   "Contrastive loss": epoch_unscaled_con_loss, # per epoch
                    "Reconstruction Loss (scRNA)": epoch_unscaled_recon_scrna_loss,
                    "Reconstruction Loss (Xenium)": epoch_unscaled_xenium_loss,
                    "KL loss": epoch_unscaled_kl_loss,
                    "Classification loss": epoch_unscaled_ce_loss,

                   "Overall validation accuracy": val_accuracy,
                   "Validation accuracy: scRNA": sc_val_accuracy,
                   "Validation accuracy: Xenium": xenium_val_accuracy,
                   "Validation CE loss": val_classification_loss,
                   "Validation CE loss: scRNA": val_classifier_loss_scrna,
                   "Validation CE loss: Xenium": val_classifier_loss_xenium,
                   "Validation Total Loss": val_total_loss,
                   "Validation Contrastive Loss": val_con_loss,
                   "Validation KL Loss": val_kl_loss,
                   "Validation Recon Xenium": val_recon_Xenium,
                   "Validation Recon scRNA": val_recon_scrna})

        # Check early stopping using cross-entropy loss instead of accuracy
        early_stopping(val_total_loss, model)
        if early_stopping.early_stop:
            print('Early stopping')
            break

    # Save the learned parameters of the model
    torch.save({
        'epoch': n_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': normalized_total_loss,
        # These are hyper-parameter sweep metrics-of-interest
        'val_accuracy': val_accuracy,
        'val_total_loss': val_total_loss,
        'val_con_loss': val_con_loss, # to compare across hyper-parameter sweeps
        'val_recon_Xenium': val_recon_Xenium,
        'val_recon_scrna': val_recon_scrna
    },
        f'{model_filename}_vae_model.pth')

    # Save the metrics to evaluate the hyperparameter sweep
    eval_metrics_dict = {"val_accuracy": val_accuracy,              # Evaluate performance on classification
                         "val_accuracy_scrna": sc_val_accuracy,
                         "val_accuracy_Xenium": xenium_val_accuracy,
                         "val_con_loss": val_con_loss,              # Evaluate contrastive loss (comparable across sweeps
                         "val_recon_Xenium": val_recon_Xenium,      # MSE on Xenium reconstruction
                         "val_recon_scrna": val_recon_scrna}        # MSE on scRNA reconstruction

    # Save the dictionary to a file
    with open(f"{model_filename}/eval_metrics.pkl", "wb") as f:
        pickle.dump(eval_metrics_dict, f)

    return model, gene_idx, t_train_cells, t_train_modalities,  t_train_cell_types, t_test_cells, t_test_modalities, t_test_cell_types, t_test_true_cell_types, test_idx


if __name__ == '__main__':
    # Log into wandb
    wandb.login()
    # n_samples = 1000
    # n_genes = 300

    # Load in real data
    # cells = np.load('/Users/jacquelinechou/240819_small_dataset_Xenium/concat_logtrnsformed_no_nans.npy')
    # cell_type_labels = np.load('/Users/jacquelinechou/240819_small_dataset_Xenium/labels_numeric.npy', allow_pickle= True)
    # modality_labels = np.load('/Users/jacquelinechou/240819_small_dataset_Xenium/modality_labels_numeric.npy', allow_pickle= True)
    parser = argparse.ArgumentParser(description='Load data from multiple files.')
    parser.add_argument('cells_file', type=str, help='Path to the cells file')
    parser.add_argument('labels_file', type=str, help='Path to the cell type labels file')
    parser.add_argument('modality_file', type=str, help='Path to the modality labels file')
    parser.add_argument('n_epochs', type = int, default = 1300, help='Number of epochs to train')
    parser.add_argument('learning_rate', type = float, help='Learning rate')
    parser.add_argument('batch_size', type = int, help='Batch size')
    parser.add_argument('--model_filename', type = str, help = 'The output file path where the results will be saved')
    parser.add_argument('--input-dir', type=str, help = 'The directory where the input files are') # put in the bound foldername
    parser.add_argument('--output-dir', type=str, help='The directory where the results will be saved') # put in the bound foldername/vae_amp_xenium/date_experiments/
    parser.add_argument('label_pct', type=float, help='Percentage of labeled cell-types')
    parser.add_argument('xenium_classifier_weight', type=float, help='Weight on Xenium classification loss')

    # Hyper-parameter tuning
    parser.add_argument('--reconstruction_penalty', type=float, help= 'Reconstruction weight')
    parser.add_argument('--contrastive_penalty', type=float, help='Contrastive Loss weight')
    parser.add_argument('--classification_penalty', type=float, help='Classification weight')
    parser.add_argument('--kl_penalty', type=float, help='KL Loss weight')   # Need a value, since it was hard-coded as 1

    # Parse the command-line arguments
    args = parser.parse_args()

    # Load the data from the specified files
    cells = np.load(os.path.join(args.input_dir, args.cells_file))
    cell_type_labels = np.load(os.path.join(args.input_dir, args.labels_file), allow_pickle=True)
    modality_labels = np.load(os.path.join(args.input_dir, args.modality_file), allow_pickle=True)

    # Print the shapes of the loaded arrays (for debugging)
    print(f"Cells shape: {cells.shape}")
    print(f"Cell type labels shape: {cell_type_labels.shape}")
    print(f"Modality labels shape: {modality_labels.shape}")
    print(f"n_epochs: {args.n_epochs}")
    print(f"learning_rate: {args.learning_rate}")
    print(f"The output file will be saved to: {os.path.join(args.output_dir, args.model_filename)}")
    print(f"Cell-type label pct: {args.label_pct}")
    print(f"Xenium classifier weight: {args.xenium_classifier_weight}")

    output_file = os.path.join(args.output_dir, args.model_filename)
    # Check if the directory does not exist
    if not os.path.exists(output_file):
        # Create the directory, including any necessary parent directories
        os.makedirs(output_file)
        print(f'Directory created: {output_file}')
    else:
        print(f'Directory already exists: {output_file}')

    # Plot to show the batch effects
    umapper = UMAP()
    xy = umapper.fit_transform(cells)
    fig, axarr = plt.subplots(1, 2, figsize=(10 * 1.5, 5 * 1.5), sharex=True, sharey=True)
    axarr[0].scatter(xy[:, 0], xy[:, 1], s=2, c=modality_labels)
    axarr[0].set_title('Modalities')
    scatter = axarr[1].scatter(xy[:, 0], xy[:, 1], s=2, c = pd.Series(cell_type_labels).astype('category').cat.codes, cmap = 'tab10')
    axarr[1].set_title('Cell types')
    handles = [mpatches.Patch(color=scatter.cmap(scatter.norm(code)), label=f'{label}') for code, label in zip(pd.Series(cell_type_labels).astype('category').cat.codes.unique(),
                                                                                                               np.unique(cell_type_labels))]
    axarr[1].legend(handles=handles, title="Cell Types")


    plt.savefig(f'{output_file}/input_data_umap.png', bbox_inches='tight')
    plt.close()

    plt.hist(cells.reshape(-1), #bins=np.linspace(0, 50, 51)
             )
    plt.savefig(f'{output_file}/input_counts_histogram.png', bbox_inches='tight')
    plt.close()
    run = wandb.init(
        # Set the project where this run will be logged
        project="Multimodal_VAE_classifier",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": args.learning_rate,
            "epochs": args.n_epochs,
            "model": 'VAE with classifier for both scRNA / Xenium',
            "latent_dim": 20,
            "Weight on Xenium classification loss": args.xenium_classifier_weight,
            # Log the hyperparameters used in the run to wandb
            "Hyperparameter: Classification Loss weight": args.classification_penalty,
            "Hyperparameter: Contrastive Loss weight": args.contrastive_penalty,
            "Hyperparameter: Reconstruction Loss weight": args.reconstruction_penalty,
            "Hyperparameter: KL Loss weight": args.kl_penalty
        },
    )
    wandb.log({'Input data UMAP': wandb.Image(f'{output_file}/input_data_umap.png',
                                                      caption='Input- umap'),
               'Input histogram': wandb.Image(f'{output_file}/input_counts_histogram.png',
                                                       caption="Input- histogram"),
               # 'confusion matrix (Xenium)': wandb.Image('classifier_cm_xenium.png',
               #                                       caption="Confusion matrix on test set (Xenium))")
               })

    (model, gene_idx, t_train_data, t_train_modality_labels, t_train_cell_types,
     t_test_cells, t_test_modalities, t_test_cell_types, t_test_true_cell_types, test_idx) = fit_vae(cells, cell_type_labels, \
                            modality_labels, n_epochs=args.n_epochs, reconstruction_penalty= args.reconstruction_penalty,     \
                            contrastive_penalty=args.contrastive_penalty, classification_penalty = args.classification_penalty, \
                            kl_penalty = args.kl_penalty,
                            model_filename=output_file, cell_type_label_pct=args.label_pct, learning_rate=args.learning_rate, \
                            batch_size=args.batch_size, patience=10, xenium_classifier_weight=args.xenium_classifier_weight)

    # Save the train_data and train_modality_labels
    np.save(f'{output_file}/classifier_train_data.npy', t_train_data.cpu().numpy())
    np.save(f'{output_file}/classifier_train_modality.npy', t_train_modality_labels.cpu().numpy())
    np.save(f'{output_file}/classifier_train_cell_types.npy', t_train_cell_types.cpu().numpy())
    np.save(f'{output_file}/classifier_test_data.npy', t_test_cells.cpu().numpy())
    np.save(f'{output_file}/classifier_test_modalities.npy', t_test_modalities.cpu().numpy())
    np.save(f'{output_file}/classifier_test_cell_types.npy', t_test_cell_types.cpu().numpy())
    # Save the true cell-type labels in test set
    np.save(f'{output_file}/classifier_test_true_cell_types.npy', t_test_true_cell_types.cpu().numpy())

    # Save the test set indices, so we can compare the pipeline results to model-derived labels
    np.save(f'{output_file}/classifier_test_indices.npy', test_idx)

    # Hyper-parameter sweep: Save the metrics of interest and aggregate the metrics across the different batches of hyperparameters
    # Then pick the optimal model and run on test dataset
    # Predict on test cells: just to see if it gives me the same embedding and to look at the classification
    # model.eval()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # # t_cells, t_modality_labels = torch.FloatTensor(cells), torch.LongTensor(modality_labels)
    # class_logits, z, mu, logstdev, recon_scrna, recon_xenium = model(t_test_cells.to(device), t_test_modalities.to(device))
    # class_preds = torch.argmax(torch.nn.functional.softmax(class_logits, dim=-1), dim=-1).cpu().detach().numpy()
    #
    # # Save class logits
    # # Save class predictions for comparison post-hoc
    # np.save(f'{output_file}/classifier_class_logits.npy', class_logits.cpu().detach().numpy())
    # np.save(f'{output_file}/classifier_class_preds.npy', class_preds)
    # print(f'unique class_preds: {np.unique(class_preds)}')
    # print(f'class_logits: {class_logits.cpu().detach().numpy()}')
    #
    # # Make a confusion matrix
    # bool_mask = t_test_true_cell_types != -1
    # cm = confusion_matrix(y_true=np.array(t_test_true_cell_types.cpu())[bool_mask], y_pred=class_preds[bool_mask])
    # # cm = confusion_matrix(y_true=np.array(cell_type_labels), y_pred=class_preds)
    # print(cm)
    #
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    # plt.xlabel('Predicted Label')
    # plt.ylabel('True Label')
    # plt.title('Confusion Matrix')
    # plt.savefig(f'{output_file}/classifier_cm.png')
    #
    # sc_bool_mask = (bool_mask) & (t_test_modalities.cpu() == 1)
    # cm_scrna = confusion_matrix(y_true=np.array(t_test_true_cell_types.cpu())[sc_bool_mask],
    #                             y_pred=class_preds[sc_bool_mask])
    # print(f'cm_scrna: {cm_scrna}')
    #
    # xenium_bool_mask = (bool_mask) & (t_test_modalities.cpu() == 0)
    # cm_xenium = confusion_matrix(y_true=np.array(t_test_true_cell_types.cpu())[xenium_bool_mask],
    #                              y_pred=class_preds[xenium_bool_mask])
    # print(f'cm_xenium: {cm_xenium}')
    #
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm_scrna, annot=True, fmt='d', cmap='Blues', cbar=False)
    # plt.xlabel('Predicted Label')
    # plt.ylabel('True Label')
    # plt.title('Confusion Matrix: scRNA only')
    # plt.savefig(f'{output_file}/classifier_cm_scrna.png')
    #
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm_xenium, annot=True, fmt='d', cmap='Blues', cbar=False)
    # plt.xlabel('Predicted Label')
    # plt.ylabel('True Label')
    # plt.title('Confusion Matrix: Xenium only')
    # plt.savefig('classifier_cm_xenium.png')
    #
    # wandb.log({'confusion matrix (both)': wandb.Image(f'{output_file}/classifier_cm.png',
    #                                                   caption='Confusion matrix on test set (Both modalities)'),
    #            'confusion matrix (scRNA)': wandb.Image(f'{output_file}/classifier_cm_scrna.png',
    #                                                 caption="Confusion matrix on test set (scRNA)"),
    #            'confusion matrix (Xenium)': wandb.Image('classifier_cm_xenium.png',
    #                                                  caption="Confusion matrix on test set (Xenium))")
    # })
    # # Plot to show the learned embeddings
    # umapper = UMAP()
    # xy = umapper.fit_transform(t_test_cells.cpu())   # Input data
    # xy_pred = umapper.fit_transform(z.cpu().detach().numpy())   # Latent variable
    # fig, axarr = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    # axarr[0, 0].scatter(xy[:, 0], xy[:, 1], s=2, c=t_test_modalities)
    # axarr[0, 0].set_title('Raw modalities')
    # axarr[0, 1].scatter(xy[:, 0], xy[:, 1], s=2, c=t_test_cell_types, cmap='tab10')
    # axarr[0, 1].set_title('Raw cell types')
    # axarr[1, 0].scatter(xy_pred[:, 0], xy_pred[:, 1], s=2, c=t_test_modalities)
    # axarr[1, 0].set_title('VAE modalities')
    # axarr[1, 1].scatter(xy_pred[:, 0], xy_pred[:, 1], s=2, c=t_test_true_cell_types,
    #                     cmap='tab10')
    # axarr[1, 1].set_title('VAE cell types')
    # plt.savefig(f'{output_file}/test_data_classifier.png', bbox_inches='tight')
    # plt.close()
    #
    # wandb.log({'umap_learned_z of test set': wandb.Image(f'{output_file}/test_data_classifier.png',
    #                                                caption= "UMAP Plot of Z (test set)",
    #                                                )})
    wandb.finish()