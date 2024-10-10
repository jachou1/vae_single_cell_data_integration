import argparse
import os

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import wandb
from matplotlib import pyplot as plt
from umap import UMAP

from vae_single_cell_data_integration.multimodal_archetypal_classifier import fit_vae


def get_parser():
    parser = argparse.ArgumentParser(description="Load data from multiple files.")
    parser.add_argument("--cells-file", type=str, help="Path to the cells file")
    parser.add_argument(
        "--labels-file", type=str, help="Path to the cell type labels file"
    )
    parser.add_argument(
        "--modality-file", type=str, help="Path to the modality labels file"
    )
    parser.add_argument(
        "--n-epochs", type=int, default=1300, help="Number of epochs to train"
    )
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument(
        "--model-filename",
        type=str,
        help="The output file path where the results will be saved",
    )
    parser.add_argument(
        "--input-dir", type=str, help="The directory where the input files are"
    )  # put in the bound foldername
    parser.add_argument(
        "--output-dir", type=str, help="The directory where the results will be saved"
    )  # put in the bound foldername/vae_amp_xenium/date_experiments/
    parser.add_argument(
        "--label-pct", type=float, help="Percentage of labeled cell-types"
    )
    parser.add_argument(
        "--xenium-classifier-weight",
        type=float,
        help="Weight on Xenium classification loss",
    )

    # Hyper-parameter tuning
    parser.add_argument(
        "--reconstruction-penalty", type=float, help="Reconstruction weight"
    )
    parser.add_argument(
        "--contrastive-penalty", type=float, help="Contrastive Loss weight"
    )
    parser.add_argument(
        "--classification-penalty", type=float, help="Classification weight"
    )
    parser.add_argument(
        "--kl-penalty", type=float, help="KL Loss weight"
    )  # Need a value, since it was hard-coded as 1
    parser.add_argument(
        "--device", type=str,
        help="Device to run the model on",
        default="cpu",
        choices=["cpu", "cuda"]
    )

    # Parse the command-line arguments
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Log into wandb
    wandb.login()
    # n_samples = 1000
    # n_genes = 300

    # Load the data from the specified files
    cells = np.load(os.path.join(args.input_dir, args.cells_file))
    cell_type_labels = np.load(
        os.path.join(args.input_dir, args.labels_file), allow_pickle=True
    )
    modality_labels = np.load(
        os.path.join(args.input_dir, args.modality_file), allow_pickle=True
    )

    # Print the shapes of the loaded arrays (for debugging)
    print(f"Cells shape: {cells.shape}")
    print(f"Cell type labels shape: {cell_type_labels.shape}")
    print(f"Modality labels shape: {modality_labels.shape}")
    print(f"n_epochs: {args.n_epochs}")
    print(f"learning_rate: {args.learning_rate}")
    print(
        f"The output file will be saved to: {os.path.join(args.output_dir, args.model_filename)}"
    )
    print(f"Cell-type label pct: {args.label_pct}")
    print(f"Xenium classifier weight: {args.xenium_classifier_weight}")

    output_dir = args.output_dir
    # Check if the directory does not exist
    if not os.path.exists(output_dir):
        # Create the directory, including any necessary parent directories
        os.makedirs(output_dir)
        print(f"Directory created: {output_dir}")
    else:
        print(f"Directory already exists: {output_dir}")

    # Plot to show the batch effects
    umapper = UMAP()
    xy = umapper.fit_transform(cells)
    fig, axarr = plt.subplots(
        1, 2, figsize=(10 * 1.5, 5 * 1.5), sharex=True, sharey=True
    )
    axarr[0].scatter(xy[:, 0], xy[:, 1], s=2, c=modality_labels)
    axarr[0].set_title("Modalities")
    scatter = axarr[1].scatter(
        xy[:, 0],
        xy[:, 1],
        s=2,
        c=pd.Series(cell_type_labels).astype("category").cat.codes,
        cmap="tab10",
    )
    axarr[1].set_title("Cell types")
    handles = [
        mpatches.Patch(color=scatter.cmap(scatter.norm(code)), label=f"{label}")
        for code, label in zip(
            pd.Series(cell_type_labels).astype("category").cat.codes.unique(),
            np.unique(cell_type_labels),
        )
    ]
    axarr[1].legend(handles=handles, title="Cell Types")

    plt.savefig(f"{output_dir}/input_data_umap.png", bbox_inches="tight")
    plt.close()

    plt.hist(cells.reshape(-1))
    plt.savefig(f"{output_dir}/input_counts_histogram.png", bbox_inches="tight")
    plt.close()
    wandb.init(
        # Set the project where this run will be logged
        project="Multimodal_VAE_classifier",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": args.learning_rate,
            "epochs": args.n_epochs,
            "model": "VAE with classifier for both scRNA / Xenium",
            "latent_dim": 20,
            "Weight on Xenium classification loss": args.xenium_classifier_weight,
            # Log the hyperparameters used in the run to wandb
            "Hyperparameter: Classification Loss weight": args.classification_penalty,
            "Hyperparameter: Contrastive Loss weight": args.contrastive_penalty,
            "Hyperparameter: Reconstruction Loss weight": args.reconstruction_penalty,
            "Hyperparameter: KL Loss weight": args.kl_penalty,
        },
    )
    wandb.log(
        {
            "Input data UMAP": wandb.Image(
                f"{output_dir}/input_data_umap.png", caption="Input- umap"
            ),
            "Input histogram": wandb.Image(
                f"{output_dir}/input_counts_histogram.png", caption="Input- histogram"
            ),
            # 'confusion matrix (Xenium)': wandb.Image('classifier_cm_xenium.png',
            #                                       caption="Confusion matrix on test set (Xenium))")
        }
    )

    (
        model,
        gene_idx,
        t_train_data,
        t_train_modality_labels,
        t_train_cell_types,
        t_test_cells,
        t_test_modalities,
        t_test_cell_types,
        t_test_true_cell_types,
        test_idx,
    ) = fit_vae(
        cells,
        cell_type_labels,
        modality_labels,
        n_epochs=args.n_epochs,
        reconstruction_penalty=args.reconstruction_penalty,
        contrastive_penalty=args.contrastive_penalty,
        classification_penalty=args.classification_penalty,
        kl_penalty=args.kl_penalty,
        model_filename=output_dir,
        cell_type_label_pct=args.label_pct,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        patience=10,
        xenium_classifier_weight=args.xenium_classifier_weight,
        device=args.device
    )

    # Save the train_data and train_modality_labels
    np.save(f"{output_dir}/classifier_train_data.npy", t_train_data.cpu().numpy())
    np.save(
        f"{output_dir}/classifier_train_modality.npy",
        t_train_modality_labels.cpu().numpy(),
    )
    np.save(
        f"{output_dir}/classifier_train_cell_types.npy",
        t_train_cell_types.cpu().numpy(),
    )
    np.save(f"{output_dir}/classifier_test_data.npy", t_test_cells.cpu().numpy())
    np.save(
        f"{output_dir}/classifier_test_modalities.npy", t_test_modalities.cpu().numpy()
    )
    np.save(
        f"{output_dir}/classifier_test_cell_types.npy", t_test_cell_types.cpu().numpy()
    )
    # Save the true cell-type labels in test set
    np.save(
        f"{output_dir}/classifier_test_true_cell_types.npy",
        t_test_true_cell_types.cpu().numpy(),
    )

    # Save the test set indices, so we can compare the pipeline results to model-derived labels
    np.save(f"{output_dir}/classifier_test_indices.npy", test_idx)

    wandb.finish()
