# --- MKL Warning Suppression ---
# MUST come before numpy/torch imports
import os
os.environ["MKL_DEBUG_CPU_TYPE"] = "5"
os.environ["MKL_VERBOSE"] = "0"
# --- End Suppression ---

import sys
import torch # For checking HAVE_PYG

# --- NEW: Import our custom modules ---
import data_loader
import gnn_models
import gnn_trainer
import visualizer

# Check for PyG
try:
    import torch_geometric
except ImportError:
    print("--- GNN ERROR ---")
    print("PyTorch or PyTorch Geometric not found.")
    print("Please install them to run the GNN section:")
    print("pip install torch torch-geometric")
    print("-----------------")
    HAVE_PYG = False
else:
    HAVE_PYG = True
# --- End imports ---

# --- 1. Configuration ---
# Use a relative data directory or allow override via environment variable
SCRIPT_DIR = os.environ.get('DATA_DIR', os.path.join(os.getcwd(), 'data'))
# This is now just a reference, as all data will be split
SEASON_YEAR = 2024
GNN_PCA_PLOT_PATH = os.path.join(SCRIPT_DIR, "mbb_gnn_pca_cluster_plot_ALL_SEASONS_INTERACTIVE.html")

def main():
    """
    Main execution flow for the GNN model.
    """
    if not HAVE_PYG:
        print("Required PyTorch libraries not found. Exiting.")
        sys.exit()

    # --- 1. Loading and Preprocessing Data ---
    print("--- 1. Loading and Preprocessing Data ---")
    try:
        data_payload = data_loader.load_and_preprocess_data(SCRIPT_DIR, SEASON_YEAR)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit()
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        sys.exit()

    # --- 2. Running GNN Training ---
    print("\n--- 2. Running GNN Training ---")
    gnn_results = gnn_trainer.run_gnn_training(
        all_games_df=data_payload['all_games_df'],
        all_nodes=data_payload['all_nodes'],
        node_to_idx=data_payload['node_to_idx'],
        src_cols=data_payload['src_cols'],
        dst_cols=data_payload['dst_cols']
    )

    # --- 3. Visualizing Results ---
    if gnn_results['model'] is not None:
        print("\n--- 3. Visualizing GNN Embeddings ---")
        visualizer.plot_gnn_pca(
            gnn_embeddings=gnn_results['embeddings'],
            gnn_idx_to_node=gnn_results['idx_to_node'],
            output_path=GNN_PCA_PLOT_PATH
        )
    else:
        print("\nSkipping visualization, GNN training failed.")

    print("\nScript finished.")

if __name__ == "__main__":
    main()