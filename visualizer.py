import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA

def plot_gnn_pca(gnn_embeddings, gnn_idx_to_node, output_path):
    """
    Takes the final GNN embeddings and idx_to_node map,
    runs PCA, and saves an interactive plot.
    """
    print("\n--- Starting PCA on GNN Node Embeddings ---")
    
    # 1. Get embeddings from GPU (if needed) and convert to numpy
    z_numpy = gnn_embeddings.cpu().detach().numpy()
    
    # 2. Run PCA
    pca_gnn = PCA(n_components=3)
    pca_gnn_results = pca_gnn.fit_transform(z_numpy)
    
    # Print the explained variance
    print(f"GNN-PCA Explained Variance Ratio:")
    print(f"  PC1: {pca_gnn.explained_variance_ratio_[0]:.4f}")
    print(f"  PC2: {pca_gnn.explained_variance_ratio_[1]:.4f}")
    print(f"  PC3: {pca_gnn.explained_variance_ratio_[2]:.4f}")
    print(f"Total (PC1-3): {(np.sum(pca_gnn.explained_variance_ratio_)):.4f}")
    
    # 3. Create DataFrame
    node_names_sorted = [gnn_idx_to_node[i] for i in range(len(gnn_idx_to_node))]
    
    pca_gnn_df = pd.DataFrame(
        data=pca_gnn_results,
        columns=['pc1', 'pc2', 'pc3'],
        index=node_names_sorted
    )
    pca_gnn_df['node'] = pca_gnn_df.index
    # Extract team and year for better plotting
    pca_gnn_df['team'] = pca_gnn_df['node'].apply(lambda x: "_".join(x.split('_')[:-1]))
    pca_gnn_df['year'] = pca_gnn_df['node'].apply(lambda x: x.split('_')[-1])
    
    
    # 5. Plot with Plotly
    print(f"Generating interactive GNN-PCA plot (saving to {output_path})...")
    
    fig_gnn = px.scatter(
        pca_gnn_df,
        x='pc1',
        y='pc2',
        color='year',           # <-- Color by year
        hover_data=['node', 'pc1', 'pc2', 'pc3'], #<-- Data on hover
        title=f"PCA of GNN (Team, Year) Node Embeddings (All Seasons)",
        labels={'pc1': 'GNN Principal Component 1', 
                'pc2': 'GNN Principal Component 2',
                'pc3': 'GNN Principal Component 3'},
        template="plotly_white"
    )

    # Add zero-lines for reference
    fig_gnn.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
    fig_gnn.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.5)

    # Update layout
    fig_gnn.update_layout(
        title_font_size=20,
        hovermode="closest"
    )
    fig_gnn.update_traces(marker=dict(size=7, opacity=0.7))

    # Save to an interactive HTML file
    fig_gnn.write_html(output_path)
    
    print(f"GNN-PCA plot saved to {output_path}. Open this file in your browser.")