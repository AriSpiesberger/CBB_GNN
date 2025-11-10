import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch_geometric.data import HeteroData
from sklearn.metrics import mean_squared_error, accuracy_score

# Import our custom models
from gnn_models import GNNModel

def train_step(model, encoder_graph, supervision_graph, optimizer, loss_fn):
    """
    This is the new, leak-free training function.
    """
    model.train()
    optimizer.zero_grad()
    
    # 1. ENCODER STEP: Create z embeddings using the message-passing graph
    z = model.encode(
        encoder_graph.x_dict, 
        encoder_graph.edge_index_dict, 
        encoder_graph.edge_attr_dict
    )
    
    # 2. DECODER STEP: Predict on the disjoint supervision graph
    loss = 0
    total_acc = 0
    num_types = 0
    
    for edge_type in supervision_graph.edge_label_index_dict.keys():
        edge_label_index = supervision_graph[edge_type].edge_label_index
        labels = supervision_graph[edge_type].edge_label
        
        if labels.numel() == 0:
            continue
            
        pred = model.decoder(z, edge_label_index)
        type_loss = loss_fn(pred, labels)
        loss += type_loss
        
        with torch.no_grad():
            pred_sign = (pred > 0).float().cpu().numpy()
            true_sign = (labels > 0).float().cpu().numpy()
            total_acc += accuracy_score(true_sign, pred_sign)
        
        num_types += 1
    
    if num_types == 0:
        return 0.0, 1.0
        
    loss.backward()
    optimizer.step()
    
    # --- FIX: Added .detach() to prevent user warning ---
    return loss.detach().item(), (total_acc / num_types)


@torch.no_grad()
def eval_step(model, encoder_graph, supervision_graph, loss_fn):
    """
    This is the leak-free test/validation function.
    """
    model.eval()
    
    # 1. ENCODER STEP: Create z embeddings
    z = model.encode(
        encoder_graph.x_dict, 
        encoder_graph.edge_index_dict, 
        encoder_graph.edge_attr_dict
    )
    
    # 2. DECODER STEP: Predict on the disjoint supervision graph
    total_loss = 0
    total_acc = 0
    total_rmse = 0
    num_types = 0
    
    for edge_type in supervision_graph.edge_label_index_dict.keys():
        edge_label_index = supervision_graph[edge_type].edge_label_index
        labels = supervision_graph[edge_type].edge_label
        
        if labels.numel() == 0:
            continue
            
        pred = model.decoder(z, edge_label_index)
        type_loss = loss_fn(pred, labels)
        total_loss += float(type_loss)
        total_rmse += float(torch.sqrt(type_loss))
        
        with torch.no_grad():
            pred_sign = (pred > 0).float().cpu().numpy()
            true_sign = (labels > 0).float().cpu().numpy()
            total_acc += accuracy_score(true_sign, pred_sign)
        
        num_types += 1
    
    if num_types == 0:
        return 0.0, 0.0, 1.0

    avg_loss = total_loss / num_types
    avg_rmse = total_rmse / num_types
    avg_acc = total_acc / num_types
    
    return avg_loss, avg_rmse, avg_acc

# --- Main GNN Execution ---
def run_gnn_training(all_games_df, all_nodes, node_to_idx, src_cols, dst_cols):
    
    idx_to_node = {v: k for k, v in node_to_idx.items()}
    num_nodes = len(all_nodes)
    EDGE_FEATURE_DIM = len(src_cols) + len(dst_cols)
    print(f"Using global map of {num_nodes} (Team, Year) nodes.")
    print(f"Encoder will use {EDGE_FEATURE_DIM} edge features.")

    # 2. Helper function to build HeteroData objects
    def build_hetero_data(games_list, is_encoder_graph=False):
        data = HeteroData()
        data['team_year'].x = torch.arange(num_nodes, dtype=torch.long)
        
        src_wins, dst_wins, label_wins, attr_wins = [], [], [], []
        src_loss, dst_loss, label_loss, attr_loss = [], [], [], []
        
        valid_games = games_list[
            games_list['home_team_year'].isin(node_to_idx) &
            games_list['away_team_year'].isin(node_to_idx)
        ]
        
        home_feats_all = valid_games[src_cols].values.astype(np.float32)
        away_feats_all = valid_games[dst_cols].values.astype(np.float32)
        
        for i, game in enumerate(valid_games.itertuples()):
            home_idx = node_to_idx[game.home_team_year]
            away_idx = node_to_idx[game.away_team_year]
            adj_val = game.adjacency_value
            
            home_feats = home_feats_all[i]
            away_feats = away_feats_all[i]
            
            if game.mov > 0: # Home win
                src_wins.append(home_idx); dst_wins.append(away_idx); label_wins.append(adj_val)
                attr_wins.append(np.concatenate([home_feats, away_feats]))
                src_loss.append(away_idx); dst_loss.append(home_idx); label_loss.append(-adj_val)
                attr_loss.append(np.concatenate([away_feats, home_feats]))
            else: # Home loss
                src_loss.append(home_idx); dst_loss.append(away_idx); label_loss.append(adj_val)
                attr_loss.append(np.concatenate([home_feats, away_feats]))
                src_wins.append(away_idx); dst_wins.append(home_idx); label_wins.append(-adj_val)
                attr_wins.append(np.concatenate([away_feats, home_feats]))
        
        dummy_attr = np.zeros(EDGE_FEATURE_DIM, dtype=np.float32)
        if not src_wins: 
            src_wins, dst_wins, label_wins, attr_wins = [0], [0], [0.0], [dummy_attr]
        if not src_loss:
            src_loss, dst_loss, label_loss, attr_loss = [0], [0], [0.0], [dummy_attr]
            
        edge_index_wins = torch.tensor(np.array([src_wins, dst_wins]), dtype=torch.long)
        edge_label_wins = torch.tensor(np.array(label_wins), dtype=torch.float)
        edge_attr_wins = torch.tensor(np.array(attr_wins), dtype=torch.float)

        edge_index_loss = torch.tensor(np.array([src_loss, dst_loss]), dtype=torch.long)
        edge_label_loss = torch.tensor(np.array(label_loss), dtype=torch.float)
        edge_attr_loss = torch.tensor(np.array(attr_loss), dtype=torch.float)
        
        if is_encoder_graph:
            data['team_year', 'wins_against', 'team_year'].edge_index = edge_index_wins
            data['team_year', 'wins_against', 'team_year'].edge_attr = edge_attr_wins
            data['team_year', 'loses_to', 'team_year'].edge_index = edge_index_loss
            data['team_year', 'loses_to', 'team_year'].edge_attr = edge_attr_loss
        else:
            data['team_year', 'wins_against', 'team_year'].edge_label_index = edge_index_wins
            data['team_year', 'wins_against', 'team_year'].edge_label = edge_label_wins
            data['team_year', 'loses_to', 'team_year'].edge_label_index = edge_index_loss
            data['team_year', 'loses_to', 'team_year'].edge_label = edge_label_loss
            
        return data

    # 3. Create DISJOINT 80/10/10 Splits
    print(f"Performing 80/10/10 game-level split on all {len(all_games_df)} games...")
    games_df_shuffled = all_games_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    num_games = len(games_df_shuffled)
    num_val = int(num_games * 0.1)
    num_test = int(num_games * 0.1)
    num_train = num_games - num_val - num_test
    
    train_games_df = games_df_shuffled.iloc[:num_train]
    val_games_df = games_df_shuffled.iloc[num_train : num_train + num_val]
    test_games_df = games_df_shuffled.iloc[num_train + num_val :]
    
    print(f"  Total Games: {num_games}")
    print(f"  Train Set: {len(train_games_df)} games (will be split for encoder/decoder)")
    print(f"  Val Set:   {len(val_games_df)} games")
    print(f"  Test Set:  {len(test_games_df)} games")
    
    # 4. Split the 80% Train set into Encoder/Decoder sets
    print("Splitting train set into encoder/decoder graphs...")
    train_encoder_df, train_supervision_df = train_test_split(
        train_games_df, 
        test_size=0.1, # 10% of the training set for supervision
        random_state=42
    )
    
    print(f"  Encoder Graph (90% of 80%): {len(train_encoder_df)} games")
    print(f"  Decoder Train Supervision (10% of 80%): {len(train_supervision_df)} games")

    # 5. Build Data Objects
    print(f"Building HeteroData objects...")
    encoder_graph = build_hetero_data(train_encoder_df, is_encoder_graph=True)
    train_supervision_graph = build_hetero_data(train_supervision_df, is_encoder_graph=False)
    val_supervision_graph = build_hetero_data(val_games_df, is_encoder_graph=False)
    test_supervision_graph = build_hetero_data(test_games_df, is_encoder_graph=False)
    
    print(f"  Encoder Graph Edges: {encoder_graph['team_year', 'wins_against', 'team_year'].num_edges + encoder_graph['team_year', 'loses_to', 'team_year'].num_edges}")
    print(f"  Train Supervision Edges: {train_supervision_graph['team_year', 'wins_against', 'team_year'].num_edges + train_supervision_graph['team_year', 'loses_to', 'team_year'].num_edges}")
    print(f"  Val Supervision Edges: {val_supervision_graph['team_year', 'wins_against', 'team_year'].num_edges + val_supervision_graph['team_year', 'loses_to', 'team_year'].num_edges}")
    print(f"  Test Supervision Edges: {test_supervision_graph['team_year', 'wins_against', 'team_year'].num_edges + test_supervision_graph['team_year', 'loses_to', 'team_year'].num_edges}")

    # 6. Initialize Model, Optimizer, Loss
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = GNNModel(
        num_nodes=num_nodes, 
        metadata=encoder_graph.metadata(), 
        embedding_dim=50,
        edge_feature_dim=EDGE_FEATURE_DIM 
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    loss_fn = torch.nn.MSELoss() 

    encoder_graph = encoder_graph.to(device)
    train_supervision_graph = train_supervision_graph.to(device)
    val_supervision_graph = val_supervision_graph.to(device)
    test_supervision_graph = test_supervision_graph.to(device)

    # 7. Run Training Loop
    print("Starting GNN training...")
    best_val_loss = float('inf')
    epochs = 500
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_step(model, encoder_graph, train_supervision_graph, optimizer, loss_fn)
        
        if np.isnan(train_loss):
            print("!!! ERROR: Loss is NaN. Stopping training. !!!")
            return {"model": None, "embeddings": None, "idx_to_node": None}
            
        if epoch % 20 == 0:
            val_loss, val_rmse, val_acc = eval_step(model, encoder_graph, val_supervision_graph, loss_fn)
            print(f"Epoch: {epoch:03d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | Val Loss: {val_loss:.4f} | Val RMSE: {val_rmse:.3f} | Val Acc: {val_acc:.3f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss

    # 8. Final Test
    print("Training complete. Running final test...")
    test_loss, test_rmse, test_acc = eval_step(model, encoder_graph, test_supervision_graph, loss_fn)
    print(f"--- GNN Test Results ---")
    print(f"  Test MSE: {test_loss:.4f} (avg across edge types)")
    print(f"  Test RMSE: {test_rmse:.4f} (avg across edge types)")
    print(f"  Test Sign Acc: {test_acc:.4f} (avg across edge types)")
    
    # 9. Show sample predictions & Get Final Embeddings
    print("\nSample Predictions (from Test Set):")
    model.eval()
    
    with torch.no_grad():
        final_z = model.encode(
            encoder_graph.x_dict, 
            encoder_graph.edge_index_dict, 
            encoder_graph.edge_attr_dict
        )
    
    print(f"{'Type':<15} | {'Team 1':<20} | {'Team 2':<20} | {'Actual':>8} | {'Predicted':>10}")
    print("-" * 79)
    
    for edge_type_name, edge_type in [('WINS_AGAINST', ('team_year', 'wins_against', 'team_year')), 
                                      ('LOSES_TO', ('team_year', 'loses_to', 'team_year'))]:
        
        if edge_type not in test_supervision_graph.edge_label_index_dict:
            continue
            
        edge_label_index = test_supervision_graph[edge_type].edge_label_index
        labels = test_supervision_graph[edge_type].edge_label
        
        num_test_samples = edge_label_index.size(1)
        if num_test_samples == 0: continue
        
        k = min(5, num_test_samples) 
        sample_indices = torch.randperm(num_test_samples)[:k]
        
        with torch.no_grad():
            pred = model.decoder(
                final_z, 
                edge_label_index[:, sample_indices]
            )
            
        for i, idx in enumerate(sample_indices.cpu().numpy()):
            team1_idx = edge_label_index[0, idx].item()
            team2_idx = edge_label_index[1, idx].item()
            
            team1 = idx_to_node.get(team1_idx, "N/A")
            team2 = idx_to_node.get(team2_idx, "N/A")
            
            actual = labels[idx].item()
            predicted = pred[i].item()
            
            print(f"{edge_type_name:<15} | {team1:<20} | {team2:<20} | {actual:>8.3f} | {predicted:>10.3f}")

    # Return results as a dictionary
    return {
        "model": model,
        "embeddings": final_z,
        "idx_to_node": idx_to_node
    }