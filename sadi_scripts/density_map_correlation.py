import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import pandas as pd


def analyze_density_map_correlation(lk_density_map, lc_density_map):
    """
    Analyze the correlation between LK and LC density maps.
    """
    # Extract cells that appear in both maps
    common_cells = set(lk_density_map.keys()).intersection(set(lc_density_map.keys()))
    print(f"Total cells in LK map: {len(lk_density_map)}")
    print(f"Total cells in LC map: {len(lc_density_map)}")
    print(f"Common cells in both maps: {len(common_cells)}")

    # Create paired data for analysis
    lk_counts = []
    lc_counts = []
    cells = []

    for cell in common_cells:
        lk_count = lk_density_map[cell]['count'] if isinstance(lk_density_map[cell], dict) else lk_density_map[cell]
        lc_count = lc_density_map[cell]['count'] if isinstance(lc_density_map[cell], dict) else lc_density_map[cell]

        lk_counts.append(lk_count)
        lc_counts.append(lc_count)
        cells.append(cell)

    # Calculate correlation
    pearson_corr, p_value_pearson = pearsonr(lk_counts, lc_counts)
    spearman_corr, p_value_spearman = spearmanr(lk_counts, lc_counts)

    print(f"Pearson correlation: {pearson_corr:.4f} (p-value: {p_value_pearson:.4e})")
    print(f"Spearman correlation: {spearman_corr:.4f} (p-value: {p_value_spearman:.4e})")

    # Create a ratio analysis
    ratios = [lk / lc if lc > 0 else float('inf') for lk, lc in zip(lk_counts, lc_counts)]
    finite_ratios = [r for r in ratios if r != float('inf')]

    print(f"\nRatio Analysis (LK/LC):")
    print(f"Min ratio: {min(finite_ratios):.2f}")
    print(f"Max ratio: {max(finite_ratios):.2f}")
    print(f"Mean ratio: {np.mean(finite_ratios):.2f}")
    print(f"Median ratio: {np.median(finite_ratios):.2f}")

    # Create dataframe for detailed analysis
    df = pd.DataFrame({
        'cell': cells,
        'lk_count': lk_counts,
        'lc_count': lc_counts,
        'ratio': ratios
    })

    # Find cells with extreme ratios
    top_lk_bias = df[df['ratio'] < float('inf')].nlargest(5, 'ratio')
    top_lc_bias = df[df['ratio'] > 0].nsmallest(5, 'ratio')

    print("\nTop 5 cells with highest LK bias (highest LK/LC ratio):")
    print(top_lk_bias[['cell', 'lk_count', 'lc_count', 'ratio']])

    print("\nTop 5 cells with highest LC bias (lowest LK/LC ratio):")
    print(top_lc_bias[['cell', 'lk_count', 'lc_count', 'ratio']])

    # Return data for plotting
    return df


def plot_density_correlation(df):
    """
    Create visualizations of the density correlation.
    """
    # Create scatterplot with log scale
    plt.figure(figsize=(10, 8))

    # Filter out infinite values for plotting
    plot_df = df[df['ratio'] < float('inf')]

    plt.scatter(plot_df['lk_count'], plot_df['lc_count'], alpha=0.5)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('LK Cell Count (log scale)')
    plt.ylabel('LC Cell Count (log scale)')
    plt.title('Correlation between LK and LC Cell Counts')

    # Add line of equality for reference
    max_val = max(max(plot_df['lk_count']), max(plot_df['lc_count']))
    min_val = min(min(plot_df['lk_count']), min(plot_df['lc_count']))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Create histogram of ratios
    plt.figure(figsize=(10, 6))

    # Use log scale for better visualization
    plt.hist(np.log10(plot_df['ratio']), bins=50)
    plt.xlabel('Log10(LK/LC Ratio)')
    plt.ylabel('Frequency')
    plt.title('Distribution of LK/LC Ratios (Log Scale)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Create heatmap of ratio by cell position
    plt.figure(figsize=(12, 10))

    # Extract cell coordinates
    cells = np.array(plot_df['cell'].tolist())
    max_n1 = max(cells[:, 0]) + 1
    max_n2 = max(cells[:, 1]) + 1

    # Create a ratio heatmap
    ratio_heatmap = np.zeros((max_n1, max_n2))
    ratio_heatmap[:] = np.nan  # Set all to NaN initially

    for i, cell in enumerate(cells):
        ratio_heatmap[cell[0], cell[1]] = np.log10(plot_df['ratio'].iloc[i])

    # Plot heatmap
    plt.imshow(ratio_heatmap, cmap='coolwarm', aspect='auto')
    plt.colorbar(label='Log10(LK/LC Ratio)')
    plt.title('Spatial Distribution of LK/LC Ratio (Log Scale)')
    plt.xlabel('LCR Probability Bin')
    plt.ylabel('LCL Probability Bin')
    plt.tight_layout()
    plt.show()


# Example usage
def main():
    # Load your cell maps
    try:
        folder_path = 'cell_maps_train_data'
        file_path = os.path.join(folder_path, 'density_cell_maps_0.0s.pkl')

        with open(file_path, 'rb') as f:
            results = pickle.load(f)

        lk_density_map = results['lk_density_map']
        lc_density_map = results['lc_density_map']

        print("Loaded cell maps successfully!")

        # Analyze correlation
        df = analyze_density_map_correlation(lk_density_map, lc_density_map)

        # Plot correlation
        plot_density_correlation(df)

    except Exception as e:
        print(f"Error loading or analyzing cell maps: {e}")


if __name__ == "__main__":
    main()