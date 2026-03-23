import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.titlesize'] = 16


def plot_threshold_metrics_improved(threshold_results, horizon_label, output_prefix='sadi'):
    """
    Create a publication-quality plot showing how different metrics
    change with reliability thresholds - improved for B&W readability
    with different markers and line styles
    """
    # Set up figure
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

    # Define colors, markers, and line styles for B&W readability
    # Using grayscale colors and distinct markers/line styles
    category_styles = {
        'Overall': {
            'color': '#000000',  # Black
            'marker': 'o',  # Circle
            'linestyle': '-',  # Solid line
            'markersize': 6,
            'linewidth': 3  # Thicker for emphasis
        },
        'TP': {
            'color': '#2F2F2F',  # Dark gray
            'marker': 's',  # Square
            'linestyle': '-',  # Solid line
            'markersize': 5,
            'linewidth': 2
        },
        'FP': {
            'color': '#000000',  # Black
            'marker': '^',  # Triangle up
            'linestyle': '--',  # Dashed line
            'markersize': 6,
            'linewidth': 2
        },
        'FN': {
            'color': '#4F4F4F',  # Medium gray
            'marker': 'D',  # Diamond
            'linestyle': '-.',  # Dash-dot line
            'markersize': 5,
            'linewidth': 2
        },
        'TN': {
            'color': '#6F6F6F',  # Light gray
            'marker': 'v',  # Triangle down
            'linestyle': ':',  # Dotted line
            'markersize': 6,
            'linewidth': 2
        },
        'Other': {
            'color': '#808080',  # Gray
            'marker': 'x',  # X mark
            'linestyle': '-',  # Solid line
            'markersize': 6,
            'linewidth': 2
        }
    }

    # Plot Overall retention rate first (most important)
    overall_style = category_styles['Overall']
    ax1.plot(threshold_results['threshold'], threshold_results['retention_rate'],
             marker=overall_style['marker'],
             linestyle=overall_style['linestyle'],
             color=overall_style['color'],
             linewidth=overall_style['linewidth'],
             markersize=overall_style['markersize'],
             label='Overall',
             markerfacecolor='white',  # White fill for better contrast
             markeredgecolor=overall_style['color'],
             markeredgewidth=2)

    # Add retention by prediction category with specific styles
    for category in ['TP', 'FP', 'FN', 'TN', 'Other']:
        col = f'retention_{category}'
        if col in threshold_results.columns and category in category_styles:
            style = category_styles[category]

            # Special handling for different categories to improve readability
            if category == 'FP':
                # Use hollow markers for FP to distinguish from TP
                markerfacecolor = 'white'
                markeredgewidth = 2
            elif category == 'FN':
                # Use half-filled markers for FN
                markerfacecolor = style['color']
                markeredgewidth = 1
            else:
                markerfacecolor = style['color']
                markeredgewidth = 1

            ax1.plot(threshold_results['threshold'], threshold_results[col],
                     marker=style['marker'],
                     linestyle=style['linestyle'],
                     color=style['color'],
                     linewidth=style['linewidth'],
                     markersize=style['markersize'],
                     label=category,
                     markerfacecolor=markerfacecolor,
                     markeredgecolor=style['color'],
                     markeredgewidth=markeredgewidth)

    # Customize axes
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel('Reliability Threshold')
    ax1.set_ylabel('Retention Rate')
    ax1.set_title(f'Retention Rate vs. Reliability Threshold - Prediction Horizon: {horizon_label}')

    # Improve legend for better readability - positioned at top right
    legend = ax1.legend(loc='upper right',
                        frameon=True,
                        fancybox=True,
                        shadow=True,
                        framealpha=0.9,
                        edgecolor='black')
    legend.get_frame().set_facecolor('white')

    # Improve grid
    ax1.grid(True, alpha=0.3, color='gray', linestyle='-', linewidth=0.5)
    ax1.set_axisbelow(True)

    # Add minor ticks for better readability
    ax1.minorticks_on()
    ax1.grid(True, which='minor', alpha=0.1, color='gray', linestyle='-', linewidth=0.25)

    plt.tight_layout()

    # Save the figure with a different name to avoid overwriting
    filename = f'{output_prefix}_retention_analysis_improved_{horizon_label}.png'
    fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved improved retention analysis figure to {filename}")

    return fig


def plot_threshold_metrics_color_version(threshold_results, horizon_label, output_prefix='sadi'):
    """
    Alternative version with better colors but still B&W readable
    """
    # Set up figure
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

    # Define high-contrast colors that work in both color and B&W
    category_styles = {
        'Overall': {
            'color': '#000000',  # Black
            'marker': 'o',  # Circle
            'linestyle': '-',  # Solid line
            'markersize': 7,
            'linewidth': 3
        },
        'TP': {
            'color': '#006400',  # Dark green (dark in B&W)
            'marker': 's',  # Square
            'linestyle': '-',  # Solid line
            'markersize': 6,
            'linewidth': 2
        },
        'FP': {
            'color': '#8B0000',  # Dark red (dark in B&W)
            'marker': '^',  # Triangle up
            'linestyle': '--',  # Dashed line
            'markersize': 7,
            'linewidth': 2
        },
        'FN': {
            'color': '#000080',  # Navy blue (medium in B&W)
            'marker': 'D',  # Diamond
            'linestyle': '-.',  # Dash-dot line
            'markersize': 5,
            'linewidth': 2
        },
        'TN': {
            'color': '#4B0082',  # Indigo (medium in B&W)
            'marker': 'v',  # Triangle down
            'linestyle': ':',  # Dotted line
            'markersize': 7,
            'linewidth': 2
        },
        'Other': {
            'color': '#808080',  # Gray
            'marker': 'x',  # X mark
            'linestyle': '-',  # Solid line
            'markersize': 7,
            'linewidth': 2
        }
    }

    # Plot Overall retention rate first
    overall_style = category_styles['Overall']
    ax1.plot(threshold_results['threshold'], threshold_results['retention_rate'],
             marker=overall_style['marker'],
             linestyle=overall_style['linestyle'],
             color=overall_style['color'],
             linewidth=overall_style['linewidth'],
             markersize=overall_style['markersize'],
             label='Overall',
             markerfacecolor='white',
             markeredgecolor=overall_style['color'],
             markeredgewidth=2)

    # Add retention by prediction category
    for category in ['TP', 'FP', 'FN', 'TN', 'Other']:
        col = f'retention_{category}'
        if col in threshold_results.columns and category in category_styles:
            style = category_styles[category]
            ax1.plot(threshold_results['threshold'], threshold_results[col],
                     marker=style['marker'],
                     linestyle=style['linestyle'],
                     color=style['color'],
                     linewidth=style['linewidth'],
                     markersize=style['markersize'],
                     label=category)

    # Customize the plot
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel('Reliability Threshold')
    ax1.set_ylabel('Retention Rate')
    ax1.set_title(f'Retention Rate vs. Reliability Threshold - AD4CHE')

    # Improve legend
    legend = ax1.legend(loc='upper right',
                        frameon=True,
                        fancybox=True,
                        shadow=True,
                        framealpha=0.9)

    # Improve grid
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save with different filename
    filename = f'{output_prefix}_retention_analysis_color_improved_{horizon_label}.png'
    fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved color improved retention analysis figure to {filename}")

    return fig


# Example usage function that you can call from your main script
def create_improved_plots(all_threshold_results, output_prefix='sadi'):
    """
    Create improved plots for all horizons
    """
    print("Creating improved retention plots...")

    for horizon_label, threshold_results in all_threshold_results.items():
        print(f"Creating improved plots for horizon: {horizon_label}")

        # Create both versions
        fig1 = plot_threshold_metrics_improved(threshold_results, horizon_label, output_prefix)
        fig2 = plot_threshold_metrics_color_version(threshold_results, horizon_label, output_prefix)

        plt.close(fig1)  # Close to free memory
        plt.close(fig2)

    print("Improved plots created successfully!")


# Test function to demonstrate the differences
def demo_marker_styles():
    """
    Create a demo plot showing all the different marker styles and line types
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    x = np.linspace(0, 1, 10)

    styles = {
        'Overall': {'marker': 'o', 'linestyle': '-', 'color': '#000000', 'linewidth': 3},
        'TP': {'marker': 's', 'linestyle': '-', 'color': '#2F2F2F', 'linewidth': 2},
        'FP': {'marker': '^', 'linestyle': '--', 'color': '#000000', 'linewidth': 2},
        'FN': {'marker': 'D', 'linestyle': '-.', 'color': '#4F4F4F', 'linewidth': 2},
        'TN': {'marker': 'v', 'linestyle': ':', 'color': '#6F6F6F', 'linewidth': 2},
        'Other': {'marker': 'x', 'linestyle': '-', 'color': '#808080', 'linewidth': 2}
    }

    for i, (label, style) in enumerate(styles.items()):
        y = 0.9 - 0.1 * np.sin(x * np.pi * 2 + i)
        ax.plot(x, y,
                marker=style['marker'],
                linestyle=style['linestyle'],
                color=style['color'],
                linewidth=style['linewidth'],
                markersize=8,
                label=label,
                markerfacecolor='white' if label in ['Overall', 'FP'] else style['color'],
                markeredgecolor=style['color'],
                markeredgewidth=2 if label in ['Overall', 'FP'] else 1)

    ax.set_title('Demo: Marker Styles and Line Types for B&W Readability')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig('demo_marker_styles.png', dpi=300, bbox_inches='tight')
    print("Saved demo plot to demo_marker_styles.png")

    return fig


if __name__ == "__main__":
    # Create demo plot
    demo_marker_styles()
    print("Demo plot created. You can now use the improved functions in your main script.")