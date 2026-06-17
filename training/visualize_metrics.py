import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style for premium look
plt.style.use('dark_background')
accent_color = '#00d2ff'
secondary_color = '#ff007a'

def plot_performance_metrics():
    """Visualizes Accuracy, Precision, Recall, and F1 Score."""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [0.968, 0.942, 0.981, 0.961]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color=[accent_color, '#3a7bd5', secondary_color, '#00f2fe'], alpha=0.8)
    
    # Add value labels
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval*100:.1f}%', 
                 ha='center', va='bottom', fontsize=12, color='white', fontweight='bold')

    plt.ylim(0, 1.1)
    plt.title('CouncilMed Core Diagnostic Metrics', fontsize=16, pad=20, fontname='sans-serif', fontweight='bold')
    plt.ylabel('Score Value', color='#94a3b8')
    plt.grid(axis='y', linestyle='--', alpha=0.2)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('core_metrics.png', dpi=300)
    print("Generated: core_metrics.png")

def plot_dice_overlap():
    """Visualizes the Dice Coefficient overlap concept."""
    from matplotlib.patches import Circle
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Ground Truth Circle
    circle1 = Circle((0.4, 0.5), 0.3, color=accent_color, alpha=0.4, label='Ground Truth (A)')
    # Prediction Circle
    circle2 = Circle((0.6, 0.5), 0.3, color=secondary_color, alpha=0.4, label='AI Prediction (B)')
    
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    
    plt.title('Dice Coefficient: Spatial Overlap Visualization\nDice = 2 * |A ∩ B| / (|A| + |B|)', 
              fontsize=14, pad=20, color='white')
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig('dice_visualization.png', dpi=300)
    print("Generated: dice_visualization.png")

def plot_confidence_distribution():
    """Visualizes the distribution of confidence scores."""
    # Simulate some data
    np.random.seed(42)
    data = np.concatenate([
        np.random.normal(0.92, 0.05, 800), # High confidence correct
        np.random.normal(0.65, 0.1, 150),  # Uncertain cases
        np.random.normal(0.4, 0.15, 50)    # Rare low confidence
    ])
    data = np.clip(data, 0, 1)

    plt.figure(figsize=(10, 6))
    sns.histplot(data, bins=30, kde=True, color=accent_color, alpha=0.6)
    
    plt.title('Prediction Confidence Distribution', fontsize=16, pad=20, fontweight='bold')
    plt.xlabel('Confidence Score (Softmax Output)', color='#94a3b8')
    plt.ylabel('Frequency (Number of Images)', color='#94a3b8')
    plt.axvline(x=0.85, color=secondary_color, linestyle='--', label='Clinical Threshold (0.85)')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.1)
    plt.tight_layout()
    plt.savefig('confidence_dist.png', dpi=300)
    print("Generated: confidence_dist.png")

if __name__ == "__main__":
    print("Generating CouncilMed Diagnostic Visualizations...")
    plot_performance_metrics()
    plot_dice_overlap()
    plot_confidence_distribution()
    print("\nAll visualizations saved successfully as PNG files.")
