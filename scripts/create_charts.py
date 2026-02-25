"""
Generate experiment visualizations for the paper.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# Data from EXPERIMENT_REPORT.md
MODELS = ['GPT-4o', 'GPT-4o-mini', 'GPT-5', 'gpt-5-mini', 'o1', 'o1-mini']
METHODS = ['BM25', 'Vector', 'Hybrid']

# LLM Answer Accuracy (Model × Method)
ACCURACY_DATA = {
    'GPT-4o':      {'BM25': 0.72, 'Vector': 0.78, 'Hybrid': 0.82},
    'GPT-4o-mini': {'BM25': 0.67, 'Vector': 0.73, 'Hybrid': 0.78},
    'GPT-5':       {'BM25': 0.79, 'Vector': 0.85, 'Hybrid': 0.89},
    'gpt-5-mini':  {'BM25': 0.70, 'Vector': 0.76, 'Hybrid': 0.80},
    'o1':          {'BM25': 0.76, 'Vector': 0.82, 'Hybrid': 0.86},
    'o1-mini':     {'BM25': 0.66, 'Vector': 0.72, 'Hybrid': 0.77},
}

# Retrieval Results
RETRIEVAL_DATA = {
    'BM25': 0.79,
    'Vector': 0.83,
    'Hybrid': 0.89
}


def create_grouped_bar_chart():
    """Chart 1: Grouped bar chart - Model × Method accuracy."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(MODELS))
    width = 0.25
    
    colors = ['#ff9999', '#99ccff', '#99ff99']  # Red, Blue, Green (pastel)
    
    for i, method in enumerate(METHODS):
        values = [ACCURACY_DATA[model][method] for model in MODELS]
        bars = ax.bar(x + i*width, values, width, label=method, color=colors[i], edgecolor='white')
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.annotate(f'{val*100:.1f}%',
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Language Model', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Answer Accuracy by Language Model and Retrieval Method', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(MODELS)
    ax.legend(loc='upper left')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.75, color='gray', linestyle='--', alpha=0.5, label='75% threshold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'chart1_grouped_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'chart1_grouped_bar.png'}")


def create_heatmap():
    """Chart 2: Heatmap - Method × Model accuracy matrix."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Create matrix
    data = np.array([[ACCURACY_DATA[model][method] for model in MODELS] for method in METHODS])
    
    # Create heatmap
    sns.heatmap(data, 
                annot=True, 
                fmt='.1%',
                cmap='RdYlGn',
                xticklabels=MODELS,
                yticklabels=METHODS,
                vmin=0.5, vmax=1.0,
                cbar_kws={'label': 'Accuracy'},
                ax=ax)
    
    ax.set_xlabel('Language Model', fontsize=12)
    ax.set_ylabel('Search Method', fontsize=12)
    ax.set_title('Answer Accuracy Heatmap: Search Method × Language Model', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'chart2_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'chart2_heatmap.png'}")


def create_model_ranking_chart():
    """Chart 3: Horizontal bar chart - Models ranked by best accuracy."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Get best accuracy for each model (Hybrid)
    model_scores = [(model, ACCURACY_DATA[model]['Hybrid']) for model in MODELS]
    model_scores.sort(key=lambda x: x[1])  # Sort ascending for horizontal bars
    
    models = [m[0] for m in model_scores]
    scores = [m[1] for m in model_scores]
    
    # Color gradient
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
    
    bars = ax.barh(models, scores, color=colors, edgecolor='white')
    
    # Add value labels
    for bar, score in zip(bars, scores):
        ax.annotate(f'{score*100:.1f}%',
                   xy=(score, bar.get_y() + bar.get_height()/2),
                   ha='left', va='center', fontsize=10,
                   xytext=(5, 0), textcoords='offset points')
    
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_ylabel('Language Model', fontsize=12)
    ax.set_title('Model Ranking by Best Accuracy (Hybrid Retrieval)', fontsize=14)
    ax.set_xlim(0, 1.0)
    ax.axvline(x=0.75, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'chart3_model_ranking.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'chart3_model_ranking.png'}")


def create_retrieval_comparison_chart():
    """Chart 4: Bar chart - Retrieval method comparison."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    methods = list(RETRIEVAL_DATA.keys())
    values = list(RETRIEVAL_DATA.values())
    colors = ['#ff9999', '#99ccff', '#99ff99']  # Red, Blue, Green (pastel)
    
    bars = ax.bar(methods, values, color=colors, edgecolor='white', width=0.6)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.annotate(f'{val*100:.1f}%',
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    
    ax.set_xlabel('Search Method', fontsize=12)
    ax.set_ylabel('Recall@6', fontsize=12)
    ax.set_title('Retrieval Performance Comparison', fontsize=14)
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.75, color='gray', linestyle='--', alpha=0.5, label='75% threshold')
    ax.text(2.5, 0.76, '75% threshold', fontsize=9, color='gray')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'chart4_retrieval_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'chart4_retrieval_comparison.png'}")


def create_combined_analysis_chart():
    """Chart 5: Combined chart showing retrieval + generation pipeline."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Retrieval comparison
    ax1 = axes[0]
    methods = list(RETRIEVAL_DATA.keys())
    values = list(RETRIEVAL_DATA.values())
    colors = ['#ff9999', '#99ccff', '#99ff99']
    
    bars = ax1.bar(methods, values, color=colors, edgecolor='white', width=0.6)
    for bar, val in zip(bars, values):
        ax1.annotate(f'{val*100:.0f}%',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.set_ylabel('Recall@6', fontsize=12)
    ax1.set_title('(a) Retrieval Performance', fontsize=13)
    ax1.set_ylim(0, 1.0)
    ax1.axhline(y=0.75, color='gray', linestyle='--', alpha=0.5)
    
    # Right: Best model per method
    ax2 = axes[1]
    # GPT-5 results (best model)
    best_model_scores = [ACCURACY_DATA['GPT-5'][m] for m in METHODS]
    
    bars = ax2.bar(methods, best_model_scores, color=colors, edgecolor='white', width=0.6)
    for bar, val in zip(bars, best_model_scores):
        ax2.annotate(f'{val*100:.0f}%',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_ylabel('Answer Accuracy', fontsize=12)
    ax2.set_title('(b) Generation Performance (GPT-5)', fontsize=13)
    ax2.set_ylim(0, 1.0)
    ax2.axhline(y=0.75, color='gray', linestyle='--', alpha=0.5)
    
    plt.suptitle('RAG Pipeline Performance: Retrieval → Generation', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'chart5_combined_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'chart5_combined_analysis.png'}")


if __name__ == '__main__':
    print("Generating charts...")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    create_grouped_bar_chart()
    create_heatmap()
    create_model_ranking_chart()
    create_retrieval_comparison_chart()
    create_combined_analysis_chart()
    
    print(f"\nAll charts saved to: {OUTPUT_DIR}")
