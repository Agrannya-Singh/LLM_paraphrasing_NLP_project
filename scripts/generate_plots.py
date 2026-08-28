import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure paper directory exists
os.makedirs('paper', exist_ok=True)

def plot_fper():
    labels = ['SBERT', 'SimCSE', 'BMX Hybrid', 'ColBERT', 'Longformer']
    tier1 = [25.0, 0.0, 90.0, 70.0, 25.0]
    tier2 = [30.0, 0.0, 90.0, 70.0, 30.0]
    corpus = [42.0, 0.0, 85.0, 70.0, 40.0]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width, tier1, width, label='Tier 1 (Static)', color='#4c72b0')
    rects2 = ax.bar(x, tier2, width, label='Tier 2 (Saliency)', color='#dd8452')
    rects3 = ax.bar(x + width, corpus, width, label='Corpus Adversarial', color='#55a868')

    ax.set_ylabel('Fidelity-Preserved Evasion Rate (%)', fontsize=12)
    ax.set_title('Evasion Success Rates by Defense Architecture and Attack Tier', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    fig.tight_layout()
    plt.savefig('paper/defense_resistance.png', dpi=300)
    plt.close()

def plot_mqc():
    labels = ['SBERT', 'SimCSE', 'BMX Hybrid', 'ColBERT', 'Longformer']
    mqc = [32.7, 50.0, 1.0, 8.9, 33.2]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#c44e52', '#4c72b0', '#ccb974', '#64b5cd', '#8172b2']
    bars = ax.bar(labels, mqc, color=colors)

    ax.set_ylabel('Mean Query Complexity (MQC)', fontsize=12)
    ax.set_title('Average Queries Consumed under Tier 2 Attack (Budget = 50)', fontsize=14)
    ax.axhline(y=50, color='r', linestyle='--', label='Query Budget Limit')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom')

    ax.legend()
    fig.tight_layout()
    plt.savefig('paper/query_complexity.png', dpi=300)
    plt.close()

plot_fper()
plot_mqc()
print("Plots generated successfully in paper/ directory.")
