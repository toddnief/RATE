import json
import random
import os
import re
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import matplotlib as mpl
import numpy as np
import seaborn as sns
from collections import defaultdict
from pathlib import Path
from rate.utils import load_dataset_from_json, write_to_json
from rate.treatment_effects import calculate_treatment_effects
import pandas as pd

def sample_rewrites_tabular(file_paths, model_key, num_samples=10, max_text_length=100):
    def load_json_lines(file_path):
        with open(file_path, 'r') as file:
            return [json.loads(line) for line in file]

    def find_model_key(item):
        return next((key for key in item.keys() if model_key in key), None)

    def truncate_text(text, max_length):
        return text[:max_length] + '...' if len(text) > max_length else text

    def escape_latex(text):
        latex_special_chars = {'&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\^{}', '\\': r'\textbackslash{}'}
        return ''.join(latex_special_chars.get(c, c) for c in text)

    def format_reward(reward):
        return f"{reward:.5f}" if isinstance(reward, (int, float)) else reward

    def get_descriptive_title(file_name):
        # Extract database and concept from file name
        match = re.match(r'(\w+)_(\w+)_', file_name)
        if match:
            database, concept = match.groups()
            database = database.upper() if database.lower() == 'eli5' else database.capitalize()
            return f"{database}, {concept.capitalize()}"
        return file_name  # Fallback to filename if pattern doesn't match

    def create_table(data, file_name):
        title = get_descriptive_title(file_name)
        latex_table = f"\\subsection*{{{title}}}\n"
        latex_table += r"\begin{tabular}{|p{0.22\textwidth}|p{0.22\textwidth}|p{0.22\textwidth}|p{0.22\textwidth}|}\hline" + "\n"
        latex_table += r"Original & Rewrite & Rewrite of Rewrite & Reward \\ \hline" + "\n"

        for item in data:
            w_original = item.get('w_original', False)
            original = escape_latex(truncate_text(item['completions'].get('original', 'N/A'), max_text_length))
            rewrite = escape_latex(truncate_text(item['completions'].get('rewrite', 'N/A'), max_text_length))
            rewrite_of_rewrite = escape_latex(truncate_text(item['completions'].get('rewritten rewrite', 'N/A'), max_text_length))
            
            model_key = find_model_key(item)
            if model_key:
                reward_original = format_reward(item[model_key].get('original', 'N/A'))
                reward_rewrite = format_reward(item[model_key].get('rewrite', 'N/A'))
                reward_rewrite_of_rewrite = format_reward(item[model_key].get('rewritten rewrite', 'N/A'))
            else:
                reward_original = reward_rewrite = reward_rewrite_of_rewrite = 'N/A'

            latex_table += f"{original} (W = {1 if w_original else 0}) & "
            latex_table += f"{rewrite} (W = {0 if w_original else 1}) & "
            latex_table += f"{rewrite_of_rewrite} & "
            latex_table += f"({reward_original}, {reward_rewrite}, {reward_rewrite_of_rewrite}) \\\\ \\hline\n"

        latex_table += r"\end{tabular}" + "\n\n"
        return latex_table

    all_tables = []
    for file_path in file_paths:
        data = load_json_lines(file_path)
        samples = random.sample(data, num_samples)
        file_name = os.path.basename(file_path)
        table = create_table(samples, file_name)
        all_tables.append(table)

    return '\n'.join(all_tables)

def sample_rewrites(file_paths, model_key):
    def load_json_lines(file_path):
        with open(file_path, 'r') as file:
            return [json.loads(line) for line in file]

    def find_model_key(item):
        return next((key for key in item.keys() if model_key in key), None)

    def escape_latex(text):
        latex_special_chars = {'&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_', 
                             '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', 
                             '^': r'\^{}', '\\': r'\textbackslash{}'}
        return ''.join(latex_special_chars.get(c, c) for c in text)

    def format_reward(reward):
        return f"{reward:.5f}" if isinstance(reward, (int, float)) else reward

    def get_descriptive_title(file_name):
        match = re.match(r'(\w+)_(\w+)_', file_name)
        if match:
            database, concept = match.groups()
            database = database.upper() if database.lower() == 'eli5' else database.capitalize()
            return f"{database}, {concept.capitalize()}"
        return file_name

    def create_sample(item, file_name):
        title = get_descriptive_title(file_name)
        latex_output = f"\\subsection*{{{title}}}\n\n"

        if 'reward_question' in item:
            latex_output += "\\textbf{Reward Question}:\n"
            latex_output += escape_latex(item['reward_question']) + "\n\n"
        
        w_original = item.get('w_original', False)
        original = escape_latex(item['completions'].get('original', 'N/A'))
        rewrite = escape_latex(item['completions'].get('rewrite', 'N/A'))
        rewrite_of_rewrite = escape_latex(item['completions'].get('rewritten rewrite', 'N/A'))
        
        model_key = find_model_key(item)
        if model_key:
            reward_original = format_reward(item[model_key].get('original', 'N/A'))
            reward_rewrite = format_reward(item[model_key].get('rewrite', 'N/A'))
            reward_rewrite_of_rewrite = format_reward(item[model_key].get('rewritten rewrite', 'N/A'))
        else:
            reward_original = reward_rewrite = reward_rewrite_of_rewrite = 'N/A'

        latex_output += "\\textbf{Original} (W = " + str(1 if w_original else 0) + "):\n"
        latex_output += original + "\n\n"
        latex_output += "\\textbf{Rewrite} (W = " + str(0 if w_original else 1) + "):\n"
        latex_output += rewrite + "\n\n"
        latex_output += "\\textbf{Rewrite of Rewrite}:\n"
        latex_output += rewrite_of_rewrite + "\n\n"
        latex_output += "\\textbf{Rewards} (Original, Rewrite, Rewrite of Rewrite):\n"
        latex_output += f"({reward_original}, {reward_rewrite}, {reward_rewrite_of_rewrite})\n\n"

        return latex_output

    all_samples = []
    for file_path in file_paths:
        data = load_json_lines(file_path)
        sample = random.choice(data)
        file_name = os.path.basename(file_path)
        output = create_sample(sample, file_name)
        all_samples.append(output)

    return '\n'.join(all_samples)

def setup_plots():
    """Helper function to set up consistent plot styling"""
    sns.set_theme(style="whitegrid", font="serif")
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['text.usetex'] = True
    # Add bolder text support in LaTeX
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{times}\usepackage{bm}'

def plot_scores(template_data, SCORED_DIR):
    setup_plots()
    fig = plt.figure(figsize=(8, 4), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    
    colors = ['#1f77b4', '#ff7f0e']  # Blue, Orange
    
    dataset_filename = Path(template_data["dataset_filename"])
    scores = load_dataset_from_json(SCORED_DIR / "complete" / dataset_filename)
    
    original_scores = []
    rewrite_scores = []
    
    for key, data_point in scores.items():
        original_scores.append(data_point[template_data["reward_key"]].get("original", 0) / 
                              data_point[template_data["reward_key"]].get("reward_std", 1))
        rewrite_scores.append(data_point[template_data["reward_key"]].get("rewritten rewrite", 0) / 
                             data_point[template_data["reward_key"]].get("reward_std", 1))

    # KDE plot
    sns.kdeplot(data=original_scores, color=colors[0], fill=True, alpha=0.5, linewidth=2, ax=ax)
    sns.kdeplot(data=rewrite_scores, color=colors[1], fill=True, alpha=0.5, linewidth=2, ax=ax)

    # Add dashed lines for means
    original_mean = np.mean(original_scores) if original_scores else 0
    rewrite_mean = np.mean(rewrite_scores) if rewrite_scores else 0
    ax.axvline(original_mean, color=colors[0], linestyle='--', linewidth=2)
    ax.axvline(rewrite_mean, color=colors[1], linestyle='--', linewidth=2)

    # Labels and formatting
    ax.set_xlabel(r'\textbf{Reward}', fontsize=18)
    ax.set_ylabel(r'\textbf{Density}', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Custom legend
    legend_lines = [
        Line2D([0], [0], color=colors[0], linewidth=2, label=r"$\textbf{Original}$"),
        Line2D([0], [0], color=colors[1], linewidth=2, label=r"$\textbf{Rewrite}^2$"),
    ]
    ax.legend(handles=legend_lines, fontsize=14, loc='upper left', frameon=True)

    plt.tight_layout()
    plt.show()

def rewrite_bias(effects_data):
    setup_plots()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=300)
    colors = ['#FF7F0E', '#FFA54F', '#1F77B4', '#6CA6CD', '#D62728', '#FF6A6A']

    for idx, ax in enumerate(axes):
        data = pd.DataFrame({
            'Effect': [
                r'$\widehat{\textbf{ATE}}\ (\textbf{Rewrite}^2)$', 
                r'$\widehat{\textbf{ATE}}\ (\textbf{Rewrite})$',
                r'$\widehat{\textbf{ATT}}\ (\textbf{Rewrite}^2)$', 
                r'$\widehat{\textbf{ATT}}\ (\textbf{Rewrite})$',
                r'$\widehat{\textbf{ATU}}\ (\textbf{Rewrite}^2)$', 
                r'$\widehat{\textbf{ATU}}\ (\textbf{Rewrite})$'
            ],
            'Value': [effects_data[idx][f'{effect}'] / effects_data[idx]['reward_std'] 
         for effect in ['ATE_rewritten_rewrite', 'ATE_single_rewrite', 
                        'ATT_rewritten_rewrite', 'ATT_single_rewrite', 
                        'ATU_rewritten_rewrite', 'ATU_single_rewrite']],
            'Stderr': [effects_data[idx][f'{effect}_stderr'] / effects_data[idx]['reward_std'] 
                    for effect in ['ATE_rewritten_rewrite', 'ATE_single_rewrite', 
                                    'ATT_rewritten_rewrite', 'ATT_single_rewrite', 
                                    'ATU_rewritten_rewrite', 'ATU_single_rewrite']]

        })

        data['CI_lower'] = data['Value'] - 1.96 * data['Stderr']
        data['CI_upper'] = data['Value'] + 1.96 * data['Stderr']

        sns.barplot(x='Effect', y='Value', data=data, ax=ax, 
                   palette=dict(zip(data['Effect'], colors)),
                   alpha=0.8, edgecolor='black')

        ax.errorbar(x=data.index, y=data['Value'], yerr=1.96 * data['Stderr'], 
                   fmt='none', color='black', capsize=5, capthick=2, elinewidth=2)
        if idx == 0:
            ax.set_ylabel(r'$\textbf{Effect Size}$', fontsize=18)
        else:
            ax.set_ylabel('')
        ax.set_xlabel('')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.tick_params(axis='both', which='major', labelsize=18)
        # ax.set_title(r'\textbf{' + effects_data[idx]['score'] + '}', fontsize=24, pad=22)
        
        # Remove legend if it's not the first subplot
        if idx > 0 and ax.get_legend() != None:
            ax.get_legend().remove()

    plt.tight_layout()
    plt.show()


def naive_vs_RATE(all_data, all_templates, reward_models, normalize=True):
    """
    Plot naive effect vs. RATE using mathtext for bold text (no external LaTeX).
    Shows y-axis grid lines, uses tight_layout, and only has y-tick labels on the leftmost subplot.
    """

    fig, axes = plt.subplots(
        1, len(reward_models),
        figsize=(22, 6),
        dpi=300,
        sharey=True  # <-- same y-axis scale
    )

    # If there's only one model, make axes a list for consistency
    if len(reward_models) == 1:
        axes = [axes]

    for idx, model in enumerate(reward_models):
        ax = axes[idx]

        # Filter data for this model
        model_data = [
            d for d, t in zip(all_data, all_templates)
            if t['score'] == model
        ]
        model_templates = [
            t for t in all_templates
            if t['score'] == model
        ]
        
        grouped_data = defaultdict(lambda: defaultdict(list))
        for data, template in zip(model_data, model_templates):
            concept = template['concept']
            for effect_type in ['naive_effect', 'ATE_rewritten_rewrite']:
                value = data[effect_type]
                error = data.get(f'{effect_type}_stderr', 0.0)
                
                if normalize:
                    norm_factor = data.get('reward_std', 1)
                    value /= norm_factor
                    error /= norm_factor
                
                grouped_data[concept][effect_type].append(value)
                grouped_data[concept][f'{effect_type}_error'].append(error)

        valid_concepts = []

        # Compute mean and mean-squared errors
        for concept in grouped_data:
            if all(len(grouped_data[concept][k]) > 0 for k in grouped_data[concept]):
                valid_concepts.append(concept)
                for key in grouped_data[concept]:
                    if key.endswith('_error'):
                        grouped_data[concept][key] = np.sqrt(
                            np.mean(np.array(grouped_data[concept][key])**2)
                        )
                    else:
                        grouped_data[concept][key] = np.mean(grouped_data[concept][key])

        if not valid_concepts:
            print(f"Error: No valid concepts found for model '{model}'. Skipping.")
            continue

        # Print out each quantity
        # Sort valid concepts alphabetically
        valid_concepts = sorted(valid_concepts)
        print(f"\n=== Model: {model} ===")
        for concept in valid_concepts:
            naive_val = grouped_data[concept]['naive_effect']
            naive_err = grouped_data[concept]['naive_effect_error']
            rate_val = grouped_data[concept]['ATE_rewritten_rewrite']
            rate_err = grouped_data[concept]['ATE_rewritten_rewrite_error']
            print(
                f"Concept: {concept:20} | "
                f"Naive: {naive_val:.3f} ± {naive_err:.3f} | "
                f"RATE: {rate_val:.3f} ± {rate_err:.3f}"
            )

        # Prepare data for plotting
        plot_data = []
        for concept in valid_concepts:
            for effect_type in ['naive_effect', 'ATE_rewritten_rewrite']:
                if effect_type == 'naive_effect':
                    label = r'$\mathbf{Naive}$'
                else:
                    label = r'$\widehat{\mathbf{ATE}}_{\mathbf{RATE}}$'
                plot_data.append({
                    'Concept': concept,
                    'Effect Type': label,
                    'Effect Size': grouped_data[concept][effect_type],
                    'Error': grouped_data[concept][f'{effect_type}_error']
                })

        df = pd.DataFrame(plot_data)
        

        colors = ['#1f77b4', '#ff7f0e']  # Blue, Orange
        # Bar plot (custom error bars)
        bar_width = 0.4
        for i, effect_type in enumerate(df['Effect Type'].unique()):
            effect_data = df[df['Effect Type'] == effect_type]
            
            # Set properties to emphasize RATE and de-emphasize Naive
            if effect_type == r'$\mathbf{Naive}$':
                alpha = 0.25
                color = colors[0]
                hatch = '///'      # Hatching for the baseline method
            else:
                alpha = 0.9        # Higher alpha for RATE to make it stand out
                color = colors[1]
                hatch = ''         # Clean, solid fill for your method
            
            # Create bars manually
            x = np.arange(len(valid_concepts))
            x_pos = x + (i - 0.5) * bar_width
            
            # Create bars with hatching
            bars = ax.bar(x_pos, 
                        effect_data['Effect Size'],
                        width=bar_width,
                        label=effect_type,
                        color=color,
                        alpha=alpha,
                        capsize=0.1,
                        hatch=hatch)
            
            # Add error bars
            y_errs = effect_data['Error'] * 1.96  # ~95% CI
            ax.errorbar(x_pos, 
                    effect_data['Effect Size'],
                    yerr=y_errs,
                    fmt='none',
                    c='black',
                    capsize=5)

        # Add a light y-axis grid
        ax.grid(axis='y', alpha=0.3)
        
        # Set title as model name, bold
        ax.set_title(f"{model}", fontsize=20, pad=20, fontweight='bold')

        # Bold x tick labels (mathtext)
        ax.set_xticks(np.arange(len(valid_concepts)))
        ax.set_xticklabels(
            [rf'$\mathbf{{{concept}}}$' for concept in valid_concepts],
            rotation=45, ha='right', fontsize=20
        )

        # If this is the leftmost subplot, show y-label and ticks
        if idx == 0:
            ax.set_ylabel(r'$\mathbf{Effect\ Size}$', fontsize=20)
            ax.tick_params(axis='y', labelleft=True)  # show y tick labels
        else:
            # Remove y-label and y tick labels for other subplots
            ax.set_ylabel('')
            ax.tick_params(axis='y', labelleft=False)

        ax.set_xlabel('')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=20)

        # Legend
        if idx == 0:
            ax.legend(title='', loc='upper left', fontsize=14, frameon=True)
        else:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

    # Use tight_layout to prevent label/title overlap
    plt.tight_layout()
    plt.show()

def synthetic_plot(data_list, effects_templates, target_concept, spurious_concept, xlab: str):
    setup_plots()
    colors = sns.color_palette("deep", 3)
    alphas = [0.3, 0.9, 0.7]
    hatches = ['///', '', 'xx']

    def prepare_plot_data(data_list, effects_templates):
        plot_data = []
        for data, template in zip(data_list, effects_templates):
            correlation = int(template['dataset_filename'].split('_')[-1].split('.')[0]) / 10
            for effect_type in ['naive_effect', 'ATE_rewritten_rewrite', 'ATE_single_rewrite']:
                plot_data.append({
                    'Correlation': correlation,
                    'Effect Type': effect_type,
                    'Effect Size': data[effect_type] / data.get('reward_std', 1),
                    'Lower CI': (data[effect_type] - data.get(f'{effect_type}_stderr', 0) * 1.96) / data.get('reward_std', 1),
                    'Upper CI': (data[effect_type] + data.get(f'{effect_type}_stderr', 0) * 1.96) / data.get('reward_std', 1)
                })
        return pd.DataFrame(plot_data)

    df = prepare_plot_data(data_list, effects_templates)
    
    # Create single plot figure
    fig = plt.figure(figsize=(7, 5), dpi=300)
    ax = fig.add_subplot(1, 1, 1)

    effect_labels = {
        'naive_effect': r'$\textbf{Naive}$',
        'ATE_rewritten_rewrite': r'$\widehat{\textbf{ATE}}\ (\textbf{Rewrite}^2)$',
        'ATE_single_rewrite': r'$\widehat{\textbf{ATE}}\ (\textbf{Rewrite})$'
    }
    
    for i, (effect_type, label) in enumerate(effect_labels.items()):
        effect_data = df[df['Effect Type'] == effect_type]
        ax.plot(effect_data['Correlation'], effect_data['Effect Size'],
                label=label, color=colors[i], linewidth=2.5, alpha=alphas[i])
        
        ax.fill_between(effect_data['Correlation'], 
                        effect_data['Lower CI'],
                        effect_data['Upper CI'], 
                        color=colors[i],
                        alpha=0.2,
                        hatch=hatches[i])

    # Labels and formatting
    ax.set_xlabel(xlab, fontsize=16)
    ax.set_ylabel(r'$\textbf{Effect Size}$', fontsize=16)
    ax.legend(loc='upper left', fontsize=12, frameon=True)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlim(0, 1)

    # Slope calculations
    for effect_type in ['ATE_rewritten_rewrite', 'ATE_single_rewrite']:
        effect_data = df[df['Effect Type'] == effect_type]
        x = effect_data['Correlation']
        y = effect_data['Effect Size']
        if len(x) > 1:  # Ensure we have enough points for linear regression
            slope = np.polyfit(x, y, 1)[0]
            print(f"Slope of {effect_labels[effect_type]}: {slope:.4f}")

    plt.tight_layout()
    plt.show()

# def att_atu(effects_data, reward_std, model_name):
#     setup_plots()

#     data = pd.DataFrame({
#         'Effect': ['ATT_rewritten_rewrite', 'ATU_rewritten_rewrite'],
#         'Value': [effects_data['ATT_rewritten_rewrite'] / reward_std, 
#                  effects_data['ATU_rewritten_rewrite'] / reward_std],
#         'Stderr': [effects_data['ATT_rewritten_rewrite_stderr'] / reward_std, 
#                   effects_data['ATU_rewritten_rewrite_stderr'] / reward_std]
#     })

#     data['CI_lower'] = data['Value'] - 1.96 * data['Stderr']
#     data['CI_upper'] = data['Value'] + 1.96 * data['Stderr']

#     fig, ax = plt.subplots(figsize=(8, 4), dpi=300)

#     sns.barplot(x='Effect', y='Value', data=data, ax=ax, 
#                 palette={'ATT_rewritten_rewrite': 'blue', 'ATU_rewritten_rewrite': 'red'},
#                 alpha=0.8, edgecolor='black')

#     ax.errorbar(x=data.index, y=data['Value'], yerr=1.96*data['Stderr'], 
#                 fmt='none', color='black', capsize=5, capthick=2, elinewidth=2)

#     # ax.set_title(f'Complexity Treatment Effects ({model_name})', fontsize=16, fontweight='bold', pad=20)
#     ax.set_ylabel('Effect Size', fontsize=14)
#     # Use LaTeX for treatment indicators with proper spacing
#     ax.set_xticklabels([r'$W=1$' + ' Responses', r'$W=0$' + ' Responses'], fontsize=14)
#     ax.set_xlabel('')
#     ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)

#     legend_elements = [
#         plt.Rectangle((0,0), 1, 1, facecolor='blue', edgecolor='black', alpha=0.8, 
#                      label=r'$\widehat{\textbf{ATT}}_{\textbf{RATE}}$'),
#         plt.Rectangle((0,0), 1, 1, facecolor='red', edgecolor='black', alpha=0.8, 
#                      label=r'$\widehat{\textbf{ATU}}_{\textbf{RATE}}$')
#     ]
    
#     # Keep original title font size and style
#     ax.legend(handles=legend_elements, fontsize=12, loc='upper right', 
#              frameon=True, edgecolor='black')

#     plt.tight_layout()
#     plt.show()