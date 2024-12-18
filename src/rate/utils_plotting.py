import json
import random
import os
import re
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns
from collections import defaultdict
from pathlib import Path
from rate.utils import load_dataset_from_json, write_to_json
from rate.treatment_effects import calculate_treatment_effects
import pandas as pd

def create_latex_tables_from_samples(file_paths, num_samples=10, max_text_length=100):
    def load_json_lines(file_path):
        with open(file_path, 'r') as file:
            return [json.loads(line) for line in file]

    def find_armo_rm_key(item):
        return next((key for key in item.keys() if 'ArmoRM' in key), None)

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
            
            armo_rm_key = find_armo_rm_key(item)
            if armo_rm_key:
                reward_original = format_reward(item[armo_rm_key].get('original', 'N/A'))
                reward_rewrite = format_reward(item[armo_rm_key].get('rewrite', 'N/A'))
                reward_rewrite_of_rewrite = format_reward(item[armo_rm_key].get('rewritten rewrite', 'N/A'))
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

def plot_scores(template, SCORED_DIR):
    sns.set_theme(style="whitegrid", font="serif")
    dataset_filename = Path(template["dataset_filename"])
    scores = load_dataset_from_json(SCORED_DIR / "complete" / dataset_filename)
    
    # Extract scores
    original_scores = []
    rewrite_scores = []
    
    for key, data_point in scores.items():
        original_scores.append(data_point[template["reward_key"]].get("original", 0))
        rewrite_scores.append(data_point[template["reward_key"]].get("rewritten rewrite", 0))
    
    # Calculate the means for both original and rewrite scores
    original_mean = np.mean(original_scores)
    rewrite_mean = np.mean(rewrite_scores)
    
    # Plot setup
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    colors = ["#1f77b4", "#ff7f0e"]
    
    # KDE plot for scores
    sns.kdeplot(original_scores, label="Original", color=colors[0], fill=True, alpha=0.5, linewidth=2, ax=ax)
    sns.kdeplot(rewrite_scores, label="Rewrite", color=colors[1], fill=True, alpha=0.5, linewidth=2, ax=ax)

    # Set titles and labels
    ax.set_title(f"{template['dataset_name']} {template['concept']} {template['score']} Rewards", fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel("Reward", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Custom legend
    ax.legend(handles=[Line2D([0], [0], color=colors[0], linewidth=2, label="Original"),
                       Line2D([0], [0], color=colors[1], linewidth=2, label="Rewrite")], 
              title="Score Type", title_fontsize='14', fontsize=12, frameon=True, fancybox=True, shadow=True)
    
    # Show plot
    plt.tight_layout()
    plt.show()

def rewrite_bias(effects_data, titles):
    sns.set_theme(style="whitegrid", font="serif")
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    # Create the plot with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=300)

    # Define colors for each effect
    colors = [
        '#FF7F0E', '#FFA54F',  # Orange pair
        '#1F77B4', '#6CA6CD',  # Blue pair
        '#D62728', '#FF6A6A'   # Red pair
    ]

    # Renaming dictionary for consistency
    rename_dict = {
        'ATE_stderr_naive': 'ATE_naive_stderr', 
        'ATT_stderr_naive': 'ATT_naive_stderr', 
        'ATU_stderr_naive': 'ATU_naive_stderr'
    }

    for idx, ax in enumerate(axes):
        # Rename fields in the current dataset entry if they exist
        for key, new_key in rename_dict.items():
            if key in effects_data[idx]:
                effects_data[idx][new_key] = effects_data[idx].pop(key)

        # Prepare the data
        data = pd.DataFrame({
            'Effect': [
                'ATE (Rewrite)', r'ATE (Rewrite$^2$)', 
                'ATT (Rewrite)', r'ATT (Rewrite$^2$)', 
                'ATU (Rewrite)', r'ATU (Rewrite$^2$)'
            ],
            'Value': [effects_data[idx][f'{effect}'] / effects_data[idx]['reward_std'] for effect in ['ATE', 'ATE_naive', 'ATT', 'ATT_naive', 'ATU', 'ATU_naive']],
            'Stderr': [effects_data[idx][f'{effect}_stderr'] / effects_data[idx]['reward_std'] for effect in ['ATE', 'ATE_naive', 'ATT', 'ATT_naive', 'ATU', 'ATU_naive']]
        })


        # Calculate confidence intervals
        data['CI_lower'] = data['Value'] - 1.96 * data['Stderr']
        data['CI_upper'] = data['Value'] + 1.96 * data['Stderr']

        # Plot the bars
        sns.barplot(x='Effect', y='Value', data=data, ax=ax, 
                    palette=dict(zip(data['Effect'], colors)),
                    alpha=0.8, edgecolor='black')

        # Add error bars
        ax.errorbar(x=data.index, y=data['Value'], yerr=1.96 * data['Stderr'], 
                    fmt='none', color='black', capsize=5, capthick=2, elinewidth=2)

        # Customize the plot
        ax.set_ylabel('Effect Size', fontsize=14)
        ax.set_xlabel('')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        ax.set_title(titles[idx], fontsize=14, fontweight='bold')

        # Rotate x-axis labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.tick_params(axis='both', which='major', labelsize=12)

    plt.suptitle('Correcting for Rewrite Bias', fontsize=22, fontweight='bold')
    plt.tight_layout()
    plt.show()

def naive_vs_RATE(all_data, all_templates, reward_models, normalize=None):
    sns.set_theme(style="whitegrid", font="serif")
    n_models = len(reward_models)
    fig, axes = plt.subplots(1, n_models, figsize=(16, 8), dpi=300, sharey=True)
    axes = [axes] if n_models == 1 else axes  # Ensure axes is always a list
    
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['font.size'] = 14
    
    for idx, model in enumerate(reward_models):
        ax = axes[idx]
        model_data = [data for data, template in zip(all_data, all_templates) if template['score'] == model]
        model_templates = [template for template in all_templates if template['score'] == model]
        
        # Group data by concept and average effects across datasets
        grouped_data = defaultdict(lambda: defaultdict(list))
        for data, template in zip(model_data, model_templates):
            concept = template['concept']
            for effect_type in ['naive_effect', 'ATE']:
                value = data[effect_type]
                error = data[f'{effect_type}_stderr']
                
                if normalize:
                    norm_factor = data['reward_std'] if normalize == "reward_std" else abs(data['naive_effect'])
                    value /= norm_factor
                    error /= norm_factor

                grouped_data[concept][effect_type].append(value)
                grouped_data[concept][f'{effect_type}_error'].append(error)
        
        # Calculate averages and standard errors
        valid_concepts = []
        for concept in grouped_data:
            if all(len(grouped_data[concept][key]) > 0 for key in grouped_data[concept]):
                valid_concepts.append(concept)
                for key in grouped_data[concept]:
                    if key.endswith('_error'):
                        grouped_data[concept][key] = np.sqrt(np.mean(np.array(grouped_data[concept][key])**2))
                    else:
                        grouped_data[concept][key] = np.mean(grouped_data[concept][key])
            else:
                print(f"Warning: Insufficient data for concept '{concept}' in model '{model}'. Skipping this concept.")
        
        if not valid_concepts:
            print(f"Error: No valid concepts found for model '{model}'. Skipping this plot.")
            continue
        
        # Prepare data for plotting
        plot_data = []
        for concept in valid_concepts:
            for effect_type in ['naive_effect', 'ATE']:
                plot_data.append({
                    'Concept': concept,
                    'Effect Type': 'Naive' if effect_type == 'naive_effect' else 'RATE',
                    'Effect Size': grouped_data[concept][effect_type],
                    'Error': grouped_data[concept][f'{effect_type}_error']
                })
        
        df = pd.DataFrame(plot_data)
        
        # Plotting
        sns.barplot(x='Concept', y='Effect Size', hue='Effect Type', data=df, ax=ax,
                    palette=['#1f77b4', '#ff7f0e'], alpha=0.7, capsize=0.1, errorbar=None)
        
        # Add error bars manually
        bar_width = 0.4
        for i, effect_type in enumerate(['Naive', 'RATE']):
            effect_data = df[df['Effect Type'] == effect_type].set_index('Concept')
            x = np.arange(len(valid_concepts))
            y = effect_data.loc[valid_concepts, 'Effect Size']
            yerr = effect_data.loc[valid_concepts, 'Error'] * 1.96
            
            # Adjust x position for each group
            x_pos = x + (i - 0.5) * bar_width
            
            ax.errorbar(x_pos, y, yerr=yerr, fmt='none', c='black', capsize=5)
        
        ax.set_title(f"{model}", fontsize=18, fontweight='bold')
        ax.set_xticks(np.arange(len(valid_concepts)))
        ax.set_xticklabels(valid_concepts, rotation=45, ha='right')
        ax.set_xlabel('')
        if idx == 0:
            ax.set_ylabel('Model-Specific Effect Size', fontsize=16)
        else:
            ax.set_ylabel('')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=18)
        if idx == n_models - 1:
            ax.legend(title='', loc='upper right', bbox_to_anchor=(1.3, 1))
            ax.legend(fontsize='x-large')
        else:
            ax.legend_.remove()
    fig.suptitle('Naive vs RATE Estimates Across Models', fontsize=22, fontweight='bold')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()
    
def synthetic_subplots(data_list1, effects_templates1, target_concept1, spurious_concept1,
              data_list2, effects_templates2, target_concept2, spurious_concept2):
    sns.set_theme(style="whitegrid", font="serif")
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    
    def prepare_plot_data(data_list, effects_templates):
        plot_data = []
        for data, template in zip(data_list, effects_templates):
            data["ATE_naive_stderr"] = data.get("ATE_stderr_naive", 0)
            correlation = int(template['dataset_filename'].split('_')[-1].split('.')[0]) / 10
            for effect_type in ['naive_effect', 'ATE', 'ATE_naive']:
                plot_data.append({
                    'Correlation': correlation,
                    'Effect Type': 'Naive Effect' if effect_type == 'naive_effect' else 
                                   'ATE' if effect_type == 'ATE' else 'ATE Naive',
                    'Effect Size': data[effect_type],
                    'Lower CI': data[effect_type] - data.get(f'{effect_type}_stderr', 0) * 1.96,
                    'Upper CI': data[effect_type] + data.get(f'{effect_type}_stderr', 0) * 1.96
                })
        return pd.DataFrame(plot_data)

    # Prepare data for plotting
    df1 = prepare_plot_data(data_list1, effects_templates1)
    df2 = prepare_plot_data(data_list2, effects_templates2)

    # Set up plot style and dimensions
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    def plot_subplot(ax, df, target_concept, spurious_concept, effects_templates):
        palette = sns.color_palette("deep", 3)  # Three colors for Naive Effect, ATE, ATE Naive
        for i, effect_type in enumerate(['Naive Effect', 'ATE', 'ATE Naive']):
            if effect_type == 'ATE Naive':
                name = 'ATE (Single Rewrite)' 
            elif effect_type == 'ATE':
                name = r'ATE (Rewrite$^2$)'
            else:
                name = 'Naive Estimate'
            effect_data = df[df['Effect Type'] == effect_type]
            sns.lineplot(x='Correlation', y='Effect Size', data=effect_data, 
                         label=name, color=palette[i], linewidth=2.5, ax=ax)
            ax.fill_between(effect_data['Correlation'], effect_data['Lower CI'], effect_data['Upper CI'],
                            color=palette[i], alpha=0.2)

        # Customize axes labels and title
        ax.set_xlabel(f'P({spurious_concept}|{target_concept})', fontsize=14, fontweight='bold')
        ax.set_ylabel('Reward', fontsize=14, fontweight='bold')
        dataset = effects_templates[0]['dataset_name']
        model_name = effects_templates[0]['score']
        ax.set_title(f"Effect of {target_concept} on {model_name}\n(Data from {dataset})", fontsize=16, fontweight='bold')
        ax.legend(title='', loc='upper left', fontsize=12, frameon=True)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)

        # add slope of both "ATE" and "ATE Naive" put on right-hand side
        for i, effect_type in enumerate(['ATE', 'ATE Naive']):
            if effect_type == 'ATE Naive':
                name = 'ATE (Single Rewrite)' 
            elif effect_type == 'ATE':
                name = r'ATE (Rewrite$^2$)'
            effect_data = df[df['Effect Type'] == effect_type]
            x = effect_data['Correlation']
            y = effect_data['Effect Size']
            slope, intercept = np.polyfit(x, y, 1)
            print(f"Slope of {name}: {slope}")

    # Plot each subplot
    plot_subplot(ax1, df1, target_concept1, spurious_concept1, effects_templates1)
    plot_subplot(ax2, df2, target_concept2, spurious_concept2, effects_templates2)

    plt.tight_layout()
    plt.show()

def synthetic_plot(data_list, templates, target_concept, spurious_concept, x_lab: str):
    sns.set_theme(style="whitegrid", font="serif")
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    def prepare_plot_data(data_list, templates):
        plot_data = []
        for data, template in zip(data_list, templates):
            # Extract the number of typos from the filename
            x_val = int(template['dataset_filename'].split('_typos_')[1].split('.')[0])
            
            # Map the effect types to their display names
            effect_mapping = {
                'naive_effect': 'Naive Effect',
                'ATE_rewritten_rewrite': 'ATE',
                'ATE_single_rewrite': 'ATE Naive'
            }
            
            for effect_type, display_name in effect_mapping.items():
                if effect_type in data:  # Check if the effect type exists in the data
                    plot_data.append({
                        x_lab: x_val,
                        'Effect Type': display_name,
                        'Effect Size': float(data[effect_type]),  # Ensure numeric value
                        'Lower CI': float(data[effect_type]) - float(data.get(f'{effect_type}_stderr', 0)) * 1.96,
                        'Upper CI': float(data[effect_type]) + float(data.get(f'{effect_type}_stderr', 0)) * 1.96
                    })
        
        return pd.DataFrame(plot_data)

    # Prepare data for plotting
    df = prepare_plot_data(data_list, templates)

    # Set up plot style and dimensions
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 6), dpi=300)
    ax = plt.gca()

    # Plot the data
    palette = sns.color_palette("deep", 3)
    effect_type_mapping = {
        'Naive Effect': 'Naive Estimate',
        'ATE': r'ATE (Rewrite$^2$)',
        'ATE Naive': 'ATE (Single Rewrite)'
    }

    for i, effect_type in enumerate(effect_type_mapping.keys()):
        effect_data = df[df['Effect Type'] == effect_type]
        if not effect_data.empty:  # Only plot if we have data for this effect type
            sns.lineplot(x=x_lab, y='Effect Size', data=effect_data, 
                        label=effect_type_mapping[effect_type], 
                        color=palette[i], linewidth=2.5, ax=ax)
            ax.fill_between(effect_data[x_lab], effect_data['Lower CI'], 
                          effect_data['Upper CI'],
                          color=palette[i], alpha=0.2)

    # Customize axes labels and title
    ax.set_xlabel('Percent Typos per Review Starting with Vowel', fontsize=14, fontweight='bold')
    ax.set_ylabel('Reward', fontsize=14, fontweight='bold')
    dataset = templates[0]['dataset_name']
    model_name = templates[0]['score']
    ax.set_title(f"Effect of {target_concept} on {model_name}\n(Data from {dataset})", 
                fontsize=16, fontweight='bold')
    ax.legend(title='', loc='lower left', fontsize=12, frameon=True)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)

    # Calculate and print slopes
    for effect_type in ['ATE', 'ATE Naive']:
        effect_data = df[df['Effect Type'] == effect_type]
        if not effect_data.empty:
            x = effect_data[x_lab]
            y = effect_data['Effect Size']
            slope, intercept = np.polyfit(x, y, 1)
            print(f"Slope of {effect_type_mapping[effect_type]}: {slope:.4f}")

    plt.tight_layout()
    plt.show()

def att_atu(effects_data, reward_std, model_name):
    sns.set_theme(style="whitegrid", font="serif")
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    # Prepare the data with normalization
    data = pd.DataFrame({
        'Effect': ['ATT', 'ATU'],
        'Value': [effects_data['ATT'] / reward_std, 
                  effects_data['ATU'] / reward_std],
        'Stderr': [effects_data['ATT_stderr'] / reward_std, 
                   effects_data['ATU_stderr'] / reward_std]
    })

    # Calculate confidence intervals
    data['CI_lower'] = data['Value'] - 1.96 * data['Stderr']
    data['CI_upper'] = data['Value'] + 1.96 * data['Stderr']

    # Create the plot with matching size and resolution
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)

    # Plot the bars
    sns.barplot(x='Effect', y='Value', data=data, ax=ax, 
                palette={'ATT': 'blue', 'ATU': 'red'},
                alpha=0.8, edgecolor='black')

    # Add error bars
    ax.errorbar(x=data.index, y=data['Value'], yerr=1.96*data['Stderr'], 
                fmt='none', color='black', capsize=5, capthick=2, elinewidth=2)

    # Customize the plot
    ax.set_title(f'Complexity Treatment Effects ({model_name})', fontsize=16, fontweight='bold', pad = 20)
    ax.set_ylabel('Effect Size', fontsize=14)
    ax.set_xticklabels(['W = 1 Responses', 'W = 0 Responses'], fontsize=14)
    ax.set_xlabel('')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)

    # Add confidence interval ranges in the legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='blue', edgecolor='black', alpha=0.8, 
                      label=f'Corrected ATT'),
        plt.Rectangle((0,0),1,1, facecolor='red', edgecolor='black', alpha=0.8, 
                      label=f'Corrected ATU')
    ]
    ax.legend(handles=legend_elements, title='95% Confidence Intervals', 
              loc='upper right', frameon=True, edgecolor='black')

    plt.tight_layout()
    plt.show()