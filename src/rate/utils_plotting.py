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

def setup_plots():
    """Helper function to set up consistent plot styling"""
    sns.set_theme(style="whitegrid", font="serif")
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['text.usetex'] = True
    # Add bolder text support in LaTeX
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{times}\usepackage{bm}'

def plot_scores(templates_data, SCORED_DIR):
    setup_plots()
    
    n_plots = len(templates_data)
    fig = plt.figure(figsize=(8 * n_plots, 4), dpi=300)
    gs = fig.add_gridspec(1, n_plots)
    axes = [fig.add_subplot(gs[0, i]) for i in range(n_plots)]
    
    if n_plots == 1:
        axes = [axes]
    
    colors = ["#1f77b4", "#ff7f0e"]
    legend_lines = [
        Line2D([0], [0], color=colors[0], linewidth=2, 
               label=r"$\textbf{Original}$"),
        Line2D([0], [0], color=colors[1], linewidth=2, 
               label=r"$\textbf{Rewrite}^2$")
    ]
    
    for idx, (ax, template) in enumerate(zip(axes, templates_data)):
        dataset_filename = Path(template["dataset_filename"])
        scores = load_dataset_from_json(SCORED_DIR / "complete" / dataset_filename)
        
        original_scores = []
        rewrite_scores = []
        
        for key, data_point in scores.items():
            original_scores.append(data_point[template["reward_key"]].get("original", 0) / data_point[template["reward_key"]].get("reward_std", 1))
            rewrite_scores.append(data_point[template["reward_key"]].get("rewritten rewrite", 0) / data_point[template["reward_key"]].get("reward_std", 1))

        
        sns.kdeplot(data=original_scores, color=colors[0], fill=True, alpha=0.5, linewidth=2, ax=ax)
        sns.kdeplot(data=rewrite_scores, color=colors[1], fill=True, alpha=0.5, linewidth=2, ax=ax)
        
        ax.set_title(r'\textbf{' + f"{template['dataset_name']} {template['concept']} {template['score']} Rewards" + '}', 
                    fontsize=24, pad=20)
        ax.set_xlabel(r'\textbf{Reward}', fontsize=18)
        if idx == 0:
            ax.set_ylabel(r'\textbf{Density}', fontsize=18)
        else:
            ax.set_ylabel("")
        
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        
        if idx == 0:
            legend = ax.legend(handles=legend_lines,
                             fontsize=18,
                             loc='upper left',
                             frameon=True, 
                             fancybox=True,
                             shadow=True)
    
    plt.subplots_adjust(wspace=0.3, left=0.1, right=0.9)
    plt.show()

def rewrite_bias(effects_data, titles):
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
        ax.set_title(r'\textbf{' + titles[idx] + '}', fontsize=24, pad=22)
        
        # Remove legend if it's not the first subplot
        if idx > 0 and ax.get_legend() != None:
            ax.get_legend().remove()

    plt.tight_layout()
    plt.show()


def naive_vs_RATE(all_data, all_templates, reward_models, normalize=True):
    setup_plots()
    
    n_models = len(reward_models)
    fig, axes = plt.subplots(1, n_models, figsize=(18, 6), dpi=300)
    axes = [axes] if n_models == 1 else axes
    
    for idx, model in enumerate(reward_models):
        ax = axes[idx]
        model_data = [data for data, template in zip(all_data, all_templates) if template['score'] == model]
        model_templates = [template for template in all_templates if template['score'] == model]
        
        grouped_data = defaultdict(lambda: defaultdict(list))
        for data, template in zip(model_data, model_templates):
            concept = template['concept']
            for effect_type in ['naive_effect', 'ATE_rewritten_rewrite']:
                value = data[effect_type]
                error = data[f'{effect_type}_stderr']
                
                if normalize:
                    print(data.get('reward_std'))
                    norm_factor = data.get('reward_std', 1)  # Default to 1 if `reward_std` missing
                    value /= norm_factor
                    error /= norm_factor
                grouped_data[concept][effect_type].append(value)
                grouped_data[concept][f'{effect_type}_error'].append(error)
        
        valid_concepts = []
        for concept in grouped_data:
            if all(len(grouped_data[concept][key]) > 0 for key in grouped_data[concept]):
                valid_concepts.append(concept)
                for key in grouped_data[concept]:
                    if key.endswith('_error'):
                        grouped_data[concept][key] = np.sqrt(np.mean(np.array(grouped_data[concept][key])**2))
                    else:
                        grouped_data[concept][key] = np.mean(grouped_data[concept][key])
        
        if not valid_concepts:
            print(f"Error: No valid concepts found for model '{model}'. Skipping this plot.")
            continue
        
        plot_data = []
        for concept in valid_concepts:
            for effect_type in ['naive_effect', 'ATE_rewritten_rewrite']:
                label = r'$\textbf{Naive}$' if effect_type == 'naive_effect' else r'$\widehat{\textbf{ATE}}_{\textbf{RATE}}$'
                plot_data.append({
                    'Concept': concept,
                    'Effect Type': label,
                    'Effect Size': grouped_data[concept][effect_type],
                    'Error': grouped_data[concept][f'{effect_type}_error']
                })
        
        df = pd.DataFrame(plot_data)
        
        sns.barplot(x='Concept', y='Effect Size', hue='Effect Type', data=df, ax=ax,
                    palette=['#1f77b4', '#ff7f0e'], alpha=0.7, capsize=0.1, errorbar=None)
        
        bar_width = 0.4
        for i, effect_type in enumerate(df['Effect Type'].unique()):
            effect_data = df[df['Effect Type'] == effect_type].set_index('Concept')
            x = np.arange(len(valid_concepts))
            y = effect_data.loc[valid_concepts, 'Effect Size']
            yerr = effect_data.loc[valid_concepts, 'Error'] * 1.96
            
            x_pos = x + (i - 0.5) * bar_width
            ax.errorbar(x_pos, y, yerr=yerr, fmt='none', c='black', capsize=5)
        
        ax.set_title(r'\textbf{' + f"{model}" + '}', fontsize=20, pad=20)
        ax.set_xticks(np.arange(len(valid_concepts)))
        ax.set_xticklabels([r'\textbf{' + concept + '}' for concept in valid_concepts], 
                           rotation=45, ha='right', fontsize=18)
        
        ax.set_xlabel('')
        if idx == 0:
            ax.set_ylabel(r'\textbf{Effect Size}', fontsize=18)
        else:
            ax.set_ylabel('')  # No y-axis label for other plots
        
        # **Explicitly show y-tick values for all plots:**
        ax.tick_params(axis='y', labelleft=True)  # Turn on y-tick labels for all subplots
        
        if idx == 0:
            legend = ax.legend(title='', loc='upper left', fontsize=20, frameon=True)
            legend.get_title().set_fontweight('bold')
        else:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
                
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=18)
    
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.15, top=0.9)
    plt.show()

def synthetic_subplots(data_list1, effects_templates1, target_concept1, spurious_concept1,
                      data_list2, effects_templates2, target_concept2, spurious_concept2):
    setup_plots()
    
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

    df1 = prepare_plot_data(data_list1, effects_templates1)
    df2 = prepare_plot_data(data_list2, effects_templates2)

    # Create figure with more space for labels
    fig = plt.figure(figsize=(14, 5), dpi=300)
    gs = fig.add_gridspec(1, 2, wspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    def plot_subplot(ax, df, target_concept, spurious_concept, effects_templates):
        palette = sns.color_palette("deep", 3)
        effect_labels = {
            'naive_effect': r'$\textbf{Naive}$',
            'ATE_rewritten_rewrite': r'$\widehat{\textbf{ATE}}\ (\textbf{Rewrite}^2)$',
            'ATE_single_rewrite': r'$\widehat{\textbf{ATE}}\ (\textbf{Rewrite})$'
        }
        
        for i, (effect_type, label) in enumerate(effect_labels.items()):
            effect_data = df[df['Effect Type'] == effect_type]
            sns.lineplot(x='Correlation', y='Effect Size', data=effect_data, 
                        label=label, color=palette[i], linewidth=2.5, ax=ax)
            ax.fill_between(effect_data['Correlation'], effect_data['Lower CI'], 
                          effect_data['Upper CI'], color=palette[i], alpha=0.2)

        # Format conditional probability with bold text
        ax.set_xlabel(fr'$\textbf{{P}}(\textbf{{{spurious_concept}}}|\textbf{{{target_concept}}})$', fontsize=18)
        ax.set_ylabel(r'$\textbf{Reward}$', fontsize=18)
        
        dataset = effects_templates[0]['dataset_name']
        model_name = effects_templates[0]['score']
        ax.set_title(r'\textbf{' + f"Effect of {target_concept} on {model_name}" + '}', 
                    fontsize=24, pad=20)
        
        if ax == ax1:
            ax.legend(loc='upper left', fontsize=16, frameon=True)
        else:
            ax.get_legend().remove()

        if ax == ax1:
            ax.set_ylabel(r'$\textbf{Effect Size}$', fontsize=18)
        else:
            ax.set_ylabel('')
            
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)

        # x lim should be 0, 1
        ax.set_xlim(0, 1)

        for effect_type in ['ATE_rewritten_rewrite', 'ATE_single_rewrite']:
            effect_data = df[df['Effect Type'] == effect_type]
            x = effect_data['Correlation']
            y = effect_data['Effect Size']
            slope, intercept = np.polyfit(x, y, 1)
            print(f"Slope of {effect_labels[effect_type]}: {slope}")

    plot_subplot(ax1, df1, target_concept1, spurious_concept1, effects_templates1)
    plot_subplot(ax2, df2, target_concept2, spurious_concept2, effects_templates2)

    # Adjust layout manually instead of using tight_layout
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.9, wspace=0.25)
    plt.show()

def synthetic_plot(data_list, templates, target_concept, spurious_concept, x_lab: str):
    setup_plots()
    
    def prepare_plot_data(data_list, templates):
        plot_data = []
        for data, template in zip(data_list, templates):
            x_val = int(template['dataset_filename'].split('_typos_')[1].split('.')[0])
            
            effect_mapping = {
                'naive_effect': 'Naive Effect',
                'ATE_rewritten_rewrite': 'ATE_rewritten_rewrite',
                'ATE_single_rewrite': 'ATE_single_rewrite'
            }
            
            for effect_type, display_name in effect_mapping.items():
                if effect_type in data:
                    plot_data.append({
                        x_lab: x_val,
                        'Effect Type': display_name,
                        'Effect Size': float(data[effect_type]) / data.get('reward_std', 1),
                        'Lower CI': (float(data[effect_type]) - float(data.get(f'{effect_type}_stderr', 0)) * 1.96) / data.get('reward_std', 1),
                        'Upper CI': (float(data[effect_type]) + float(data.get(f'{effect_type}_stderr', 0)) * 1.96) / data.get('reward_std', 1)
                    })
        return pd.DataFrame(plot_data)

    df = prepare_plot_data(data_list, templates)
    plt.figure(figsize=(8, 6), dpi=300)
    ax = plt.gca()

    palette = sns.color_palette("deep", 3)
    effect_type_mapping = {
        'Naive Effect': 'Naive Estimate',
        'ATE_rewritten_rewrite': r'$\widehat{\textbf{ATE}}\ (\textbf{Rewrite}^2)$',
        'ATE_single_rewrite': r'$\widehat{\textbf{ATE}}\ (\textbf{Single\ Rewrite})$'
    }

    for i, effect_type in enumerate(effect_type_mapping.keys()):
        effect_data = df[df['Effect Type'] == effect_type]
        if not effect_data.empty:
            sns.lineplot(x=x_lab, y='Effect Size', data=effect_data, 
                        label=effect_type_mapping[effect_type], 
                        color=palette[i], linewidth=2.5, ax=ax)
            ax.fill_between(effect_data[x_lab], effect_data['Lower CI'], 
                          effect_data['Upper CI'],
                          color=palette[i], alpha=0.2)

    ax.set_xlabel('Percent Typos per Review Starting with Vowel', fontsize=18, fontweight='bold')
    ax.set_ylabel('Reward', fontsize=18, fontweight='bold')
    dataset = templates[0]['dataset_name']
    model_name = templates[0]['score']
    ax.set_title(f"Effect of {target_concept} on {model_name}", 
                fontsize=24, fontweight='bold')
    ax.legend(title='', loc='lower left', fontsize=12, frameon=True)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)

    for effect_type in ['ATE_rewritten_rewrite', 'ATE_single_rewrite']:
        effect_data = df[df['Effect Type'] == effect_type]
        if not effect_data.empty:
            x = effect_data[x_lab]
            y = effect_data['Effect Size']
            slope, intercept = np.polyfit(x, y, 1)
            print(f"Slope of {effect_type_mapping.get(effect_type, effect_type)}: {slope:.4f}")

    plt.tight_layout()
    plt.show()

def att_atu(effects_data, reward_std, model_name):
    setup_plots()

    data = pd.DataFrame({
        'Effect': ['ATT_rewritten_rewrite', 'ATU_rewritten_rewrite'],
        'Value': [effects_data['ATT_rewritten_rewrite'] / reward_std, 
                 effects_data['ATU_rewritten_rewrite'] / reward_std],
        'Stderr': [effects_data['ATT_rewritten_rewrite_stderr'] / reward_std, 
                  effects_data['ATU_rewritten_rewrite_stderr'] / reward_std]
    })

    data['CI_lower'] = data['Value'] - 1.96 * data['Stderr']
    data['CI_upper'] = data['Value'] + 1.96 * data['Stderr']

    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)

    sns.barplot(x='Effect', y='Value', data=data, ax=ax, 
                palette={'ATT_rewritten_rewrite': 'blue', 'ATU_rewritten_rewrite': 'red'},
                alpha=0.8, edgecolor='black')

    ax.errorbar(x=data.index, y=data['Value'], yerr=1.96*data['Stderr'], 
                fmt='none', color='black', capsize=5, capthick=2, elinewidth=2)

    ax.set_title(f'Complexity Treatment Effects ({model_name})', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Effect Size', fontsize=14)
    # Use LaTeX for treatment indicators with proper spacing
    ax.set_xticklabels([r'$W=1$' + ' Responses', r'$W=0$' + ' Responses'], fontsize=14)
    ax.set_xlabel('')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)

    legend_elements = [
        plt.Rectangle((0,0), 1, 1, facecolor='blue', edgecolor='black', alpha=0.8, 
                     label=r'$\widehat{\textbf{ATT}}_{\textbf{RATE}}$'),
        plt.Rectangle((0,0), 1, 1, facecolor='red', edgecolor='black', alpha=0.8, 
                     label=r'$\widehat{\textbf{ATU}}_{\textbf{RATE}}$')
    ]
    
    # Keep original title font size and style
    ax.legend(handles=legend_elements, fontsize=12, loc='upper right', 
             frameon=True, edgecolor='black')

    plt.tight_layout()
    plt.show()