import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns
from collections import defaultdict
import pandas as pd

def rewrite_bias(effects_data):
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

    # Titles for each subplot
    titles = [
        "FsfairX-LLAMA3-RM-v0.1",
        "NCSOFT/Llama-3-OffsetBias-RM-8B",
        "ArmoRM"
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
            correlation = int(template['dataset_filename'].split('_')[-1].split('.')[0]) / 10
            for effect_type in ['naive_effect', 'ATE']:
                plot_data.append({
                    'Correlation': correlation,
                    'Effect Type': 'Naive' if effect_type == 'naive_effect' else 'RATE',
                    'Effect Size': data[effect_type],
                    'Lower CI': data[effect_type] - data[f'{effect_type}_stderr'] * 1.96,
                    'Upper CI': data[effect_type] + data[f'{effect_type}_stderr'] * 1.96
                })
        return pd.DataFrame(plot_data)

    df1 = prepare_plot_data(data_list1, effects_templates1)
    df2 = prepare_plot_data(data_list2, effects_templates2)

    # Set up the plot style and size
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=300)

    def plot_subplot(ax, df, target_concept, spurious_concept, effects_templates):
        palette = sns.color_palette("deep", 2)
        for i, effect_type in enumerate(['Naive', 'RATE']):
            effect_data = df[df['Effect Type'] == effect_type]
            sns.lineplot(x='Correlation', y='Effect Size', data=effect_data, 
                         label=effect_type, color=palette[i], linewidth=2.5, ax=ax)
            ax.fill_between(effect_data['Correlation'], effect_data['Lower CI'], effect_data['Upper CI'],
                            color=palette[i], alpha=0.2)

        ax.set_xlabel(f'P({spurious_concept}|{target_concept})', fontsize=14, fontweight='bold')
        ax.set_ylabel('Reward', fontsize=14, fontweight='bold')

        dataset = effects_templates[0]['dataset_name']
        model_name = effects_templates[0]['score']
        ax.set_title(f"Effect of {target_concept} on {model_name}\n(Data from {dataset})", fontsize=16, fontweight='bold')

        ax.legend(title='', loc='upper left', fontsize=12, frameon=True)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)

    plot_subplot(ax1, df1, target_concept1, spurious_concept1, effects_templates1)
    plot_subplot(ax2, df2, target_concept2, spurious_concept2, effects_templates2)

    plt.tight_layout()
    plt.show()

def synthetic(data_list, effects_templates, target_concept, spurious_concept):
    sns.set_theme(style="whitegrid", font="serif")
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    plot_data = []
    for data, template in zip(data_list, effects_templates):
        correlation = int(template['dataset_filename'].split('_')[-1].split('.')[0]) / 10
        for effect_type in ['naive_effect', 'ATE']:
            plot_data.append({
                'Correlation': correlation,
                'Effect Type': 'Naive' if effect_type == 'naive_effect' else 'RATE',
                'Effect Size': data[effect_type],
                'Lower CI': data[effect_type] - data[f'{effect_type}_stderr'] * 1.96,
                'Upper CI': data[effect_type] + data[f'{effect_type}_stderr'] * 1.96
            })

    df = pd.DataFrame(plot_data)

    # Set up the plot style and size (standardized)
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 4), dpi=300)  # Standardized size and DPI
    
    # Plot lines with error bands
    palette = sns.color_palette("deep", 2)
    for i, effect_type in enumerate(['Naive', 'RATE']):
        effect_data = df[df['Effect Type'] == effect_type]
        sns.lineplot(x='Correlation', y='Effect Size', data=effect_data, 
                     label=effect_type, color=palette[i], linewidth=2.5)
        plt.fill_between(effect_data['Correlation'], effect_data['Lower CI'], effect_data['Upper CI'],
                         color=palette[i], alpha=0.2)

    plt.xlabel(f'P({spurious_concept}|{target_concept})', fontsize=14, fontweight='bold')
    plt.ylabel('Reward', fontsize=14, fontweight='bold')

    dataset = effects_templates[0]['dataset_name']
    model_name = effects_templates[0]['score']
    plt.title(f"Effect of {target_concept} on {model_name}\n(Data from {dataset})", fontsize=16, fontweight='bold')

    plt.legend(title='', loc='upper left', fontsize=12, frameon=True)
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    # Add grid lines
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

def synthetic_3way(data_list, effects_templates, target_concept, spurious_concept):
    sns.set_theme(style="whitegrid", font="serif")
    plt.rcParams["font.size"] = 14
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

    plot_data = []
    for data, template in zip(data_list, effects_templates):
        correlation = (
            int(template["dataset_filename"].split("_")[-1].split(".")[0]) / 10
        )
        for effect_type in ["naive_effect", "ATE_naive", "ATE"]:
            if effect_type == "naive_effect":
                effect_name = "Naive"
                stderr = data["naive_effect_stderr"]
            elif effect_type == "ATE_naive":
                effect_name = r"$ATE\ (Rewrite)$"
                stderr = data["ATE_stderr_naive"]
            else:
                effect_name = r"$ATE\ (Rewrite^{2})$"
                stderr = data["ATE_stderr"]
            effect_name
            plot_data.append(
                {
                    "Correlation": correlation,
                    "Effect Type": effect_name,
                    "Effect Size": data[effect_type],
                    "Lower CI": data[effect_type]
                    - stderr * 1.96,
                    "Upper CI": data[effect_type]
                    + stderr * 1.96,
                }
            )

    df = pd.DataFrame(plot_data)

    # Set up the plot style and size (standardized)
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 4), dpi=300)  # Standardized size and DPI

    # Plot lines with error bands
    palette = sns.color_palette("deep", 3)
    for i, effect_type in enumerate(["Naive", r"$ATE\ (Rewrite)$", r"$ATE\ (Rewrite^{2})$"]):
        effect_data = df[df["Effect Type"] == effect_type]
        sns.lineplot(
            x="Correlation",
            y="Effect Size",
            data=effect_data,
            label=effect_type,
            color=palette[i],
            linewidth=2.5,
        )
        plt.fill_between(
            effect_data["Correlation"],
            effect_data["Lower CI"],
            effect_data["Upper CI"],
            color=palette[i],
            alpha=0.2,
        )

    plt.xlabel(
        f"P({spurious_concept}|{target_concept})", fontsize=14, fontweight="bold"
    )
    plt.ylabel("Reward", fontsize=14, fontweight="bold")

    dataset = effects_templates[0]["dataset_name"]
    model_name = effects_templates[0]["score"]
    plt.title(
        f"Effect of {target_concept} on {model_name}\n(Data from {dataset})",
        fontsize=16,
        fontweight="bold",
    )

    plt.legend(title="", loc="upper left", fontsize=12, frameon=True)
    plt.tick_params(axis="both", which="major", labelsize=12)

    # Add grid lines
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()

def att_atu(effects_data, reward_std):
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
    ax.set_title('Complexity Treatment Effects (FsfairX-LLaMA3-RM-v0.1)', fontsize=16, fontweight='bold', pad = 20)
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