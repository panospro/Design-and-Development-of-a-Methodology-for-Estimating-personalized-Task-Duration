import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from datetime import datetime
from scipy.stats import linregress

def plot_progress(data, subplots_per_figure=2):
    grouped_data = defaultdict(list)
    for item in data:
        for category in item['categories']:
            for focus_area in item['focus_areas']:
                grouped_data[(category, focus_area)].append(item)
    
    plot_data = sorted(
        {key: items for key, items in grouped_data.items() if len(items) > 2}.items(),
        key=lambda x: (x[0][0], x[0][1])
    )
    
    num_plots = len(plot_data)
    num_figures = (num_plots + subplots_per_figure - 1) // subplots_per_figure

    for fig_index in range(num_figures):
        fig, axes = plt.subplots(
            nrows=min(subplots_per_figure, num_plots - fig_index * subplots_per_figure),
            figsize=(10, 5 * min(subplots_per_figure, num_plots - fig_index * subplots_per_figure))
        )
        
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        
        for ax, ((category, focus_area), items) in zip(axes, plot_data[fig_index * subplots_per_figure:(fig_index + 1) * subplots_per_figure]):
            items.sort(key=lambda x: datetime.fromisoformat(x['createdAt'].replace('Z', '')))
            dates = [datetime.fromisoformat(item['createdAt'].replace('Z', '')) for item in items]
            progress = [item['points']['done'] for item in items]
            expected_points = [item['points']['total'] for item in items]
            
            ax.plot(dates, progress, marker='o', label='Points Done')
            ax.plot(dates, expected_points, marker='x', linestyle='--', label='Expected Points')

            avg_prog = np.mean(progress)
            std_prog = np.std(progress)
            
            ax.axhline(avg_prog, color='r', linestyle='--', label=f'Average ({avg_prog:.2f})')
            ax.fill_between(dates, avg_prog - std_prog, avg_prog + std_prog, color='r', alpha=0.2, label=f'Std Dev ({std_prog:.2f})')
            
            max_idx, min_idx = np.argmax(progress), np.argmin(progress)
            ax.annotate(f'Max: {progress[max_idx]}', xy=(dates[max_idx], progress[max_idx]), xytext=(10, 10), textcoords='offset points', arrowprops=dict(arrowstyle='->', lw=1))
            ax.annotate(f'Min: {progress[min_idx]}', xy=(dates[min_idx], progress[min_idx]), xytext=(10, -10), textcoords='offset points', arrowprops=dict(arrowstyle='->', lw=1))
            
            slope, intercept, r_value, _, _ = linregress([date.timestamp() for date in dates], progress)
            ax.plot(dates, [slope * date.timestamp() + intercept for date in dates], color='g', linestyle='-', label=f'Trendline (R²={r_value**2:.2f})')
            
            ax.set_title(f"({category},{focus_area})")
            ax.set_xlabel("Date")
            ax.set_ylabel("Points Done")
            ax.grid(True)
            ax.legend()
        
        plt.tight_layout()
        plt.show(block=False)

def plot_average_points_per_task(data):
    titles = [item['title'] for item in data]
    total_points = [item['points']['total'] for item in data]
    done_points = [item['points']['done'] for item in data]

    x = range(len(data))

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    ax.bar(x, total_points, width=bar_width, label='Expected Points', align='center', color='skyblue', edgecolor='black')
    ax.bar(x, done_points, width=bar_width, label='Burned Points', align='edge', color='lightgreen', edgecolor='black')

    ax.set_xlabel('Tasks')
    ax.set_ylabel('Points')
    ax.set_title('Points Distribution per Task')
    ax.set_xticks(x)
    ax.set_xticklabels(titles, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.show(block=False)

def plot_average_points_per_category(data):
    def aggregate_data(data):
        aggregated_data = defaultdict(lambda: {'total_sum': 0, 'done_sum': 0, 'count': 0})
        
        for item in data:
            key = (tuple(item['categories']), tuple(item['focus_areas']))
            aggregated_data[key]['total_sum'] += item['points']['total']
            aggregated_data[key]['done_sum'] += item['points']['done']
            aggregated_data[key]['count'] += 1
        
        for key in aggregated_data:
            aggregated_data[key]['total_avg'] = aggregated_data[key]['total_sum'] / aggregated_data[key]['count']
            aggregated_data[key]['done_avg'] = aggregated_data[key]['done_sum'] / aggregated_data[key]['count']
        
        return aggregated_data

    aggregated_data = aggregate_data(data)
    
    categories_areas = [' / '.join(cat + foc) for cat, foc in aggregated_data.keys()]
    total_points_avg = [value['total_avg'] for value in aggregated_data.values()]
    done_points_avg = [value['done_avg'] for value in aggregated_data.values()]

    x = range(len(aggregated_data))

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    ax.bar(x, total_points_avg, width=bar_width, label='Average Expected Points', align='center', color='skyblue', edgecolor='black')
    ax.bar(x, done_points_avg, width=bar_width, label='Average Done Points', align='edge', color='lightgreen', edgecolor='black')

    ax.set_xlabel('Category / Area of Focus')
    ax.set_ylabel('Points')
    ax.set_title('Average Points Distribution per Category and Area of Focus')
    ax.set_xticks(x)
    ax.set_xticklabels(categories_areas, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.show(block=False)

def plot_people_results(data):
    plot_progress(data)
    plot_average_points_per_task(data)
    plot_average_points_per_category(data)