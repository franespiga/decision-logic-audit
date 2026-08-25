import matplotlib.pyplot as plt

def plot_attribution_rankings(top_values, top_names, title, **kwargs):
    # Create the plot
    top_values, indices = top_values.sort(descending = False)
    top_names = top_names[indices]

    config = kwargs.get('config', {'figure_size': (10, 8)})
    plt.figure(figsize=config.figure_size)
    bars = plt.barh(range(len(top_values)), top_values)

    # Color bars based on sign
    for i, (bar, value) in enumerate(zip(bars, top_values)):
        if value < 0:
            bar.set_color('red')
        else:
            bar.set_color('blue')

    plt.yticks(range(len(top_values)), top_names)
    plt.xlabel('Attribution')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if 'output_path' in kwargs:
        plt.savefig(kwargs['output_path'], format='png', dpi=900)
    else:
        plt.show()
