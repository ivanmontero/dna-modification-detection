from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
import numpy as np

# Plot a multipage PDF with various metrics.
def plot_pdf(results, filename):
    history = results['history']
    receiver_operator = results['receiver_operator']
    precision_recall = results['precision_recall']
    peak_curve = results['peak_curve']
    peak_baseline = results['peak_baseline']

    with PdfPages(filename) as pdf:
        # Plot training and validation history.
        figure, ax = create_plot()
        plot_page(
            figure = figure,
            ax = ax,
            values = history,
            title = 'Training History',
            x_label = 'Epoch',
            y_label = 'Accuracy',
            paired_colors = True,
            skip_area = True)
        pdf.savefig(figure)
        plt.close()

        # Plot ROC curves.
        figure, ax = create_plot()
        plot_page(
            figure = figure,
            ax = ax,
            values = receiver_operator,
            title = 'ROC Curve',
            x_label = 'False Positive Rate',
            y_label = 'True Positive Rate')
        # Plot baseline.
        ax.plot(
            [0,1],
            [0,1],
            linestyle = '--',
            color = 'black')
        pdf.savefig(figure)
        plt.close()

        # Plot precision recall curves.
        figure, ax = create_plot()
        plot_page(
            figure = figure,
            ax = ax,
            values = precision_recall,
            title = 'PR Curve',
            x_label = 'Recall',
            y_label = 'Precision')
        # Plot baseline.
        ax.plot(
            [0, 1], 
            [peak_baseline, peak_baseline], 
            linestyle = '--', 
            color = 'black')
        pdf.savefig(figure)
        plt.close()

        # Plot peak curves.
        figure, ax = create_plot()
        plot_page(
            figure = figure,
            ax = ax,
            values = peak_curve,
            title = 'Peak Recall',
            x_label = '% Peaks with J',
            y_label = '% Js in Peak')
        # Plot baseline.
        ax.plot(
            [0, 1], 
            [peak_baseline, peak_baseline], 
            linestyle = '--', 
            color = 'black')
        pdf.savefig(figure)
        plt.close()

# Plot a page with all the different baselines.
def plot_page(figure, ax, values, title, x_label, y_label, paired_colors = False, skip_area = False):
    # Pair the colors for training vs validation so it's easier to interpret.
    if paired_colors:
        colors = plt.cm.tab20(range(20))
    else:
        colors = plt.cm.tab10(range(10))

    # For Each Curve on the Plot
    for i in range(len(values)):
        curve = values[i]
        x = curve['x']
        y = curve['y']

        if not skip_area:
            area = curve['area']
        else:
            area = None

        color = colors[i]
        label = curve['label']

        plot_curve(ax, x, y, area, color, label)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(bbox_to_anchor = (1.05, 1), loc = 'upper left')
    figure.tight_layout()

# Plot the folds and average values.
def plot_curve(ax, x_values, y_values, area, color, label):
    # # Plot Individual Folds.
    # for i in range(len(x_values)):
    #     x = x_values[i]
    #     y = y_values[i]

    #     ax.plot(
    #         x, 
    #         y, 
    #         linewidth = 1,
    #         color = color, 
    #         alpha = 0.3)

    # Calculate the Means and Standard Deviations
    if not area: 
        mean_x, mean_y, lower_y, upper_y = interpolate_curve(x_values, y_values)
        print (f'{label}, {mean_y[-1]}')

        label = f'{label}'
    else:
        mean_x, mean_y, lower_y, upper_y, mean_area, std_area = interpolate_curve(x_values, y_values, area)
        print (f'{label}, {mean_area}, {std_area}')

        label = f'{label}\n' + rf'(AUC = {mean_area:.2f} $\pm$ {std_area:.2f})'

        

    # Plot the Average Across Folds
    ax.plot(
        mean_x, 
        mean_y, 
        color = color,
        linewidth = 2,
        label = label)

    # Plot the Standard Deviation Across Folds
    ax.fill_between(
        mean_x, 
        lower_y, 
        upper_y, 
        color = color, 
        alpha = 0.2)

    # Fix the y axis between 0 and 1. 
    ax.set_ylim([0,1])

# Return a boilerplate figure.
def create_plot():
    figure, ax = plt.subplots(
        figsize = (8, 4), 
        dpi = 100, 
        facecolor = 'white')

    return figure, ax

# Given a list of lists, return the shortest list.
def shortest_row(array):
    current = 0
    length = len(array[0])
    for i in range(len(array)):
        if len(array[i]) < length:
            current = i

    return array[i]

# Interpolate values so curves with different data points can be averaged.
def interpolate_curve(x, y, area = None):
    mean_x = shortest_row(x)
    interpolate_y = []
        
    for i in range(len(x)):
        new_y = np.interp(mean_x, x[i], y[i])
        interpolate_y.append(new_y)
    
    mean_y = np.mean(interpolate_y, axis = 0)
    std_y = np.std(interpolate_y, axis = 0)

    lower_y = mean_y - std_y
    upper_y = mean_y + std_y

    if not area:
        return mean_x, mean_y, lower_y, upper_y

    mean_area = np.mean(area)
    std_area = np.std(area)

    return mean_x, mean_y, lower_y, upper_y, mean_area, std_area


