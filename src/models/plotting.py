from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt

# Plot a multipage PDF with various metrics.
def plot_pdf(history, reciever_operator, precision_recall, peak_curve, filename):
    with PdfPages(filename) as pdf:
        # Plot Training and Validation History
        figure, ax = create_plot()
        plot_page(ax,
            values = history,
            title = 'Training History',
            x_label = 'Epoch',
            y_label = 'Accuracy',
            paired_colors = True)
        pdf.savefig(figure)
        plt.close()

        figure, ax = create_plot()
        plot_page(ax,
            values = reciever_operator,
            title = 'ROC Curve',
            x_label = 'False Positive Rate',
            y_label = 'True Positive Rate')
        pdf.savefig(figure)
        plt.close()

        figure, ax = create_plot()
        plot_page(ax,
            values = precision_recall,
            title = 'PR Curve',
            x_label = 'Recall',
            y_label = 'Precision')
        pdf.savefig(figure)
        plt.close()

        figure, ax = create_plot()
        plot_page(ax,
            values = peak_curve,
            title = 'Peak Recall',
            x_label = '% Peaks with J',
            y_label = '% Js in Peak')
        pdf.savefig(figure)
        plt.close()

# Plot a page with all the different baselines.
def plot_page(ax, values, title, x_label, y_label, paired_colors = False):
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
        area = curve['area']
        color = colors[i]
        label = curve['label']

        plot_curve(ax, x, y, area, color, label)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

# Plot the folds and average values.
def plot_curve(ax, x_values, y_values, area, color, label):
    # Plot Individual Folds.
    for i in range(len(x_values)):
        x = x_values[i]
        y = y_values[i]

        ax.plot(
            x, 
            y, 
            linewidth = 1,
            color = color, 
            alpha = 0.3)

    # Calculate the Means and Standard Deviations
    mean_x, mean_y, lower_y, upper_y, mean_area, std_area = interpolate_curve(x_values, y_values, area)

    # Plot the Average Across Folds
    ax.plot(
        mean_x, 
        mean_y, 
        color = 'C0',
        linewidth = 2,
        label = rf'{label} ROC (AUC = {mean_area:.2f} $\pm$ {std_area:.2f})')

    # Plot the Standard Deviation Across Folds
    ax.fill_between(
        mean_x, 
        lower_y, 
        upper_y, 
        color = color, 
        alpha = 0.2)

# Return a boilerplate figure.
def create_plot():
    figure, ax = plt.subplots(
        figsize = (8, 4), 
        dpi = 150, 
        facecolor = 'white')

    return figure, axes

# Given a list of lists, return the shortest list.
def shortest_row(array):
    current = 0
    length = len(array[0])
    for i in range(len(array)):
        if len(array[i]) < length:
            current = i
    return array[i]

# Interpolate values so curves with different data points can be averaged.
def interpolate_curve(x, y, area):
    mean_x = shortest_row(x)
    interpolate_y = []
        
    for i in range(len(x)):
        new_y = np.interp(mean_x, x[i], y[i])
        interpolate_y.append(new_y)
    
    mean_y = np.mean(interpolate_y, axis = 0)
    std_y = np.std(interpolate_y, axis = 0)

    mean_area = np.mean(area)
    std_area = np.std(area)

    lower_y = mean_y - std_y
    upper_y = mean_y + std_y
    
    return mean_x, mean_y, lower_y, upper_y, mean_area, std_area


