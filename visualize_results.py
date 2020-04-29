import matplotlib.pyplot as plt


def draw_function(x, y, plot = plt, as_line = False, color = 'blue'):
    if as_line:
        plot.plot(x[:, 1], y, color = color)
    else:
        plot.scatter(x[:, 1],y, color = color, s=0.001)

def visualize_gd_results(epoch_results, lr, x, y):
    saved_epochs_n = len(epoch_results)

    n_rows = saved_epochs_n // 4
    n_cols = saved_epochs_n//n_rows

    subplots = plt.subplots(n_rows, n_cols)
    subplots[0].suptitle(f'Linear regression with Gradient Descent (lr = {lr})')
    subplots = subplots[1].reshape(saved_epochs_n, )

    for i, plot in enumerate(subplots):
        draw_function(x, y, plot)
        draw_function(x, epoch_results[i]['y_hat'], plot, True, 'red')
        
        counted_legend = f'w_1 = {"{:.3f}".format(epoch_results[i]["w"][1])}\nw_0 = {"{:.4f}".format(epoch_results[i]["w"][0])}'
        data_legend = 'Data'
        
        plot.legend([counted_legend, data_legend])
        plot.set_title(label = f"epoch = {epoch_results[i]['e']}")
