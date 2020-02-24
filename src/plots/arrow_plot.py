import matplotlib.pyplot as plt

def arrow_plot(x,y, plot_code='ro'):
    for i in range(len(x) - 1):
        plt.arrow(x[i], y[i], (x[i + 1] - x[i]), (y[i + 1] - y[i]),
                  head_width=0.1, head_length=0.1)
    plt.plot(x, y, plot_code)


def arrow_interp_plot(x, y, interp_x, interp_y, fig_size=(10, 10)):
    plt.figure(figsize=fig_size)
    plt.plot(interp_x, interp_y, 'bo')
    for i in range(len(x) - 1):
        plt.arrow(x[i], y[i], (x[i + 1] - x[i]), (y[i + 1] - y[i]),
                  head_width=0.1, head_length=0.1)
    plt.plot(x, y, 'ro')
