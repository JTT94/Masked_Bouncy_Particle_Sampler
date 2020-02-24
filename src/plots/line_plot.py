import matplotlib.pyplot as plt
from matplotlib import rc

# plot settings
rc('text', usetex=True)


def line_plot(x, y, col = 'r',
                 title='', xlab='x', ylab= 'y',
                 file_path=None, fig_size = (6 ,5),
                 base_x=None, base_y=None, base_col = 'b'):
    fig = plt.figure(figsize=fig_size)

    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes([left, bottom, width, height])
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    plt.plot(x, y, col)

    if (base_x is not None) and (base_y is not None):
        plt.plot(base_x, base_y, base_col)

    plt.show()

    if file_path is not None:
        plt.savefig(file_path)
