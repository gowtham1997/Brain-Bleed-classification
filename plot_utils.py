import matplotlib.pyplot as plt
import numpy as np
import windowing as w


def show_slice_details(ax, scan_index, shape, component='images'):
    font_caption = {'family': 'serif',
                    'color': 'darkred',
                    'weight': 'normal',
                    'size': 18}
    font = {'family': 'serif',
            'color': 'darkred',
            'weight': 'normal',
            'size': 15}
    ax.set_xlabel('Shape: {}'.format(shape[1]), fontdict=font)
    ax.set_ylabel('Shape: {}'.format(shape[2]), fontdict=font)
    title = 'Scan' if component == 'images' else 'Mask'
    ax.set_title('{} #{}, slice #{} \n \n'.format(
        title, scan_index, ax.index), fontdict=font_caption)
    ax.text(0.2, -0.25, 'Total slices: {}'.format(shape[0]),
            fontdict=font_caption, transform=ax.transAxes)


def multi_slice_viewer(volume, scan_index, max_len, component='images'):

    def process_key(event):
        fig = event.canvas.figure
        ax = fig.axes[0]
        if event.key == 'left':
            previous_slice(ax)
        elif event.key == 'right':
            next_slice(ax)
        fig.canvas.draw()

    def previous_slice(ax):
        """Go to the previous slice."""
        volume = ax.volume
        ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
        ax.images[0].set_array(volume[ax.index])
        show_slice_details(ax, scan_index, volume.shape)

    def next_slice(ax):
        """Go to the next slice."""
        volume = ax.volume
        ax.index = (ax.index + 1) % volume.shape[0]
        ax.images[0].set_array(volume[ax.index])
        show_slice_details(ax, scan_index, volume.shape)

    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[0] // 2
    ax.imshow(volume[ax.index], cmap=plt.cm.gray)
    show_slice_details(ax, scan_index, volume.shape)
    fig.canvas.mpl_connect('key_press_event', process_key)
    # plt.show()


def show_slices(batches, scan_indices, grid=True, **kwargs):
    """ Plot slice with number n_slice from scan with index
        given by scan_index from batch
    """

    # fetch some arguments, make iterables out of args
    def iterize(arg):
        return arg if isinstance(arg, (list, tuple)) else (arg, )

    components = kwargs.get('components', 'images')
    batches, scan_indices, components = [iterize(arg) for arg in (
        batches, scan_indices, components)]

    # lengthen args
    n_boxes = max(len(arg) for arg in (batches, scan_indices))

    def lengthen(arg):
        return arg if len(arg) == n_boxes else arg * n_boxes

    batches, scan_indices, components = [lengthen(arg) for arg in (
        batches, scan_indices, components)]

    # plot slices
    _, axes = plt.subplots(1, n_boxes, squeeze=False,
                           figsize=(10, 4 * n_boxes))

    zipped = zip(range(n_boxes), batches, scan_indices,
                 components)
    for i, batch, scan_index, component in zipped:
        # print(scan_index, n_slice, component)
        # print(component)
        slc = batch.get(scan_index, component)
        slc = w.GetLUTValue(slc, window=100, level=40)
        # reversing the slices
        slc = slc[::-1]
        multi_slice_viewer(slc, scan_index, len(slc))

        # set inverse-spacing grid
        if grid:
            inv_spacing = 1 / batch.get(scan_index, 'spacing').reshape(-1)[1:]
            step_mult = 50
            xticks = np.arange(0, slc.shape[0], step_mult * inv_spacing[0])
            yticks = np.arange(0, slc.shape[1], step_mult * inv_spacing[1])
            axes[0][i].set_xticks(xticks, minor=True)
            axes[0][i].set_yticks(yticks, minor=True)
            axes[0][i].set_xticks([], minor=False)
            axes[0][i].set_yticks([], minor=False)

            axes[0][i].grid(color='r', linewidth=1.5, alpha=0.5, which='minor')

    plt.show()
