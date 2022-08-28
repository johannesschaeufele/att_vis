import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.widgets as widgets
from matplotlib.backend_bases import MouseButton

epsilon = 1E-4


def elu(x):
    return np.where(x > 0, x, np.exp(x) - 1)


def softmax(x, axis):
    x = x - np.max(x, axis=axis, keepdims=True)

    x = np.exp(x)
    sums = np.sum(x, axis=axis, keepdims=True)
    x /= sums + epsilon

    return x


class AttentionVisualization:

    def __init__(self, qks, precompute_sim=True):
        """
        Constructs and starts the attention visualization for the given query and key inputs

        Args:
            qks: numpy ndarray of shape (N, 2, H, W, C), containing the query and key tensor data
            precompute_sim: Whether to pre-compute the similarity (pre-activation attention) volume
        """
        self.precompute_sim = precompute_sim

        self.h = qks[0][0].shape[0]
        self.w = qks[0][0].shape[1]

        self.data_index = 0
        self.selected_x = 0
        self.selected_y = 0
        self.attention_mode = 2

        self.qks = qks
        self.sims = [None for _ in range(len(qks))]

        divider_x = 0.2
        divider_y = 0.7
        pad_x = 0.025
        pad_y = 0.05

        self.ax_radio = plt.axes([pad_x, divider_y + pad_y, divider_x - 2 * pad_x, 1.0 - divider_y - 2 * pad_y])
        self.ax_vis = plt.axes([divider_x + pad_x, pad_y, 1.0 - divider_x - 2 * pad_x, 1.0 - 2 * pad_y])
        self.vis_colorbar = None

        plt.connect("button_press_event", lambda event: self.on_click(event))

        if len(self.qks) > 1:
            self.ax_slider = plt.axes([pad_x, pad_y, divider_x - 2 * pad_x, divider_y - 2 * pad_y])
            self.slider = widgets.Slider(self.ax_slider,
                                         "Data index",
                                         0,
                                         len(self.qks) - 1,
                                         valinit=self.data_index,
                                         valstep=1,
                                         orientation="vertical",
                                         initcolor="none")
            self.slider.on_changed(lambda value: self.on_slider(value))

        plt.text(0.5, 1.075, "Activation", transform=self.ax_radio.transAxes, ha="center")
        self.radio = widgets.RadioButtons(self.ax_radio, ("softmax", "elu", "identity"), active=self.attention_mode, activecolor="blue")
        self.radio.on_clicked(lambda label: self.on_radio(label))

    def get_attention(self):
        q, k = self.qks[self.data_index]

        vmin = 0.0
        vmax = epsilon

        if self.precompute_sim:
            if self.sims[self.data_index] is not None:
                sim = self.sims[self.data_index]
            else:
                sim = np.einsum("x y c, u v c -> x y u v", q, k)
                self.sims[self.data_index] = sim
            sim = sim[self.selected_y:self.selected_y + 1, self.selected_x:self.selected_x + 1, :, :]
        else:
            q_value = q[self.selected_y:self.selected_y + 1, self.selected_x:self.selected_x + 1, :]
            sim = np.einsum("x y c, u v c -> x y u v", q_value, k)

        if self.attention_mode == 0:
            attention = softmax(sim, (2, 3))

            vmax = 1.0
        elif self.attention_mode == 1:
            attention = elu(sim) + 1

            vmax = np.max(attention)
        else:
            attention = sim

            vmin = np.min(attention)
            vmax = np.max(attention)

        attention_window = attention[0, 0, :, :]

        return attention_window, vmin, vmax

    def on_click(self, event):
        if event.inaxes is not self.ax_vis:
            return

        if event.button is MouseButton.LEFT:
            x_data = event.xdata
            y_data = event.ydata
            if x_data is not None and y_data is not None:
                x_int = int(round(x_data))
                y_int = int(round(y_data))

                if x_int != self.selected_x or y_int != self.selected_y:
                    self.selected_x = x_int
                    self.selected_y = y_int
                    self.redraw()

    def on_radio(self, label):
        index = [t.get_text() for t in self.radio.labels].index(label)

        if index != self.attention_mode:
            self.attention_mode = index
            self.redraw()

    def on_slider(self, value):
        if value != self.data_index:
            self.data_index = value
            self.redraw()

    def redraw(self):
        cmap = "gray"

        rect = patches.Rectangle((self.selected_x - 0.5, self.selected_y - 0.5), 1, 1, linewidth=2, edgecolor="r", facecolor="none")
        self.ax_vis.patches = []
        self.ax_vis.add_patch(rect)

        data, vmin, vmax = self.get_attention()
        cm = self.ax_vis.matshow(data, cmap=cmap, vmin=vmin, vmax=vmax)

        if self.vis_colorbar:
            self.vis_colorbar.update_normal(cm)
        else:
            self.vis_colorbar = plt.colorbar(cm, ax=self.ax_vis)

        plt.show()

    def show(self):
        self.redraw()
