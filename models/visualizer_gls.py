import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.colors import ListedColormap

def objective_function(x, y):
    return np.sin(0.5*x)*np.cos(0.3*y) + 0.3*np.cos(0.2*x*y) + 0.1*x

class GuidedLocalSearchVisualizer:
    def __init__(self):
        self.x_range = (-10, 10)
        self.y_range = (-10, 10)
        self.penalty_factor = 0.2
        self.max_iter = 50
        self.current_iter = 0

        self.reset_solution()

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(14, 6))
        plt.subplots_adjust(bottom=0.2)

        x = np.linspace(-10, 10, 100)
        y = np.linspace(-10, 10, 100)
        self.X, self.Y = np.meshgrid(x, y)
        self.Z = objective_function(self.X, self.Y)

        self.contour = self.ax1.contourf(self.X, self.Y, self.Z, levels=20, cmap='viridis')
        self.fig.colorbar(self.contour, ax=self.ax1, label='Objective Value')
        self.ax1.set_title('Objective Function and Search Path')
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')

        self.path_line, = self.ax1.plot([], [], 'r-', lw=1, alpha=0.5)
        self.current_point, = self.ax1.plot([], [], 'ro', markersize=8)

        self.penalty_cmap = ListedColormap(['white', *plt.cm.Reds(np.linspace(0, 1, 256))])
        self.penalty_plot = self.ax2.imshow(self.penalty_grid.T, origin='lower',
                                            extent=[-10, 10, -10, 10],
                                            cmap=self.penalty_cmap, vmin=0.1)
        self.fig.colorbar(self.penalty_plot, ax=self.ax2, label='Penalty Count')
        self.ax2.set_title('Penalty Landscape')
        self.ax2.set_xlabel('X')
        self.ax2.set_ylabel('Y')

        self.ax_next = plt.axes([0.3, 0.05, 0.15, 0.075])
        self.ax_reset = plt.axes([0.55, 0.05, 0.15, 0.075])
        self.btn_next = Button(self.ax_next, 'Next Step')
        self.btn_reset = Button(self.ax_reset, 'Reset')

        self.btn_next.on_clicked(self.next_step)
        self.btn_reset.on_clicked(self.reset)

        self.update_plot()

    def reset_solution(self):
        """Initialize or reset the solution and penalties"""
        self.current_x = np.random.uniform(*self.x_range)
        self.current_y = np.random.uniform(*self.y_range)
        self.current_score = objective_function(self.current_x, self.current_y)
        self.penalty_grid = np.zeros((100, 100))
        self.history = [(self.current_x, self.current_y, self.current_score, self.penalty_grid.copy())]
        self.current_iter = 0

    def next_step(self, event):
        if self.current_iter >= self.max_iter:
            return

        neighbors = []
        for _ in range(20):
            step_size = 2 * (1 - self.current_iter/self.max_iter)
            dx, dy = np.random.uniform(-step_size, step_size, 2)
            x = np.clip(self.current_x + dx, *self.x_range)
            y = np.clip(self.current_y + dy, *self.y_range)
            score = objective_function(x, y)

            grid_x = int((x - self.x_range[0]) / (self.x_range[1] - self.x_range[0]) * 99)
            grid_y = int((y - self.y_range[0]) / (self.y_range[1] - self.y_range[0]) * 99)
            augmented_score = score + self.penalty_factor * self.penalty_grid[grid_x, grid_y]

            neighbors.append((x, y, score, augmented_score))

        neighbors.sort(key=lambda x: x[3])
        best_x, best_y, best_score, best_augmented = neighbors[0]

        if best_augmented >= self.current_score:
            grid_x = int((self.current_x - self.x_range[0]) / (self.x_range[1] - self.x_range[0]) * 99)
            grid_y = int((self.current_y - self.y_range[0]) / (self.y_range[1] - self.y_range[0]) * 99)
            self.penalty_grid[grid_x, grid_y] += 1

        self.current_x, self.current_y, self.current_score = best_x, best_y, best_score
        self.current_iter += 1

        self.history.append((self.current_x, self.current_y, self.current_score, self.penalty_grid.copy()))

        self.update_plot()

    def reset(self, event):
        self.reset_solution()
        self.update_plot()

    def update_plot(self):
        x_vals = [h[0] for h in self.history]
        y_vals = [h[1] for h in self.history]

        if len(x_vals) > 0:
            self.path_line.set_data(x_vals, y_vals)
            self.current_point.set_data([x_vals[-1]], [y_vals[-1]])

        self.penalty_plot.set_array(self.history[-1][3].T)
        if np.max(self.history[-1][3]) > 0:
            self.penalty_plot.set_clim(vmax=np.max(self.history[-1][3]))

        self.fig.suptitle(f'Guided Local Search - Iteration {self.current_iter}/{self.max_iter}\n'
                          f'Current Value: {self.history[-1][2]:.4f}', y=0.95)

        self.fig.canvas.draw()

if __name__ == "__main__":
    visualizer = GuidedLocalSearchVisualizer()
    plt.show()