import tkinter as tk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class PatternPlotter(tk.Tk):
    """
    Pattern Plotter class for showing the different patterns on a Tkinter window

    :param patterns: temporal patterns to show
    """

    def __init__(self, patterns):
        tk.Tk.__init__(self)
        self.patterns = patterns
        self.current_pattern = 0

        # Create the figure and the line
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.line, = self.ax.plot(patterns[0])

        # Create the canvas for the plot and add it to the tkinter window
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Create the slider
        self.slider = tk.Scale(self, from_=0, to=len(patterns) - 1, orient=tk.HORIZONTAL, command=self.update)
        self.slider.pack(side=tk.BOTTOM, fill=tk.X)

    def update(self, event):
        # Update the value of the figure
        self.current_pattern = int(self.slider.get())
        self.line.set_ydata(self.patterns[self.current_pattern])
        self.canvas.draw()
