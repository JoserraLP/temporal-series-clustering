from temporal_series_clustering.patterns.experiment_patterns import *
from temporal_series_clustering.visualization.pattern_plotter import PatternPlotter

if __name__ == "__main__":
    patterns = generate_similar_patterns(weekday='weekday')
    # patterns = generate_patterns(weekday='weekday')
    app = PatternPlotter(patterns)
    app.mainloop()
