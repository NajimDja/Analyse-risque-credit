import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class DataVisualisation:

    def _make_grid(self, n: int, axes=None):
        """
        Returns a flat list of Axes of length n.
        - If axes is provided (single Axes, list, or ndarray), flattens and returns it.
        - Otherwise, creates a new figure with an auto-computed (rows x cols) grid.
        """
        if axes is not None:
            return np.array(axes).flatten(), None

        cols = min(n, 3)
        rows = int(np.ceil(n / cols))
        fig, ax_grid = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        axes_flat = np.array(ax_grid).flatten()

        # Hide unused axes
        for i in range(n, len(axes_flat)):
            axes_flat[i].set_visible(False)

        return axes_flat, fig

    def plot_density(self, df: pd.DataFrame, columns: list[str], bandwidth=0.3, axes=None) -> None:
        """
        Plots Kernel Density Estimation (KDE) curves for multiple numerical variables.

        Parameters:
         - df (pd.DataFrame)             : The DataFrame containing the data.
         - columns (list of str)         : List of column names to plot.
         - bandwidth (float)             : Bandwidth for KDE smoothing (default=0.3).
         - axes (Axes or array, optional): External axes to draw into. If None, a subplot
                                          grid is created automatically.
        """
        axes_flat, fig = self._make_grid(len(columns), axes)
        for ax, column in zip(axes_flat, columns):
            sns.kdeplot(df[column], bw_adjust=bandwidth, label=column, fill=True, alpha=0.5, ax=ax)
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.set_title(f"Density Plot (KDE) — {column}")
            ax.legend()
            ax.grid(True)
        if fig is not None:
            plt.tight_layout()
            plt.show()

    def plot_histogram(self, df: pd.DataFrame, columns: list[str], bins=30, axes=None) -> None:
        """
        Plots histograms for multiple variables.

        Parameters:
         - df (pd.DataFrame)             : DataFrame containing the data.
         - columns (list of str)         : List of column names to plot.
         - bins (int)                    : Number of bins (default=30).
         - axes (Axes or array, optional): External axes to draw into.
        """
        axes_flat, fig = self._make_grid(len(columns), axes)
        for ax, column in zip(axes_flat, columns):
            ax.hist(df[column], bins=bins, alpha=0.6, label=column, edgecolor='black')
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
            ax.set_title(f"Histogram — {column}")
            ax.legend()
            ax.grid(True)
        if fig is not None:
            plt.tight_layout()
            plt.show()

    def plot_hist2d(self, df: pd.DataFrame, x_column: str, y_column: str, bins=30, ax=None) -> None:
        """
        Plots a 2D histogram (heatmap) for two variables.

        Parameters:
         - df (pd.DataFrame)      : DataFrame containing the data.
         - x_column (str)         : Name of the column for X-axis.
         - y_column (str)         : Name of the column for Y-axis.
         - bins (int)             : Number of bins (default=30).
         - ax (Axes, optional)    : External axes to draw into.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = None
        h = ax.hist2d(df[x_column], df[y_column], bins=bins, cmap='Blues')
        plt.colorbar(h[3], ax=ax, label="Frequency")
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        ax.set_title(f"2D Histogram — {x_column} vs {y_column}")
        if fig is not None:
            plt.tight_layout()
            plt.show()

    def plot_boxplot(self, df: pd.DataFrame, columns: list[str], ax=None) -> None:
        """
        Plots boxplots for multiple numerical variables.

        Parameters:
         - df (pd.DataFrame)             : DataFrame containing the data.
         - columns (list of str)         : List of column names to plot.
         - ax (Axes or array, optional)  : External axes to draw into.
             - Single Axes  → all boxplots together on that axes.
             - Array of Axes → one boxplot per column.
             - None         → subplot grid created automatically.
        """
        if ax is None:
            axes_flat, fig = self._make_grid(len(columns), None)
            for single_ax, column in zip(axes_flat, columns):
                single_ax.boxplot([df[column]], labels=[column], patch_artist=True)
                single_ax.set_ylabel("Value")
                single_ax.set_title(f"Boxplot — {column}")
                single_ax.grid(True)
            plt.tight_layout()
            plt.show()
        elif isinstance(ax, (list, np.ndarray)):
            axes_flat = np.array(ax).flatten()
            for single_ax, column in zip(axes_flat, columns):
                single_ax.boxplot([df[column]], labels=[column], patch_artist=True)
                single_ax.set_ylabel("Value")
                single_ax.set_title(f"Boxplot — {column}")
                single_ax.grid(True)
        else:
            # Single external axes: all columns together
            ax.boxplot([df[col] for col in columns], labels=columns, patch_artist=True)
            ax.set_ylabel("Value")
            ax.set_title("Boxplot of Variables")
            ax.grid(True)

    def plot_pie(self, df: pd.DataFrame, columns: list[str], axes=None) -> None:
        """
        Plots a pie chart for categorical data.

        Parameters:
         - df (pd.DataFrame)             : DataFrame containing the categorical data.
         - columns (list of str)         : List of column names to plot.
         - axes (Axes or array, optional): External axes to draw into.
        """
        axes_flat, fig = self._make_grid(len(columns), axes)
        for ax, column in zip(axes_flat, columns):
            values = df[column].value_counts()
            ax.pie(values, labels=values.index, autopct='%1.1f%%',
                   startangle=140, colors=plt.cm.Paired.colors)
            ax.set_title(f"Pie Chart — {column}")
        if fig is not None:
            plt.tight_layout()
            plt.show()

    def plot_stairs(self, df: pd.DataFrame, columns: list[str], axes=None) -> None:
        """
        Plots stairs plots for multiple variables.

        Parameters:
         - df (pd.DataFrame)             : DataFrame containing the data.
         - columns (list of str)         : List of column names to plot.
         - axes (Axes or array, optional): External axes to draw into.
        """
        axes_flat, fig = self._make_grid(len(columns), axes)
        for ax, column in zip(axes_flat, columns):
            values, bin_edges = np.histogram(df[column], bins=30)
            ax.stairs(values, bin_edges, fill=True, label=column)
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
            ax.set_title(f"Stairs Plot — {column}")
            ax.legend()
            ax.grid(True)
        if fig is not None:
            plt.tight_layout()
            plt.show()

    def plot_bar(self, df: pd.DataFrame, x_column: str, y_columns: list[str], axes=None) -> None:
        """
        Plots a bar chart for multiple numerical variables.

        Parameters:
         - df (pd.DataFrame)             : DataFrame containing the data.
         - x_column (str)                : Column for X-axis (categories).
         - y_columns (list of str)       : Columns for Y-axis (numerical values).
         - axes (Axes or array, optional): External axes to draw into.
        """
        axes_flat, fig = self._make_grid(len(y_columns), axes)
        for ax, y_col in zip(axes_flat, y_columns):
            ax.bar(df[x_column], df[y_col], alpha=0.7, label=y_col)
            ax.set_xlabel(x_column)
            ax.set_ylabel("Values")
            ax.set_title(f"Bar Chart — {y_col}")
            ax.tick_params(axis='x', rotation=45)
            ax.legend()
        if fig is not None:
            plt.tight_layout()
            plt.show()

    def plot_scatter(self, df: pd.DataFrame, x_column: str, y_columns: list[str], axes=None) -> None:
        """
        Plots scatter plots for multiple numerical variables against a common X-axis.

        Parameters:
         - df (pd.DataFrame)             : DataFrame containing the data.
         - x_column (str)                : Column for X-axis.
         - y_columns (list of str)       : Columns for Y-axis.
         - axes (Axes or array, optional): External axes to draw into.
        """
        axes_flat, fig = self._make_grid(len(y_columns), axes)
        for ax, y_col in zip(axes_flat, y_columns):
            ax.scatter(df[x_column], df[y_col], alpha=0.6, label=y_col)
            ax.set_xlabel(x_column)
            ax.set_ylabel("Values")
            ax.set_title(f"Scatter — {x_column} vs {y_col}")
            ax.legend()
            ax.grid(True)
        if fig is not None:
            plt.tight_layout()
            plt.show()

    def plot_fill_between(self, df: pd.DataFrame, x_column: str, y_columns: list[str],
                          lower_offset=5, upper_offset=5, axes=None) -> None:
        """
        Plots filled area charts for multiple variables.

        Parameters:
         - df (pd.DataFrame)             : DataFrame containing the data.
         - x_column (str)                : Column for X-axis.
         - y_columns (list of str)       : Columns for Y-axis.
         - lower_offset (float)          : Offset for lower boundary (default=5).
         - upper_offset (float)          : Offset for upper boundary (default=5).
         - axes (Axes or array, optional): External axes to draw into.
        """
        axes_flat, fig = self._make_grid(len(y_columns), axes)
        for ax, y_col in zip(axes_flat, y_columns):
            ax.plot(df[x_column], df[y_col], label=y_col)
            ax.fill_between(df[x_column],
                            df[y_col] - lower_offset,
                            df[y_col] + upper_offset,
                            alpha=0.2)
            ax.set_xlabel(x_column)
            ax.set_ylabel("Values")
            ax.set_title(f"Fill Between — {y_col}")
            ax.legend()
            ax.grid(True)
        if fig is not None:
            plt.tight_layout()
            plt.show()

    def plot_ecdf(self, df: pd.DataFrame, columns: list[str], axes=None) -> None:
        """
        Plots ECDF (Empirical Cumulative Distribution Function) for multiple variables.

        Parameters:
         - df (pd.DataFrame)             : DataFrame containing the data.
         - columns (list of str)         : Columns to plot.
         - axes (Axes or array, optional): External axes to draw into.
        """
        axes_flat, fig = self._make_grid(len(columns), axes)
        for ax, column in zip(axes_flat, columns):
            sorted_data = np.sort(df[column])
            y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            ax.plot(sorted_data, y, marker=".", linestyle="none", label=column)
            ax.set_xlabel("Value")
            ax.set_ylabel("ECDF")
            ax.set_title(f"ECDF — {column}")
            ax.legend()
            ax.grid(True)
        if fig is not None:
            plt.tight_layout()
            plt.show()