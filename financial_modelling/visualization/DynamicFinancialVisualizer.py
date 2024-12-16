from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QLabel,
    QComboBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QSplitter,
)
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import pandas as pd


class DynamicFinancialVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dynamic Financial Data Visualizer")
        self.setGeometry(100, 100, 1200, 800)

        # Splitter for layout
        splitter = QSplitter()

        # Left panel (controls + plot)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Right panel (data table)
        self.data_table = QTableWidget()
        splitter.addWidget(left_panel)
        splitter.addWidget(self.data_table)
        splitter.setSizes([800, 400])

        self.setCentralWidget(splitter)

        # Instruction label
        self.label = QLabel("Select columns and expiry to plot:")
        left_layout.addWidget(self.label)

        # Dropdowns for selecting expiry, deltas, and IVs
        self.expiry_dropdown = QComboBox()
        left_layout.addWidget(QLabel("Select Expiry:"))
        left_layout.addWidget(self.expiry_dropdown)

        # Dropdowns for Calls
        self.calls_delta_dropdown = QComboBox()
        self.calls_iv_dropdown = QComboBox()
        left_layout.addWidget(QLabel("Select X-Axis Column (Delta for Calls):"))
        left_layout.addWidget(self.calls_delta_dropdown)
        left_layout.addWidget(QLabel("Select Y-Axis Column (IV for Calls):"))
        left_layout.addWidget(self.calls_iv_dropdown)

        # Dropdowns for Puts
        self.puts_delta_dropdown = QComboBox()
        self.puts_iv_dropdown = QComboBox()
        left_layout.addWidget(QLabel("Select X-Axis Column (Delta for Puts):"))
        left_layout.addWidget(self.puts_delta_dropdown)
        left_layout.addWidget(QLabel("Select Y-Axis Column (IV for Puts):"))
        left_layout.addWidget(self.puts_iv_dropdown)

        # Plot area
        self.plot_widget = pg.PlotWidget(title="Dynamic Financial Chart")
        left_layout.addWidget(self.plot_widget)

        # Buttons for loading data and plotting
        self.load_button = QPushButton("Load Data")
        self.load_button.clicked.connect(self.load_data)
        left_layout.addWidget(self.load_button)

        self.plot_button = QPushButton("Plot Data")
        self.plot_button.clicked.connect(self.plot_data)
        left_layout.addWidget(self.plot_button)

        # Connect expiry dropdown change to dynamic updates
        self.expiry_dropdown.currentTextChanged.connect(self.plot_data)

        # Initialize data
        self.data = None

    def load_data(self):
        """Load data from a CSV file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv)")
        if file_path:
            try:
                # Load CSV with ";" separator
                self.data = pd.read_csv(file_path, sep=";")
                self.data.columns = self.data.columns.str.strip()  # Clean column names

                # Replace commas with periods and convert to float
                for col in self.data.columns:
                    try:
                        self.data[col] = self.data[col].str.replace(",", ".").astype(float)
                    except (ValueError, AttributeError):
                        continue

                # Populate dropdowns
                self.expiry_dropdown.clear()
                self.calls_delta_dropdown.clear()
                self.calls_iv_dropdown.clear()
                self.puts_delta_dropdown.clear()
                self.puts_iv_dropdown.clear()

                # Populate unique expiries
                if "EXPIRE_DATE" in self.data.columns:
                    unique_expiries = self.data["EXPIRE_DATE"].unique()
                    self.expiry_dropdown.addItems(map(str, unique_expiries))
                else:
                    self.label.setText("Error: 'EXPIRE_DATE' column not found in the dataset.")
                    return

                # Populate column dropdowns
                column_list = self.data.columns.tolist()
                self.calls_delta_dropdown.addItems(column_list)
                self.calls_iv_dropdown.addItems(column_list)
                self.puts_delta_dropdown.addItems(column_list)
                self.puts_iv_dropdown.addItems(column_list)

                # Set default dropdown selections
                if "STRIKE" in column_list:
                    self.calls_delta_dropdown.setCurrentText("STRIKE")
                    self.puts_delta_dropdown.setCurrentText("STRIKE")
                if "C_IV" in column_list:
                    self.calls_iv_dropdown.setCurrentText("C_IV")
                if "P_IV" in column_list:
                    self.puts_iv_dropdown.setCurrentText("P_IV")

                self.label.setText("Data loaded successfully! Select expiry and columns to plot.")
            except Exception as e:
                self.label.setText(f"Error loading data: {e}")

    def plot_data(self):
        """Plot data based on selected columns and expiry."""
        if self.data is None:
            self.label.setText("No data loaded. Please load a file first.")
            return

        try:
            # Filter data by expiry
            selected_expiry = self.expiry_dropdown.currentText()
            print(f"Selected Expiry: {selected_expiry}")
            
            self.data["EXPIRE_DATE"] = self.data["EXPIRE_DATE"].astype(str)
            filtered_data = self.data[self.data["EXPIRE_DATE"] == selected_expiry].copy()
            
            if filtered_data.empty:
                self.plot_widget.clear()
                self.label.setText(f"No data available for expiry '{selected_expiry}'.")
                self.update_table(pd.DataFrame())  # Clear the table if no data
                return

            # Debug: Print filtered data shape
            print(f"Filtered Data Shape: {filtered_data.shape}")

            # Get selected columns for calls and puts
            calls_delta_col = self.calls_delta_dropdown.currentText()
            calls_iv_col = self.calls_iv_dropdown.currentText()
            puts_delta_col = self.puts_delta_dropdown.currentText()
            puts_iv_col = self.puts_iv_dropdown.currentText()

            # Debug: Print selected column names
            print(f"Calls Delta Column: {calls_delta_col}, Calls IV Column: {calls_iv_col}")
            print(f"Puts Delta Column: {puts_delta_col}, Puts IV Column: {puts_iv_col}")

            # Ensure numeric data for plotting
            filtered_data[calls_delta_col] = pd.to_numeric(filtered_data[calls_delta_col], errors="coerce")
            filtered_data[calls_iv_col] = pd.to_numeric(filtered_data[calls_iv_col], errors="coerce")
            filtered_data[puts_delta_col] = pd.to_numeric(filtered_data[puts_delta_col], errors="coerce")
            filtered_data[puts_iv_col] = pd.to_numeric(filtered_data[puts_iv_col], errors="coerce")

            # Drop NaNs
            filtered_calls = filtered_data.dropna(subset=[calls_delta_col, calls_iv_col])
            filtered_puts = filtered_data.dropna(subset=[puts_delta_col, puts_iv_col])

            # Debug: Print filtered calls and puts data
            print("Filtered Calls Data (First 5 Rows):")
            print(filtered_calls.head())
            print("Filtered Puts Data (First 5 Rows):")
            print(filtered_puts.head())

            # Clear plot
            self.plot_widget.clear()

            # Scatter plot for Calls
            self.plot_widget.plot(
                filtered_calls[calls_delta_col].values,
                filtered_calls[calls_iv_col].values,
                pen=None,
                symbol="o",
                symbolBrush="b",
                symbolSize=8,
                name="Calls IV",
            )

            # Scatter plot for Puts
            self.plot_widget.plot(
                filtered_puts[puts_delta_col].values,
                filtered_puts[puts_iv_col].values,
                pen=None,
                symbol="x",
                symbolBrush="r",
                symbolSize=8,
                name="Puts IV",
            )

            # Set axis labels
            self.plot_widget.getPlotItem().setLabel("bottom", text="Delta")
            self.plot_widget.getPlotItem().setLabel("left", text="Implied Volatility")

            # Combine filtered_calls and filtered_puts for table update
            combined_data = pd.DataFrame()

            # Dynamically add selected columns for calls
            if not filtered_calls.empty:
                combined_data[f"{calls_delta_col} (Calls)"] = filtered_calls[calls_delta_col]
                combined_data[f"{calls_iv_col} (Calls)"] = filtered_calls[calls_iv_col]

            # Dynamically add selected columns for puts
            if not filtered_puts.empty:
                combined_data[f"{puts_delta_col} (Puts)"] = filtered_puts[puts_delta_col]
                combined_data[f"{puts_iv_col} (Puts)"] = filtered_puts[puts_iv_col]

            # Debug: Print combined data for verification
            print("Combined Data for Table (First 5 Rows):")
            print(combined_data.head())

            # Update data table
            self.update_table(combined_data)

            self.label.setText("Scatter plot generated successfully.")
        except Exception as e:
            self.label.setText(f"Error plotting data: {e}")

    def update_table(self, filtered_data):
        """Update the data table with a fixed number of rows (100)."""
        MAX_ROWS = 100  # Fixed maximum row count

        # Replace NaN values with 0
        filtered_data = filtered_data.fillna(0)

        # Debug: Print data being passed to the table
        print("Updating Table with Data (First 5 Rows):")
        print(filtered_data.head())

        # Clear the table before updating
        self.data_table.clear()

        # Set fixed number of rows and dynamically set columns
        self.data_table.setRowCount(MAX_ROWS)
        self.data_table.setColumnCount(len(filtered_data.columns))
        self.data_table.setHorizontalHeaderLabels(filtered_data.columns.tolist())

        # Populate the table with the filtered data
        for row_idx in range(MAX_ROWS):
            if row_idx < len(filtered_data):
                # Fill with data if available
                for col_idx, value in enumerate(filtered_data.iloc[row_idx]):
                    item = QTableWidgetItem(str(value))
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)  # Make cells read-only
                    self.data_table.setItem(row_idx, col_idx, item)
            else:
                # Fill empty rows with blanks
                for col_idx in range(len(filtered_data.columns)):
                    item = QTableWidgetItem("")
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)  # Make cells read-only
                    self.data_table.setItem(row_idx, col_idx, item)

        # Debug: Confirm table update
        print(f"Table updated with {len(filtered_data)} rows and {len(filtered_data.columns)} columns.")

# Run the application
if __name__ == "__main__":
    app = QApplication([])
    window = DynamicFinancialVisualizer()
    window.show()
    app.exec_()
