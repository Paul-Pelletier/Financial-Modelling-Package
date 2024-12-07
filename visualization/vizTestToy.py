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
import pyqtgraph as pg
import pandas as pd


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dynamic Financial Data Visualizer")
        self.setGeometry(100, 100, 1200, 800)

        # Splitter for the layout
        splitter = QSplitter()

        # Main layout
        self.central_widget = QWidget()
        self.setCentralWidget(splitter)

        # Left panel (Controls + Plot)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Right panel (Data Table)
        self.data_table = QTableWidget()
        self.data_table.setColumnCount(0)  # Will be set dynamically
        self.data_table.setRowCount(0)

        splitter.addWidget(left_panel)
        splitter.addWidget(self.data_table)
        splitter.setSizes([800, 400])  # Set initial sizes of panels

        # Label for instructions
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

        # Button to load data
        self.load_button = QPushButton("Load Data")
        self.load_button.clicked.connect(self.load_data)
        left_layout.addWidget(self.load_button)

        # Button to plot data
        self.plot_button = QPushButton("Plot Data")
        self.plot_button.clicked.connect(self.plot_data)
        left_layout.addWidget(self.plot_button)

        # Connect expiry dropdown change to the plot function
        self.expiry_dropdown.currentTextChanged.connect(self.plot_data)

        self.data = None  # Initialize data to None


    def load_data(self):
        # Open file dialog to select CSV
        file_path, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv)")
        if file_path:
            try:
                # Load the data with ";" as the separator
                self.data = pd.read_csv(file_path, sep=";")
                self.data.columns = self.data.columns.str.strip()  # Clean up column names

                # Replace commas with periods and convert to float for numeric columns
                for col in self.data.columns:
                    try:
                        self.data[col] = self.data[col].str.replace(",", ".").astype(float)
                    except (ValueError, AttributeError):
                        # If conversion fails, assume it's a non-numeric column and skip
                        continue

                # Debugging: Print column names
                print("Columns in the dataset:", self.data.columns)

                # Populate the dropdowns with column names
                self.expiry_dropdown.clear()
                self.calls_delta_dropdown.clear()
                self.calls_iv_dropdown.clear()
                self.puts_delta_dropdown.clear()
                self.puts_iv_dropdown.clear()

                # Add unique expiries to the expiry dropdown
                if 'EXPIRE_DATE' in self.data.columns:
                    unique_expiries = self.data['EXPIRE_DATE'].unique()
                    self.expiry_dropdown.addItems(map(str, unique_expiries))
                else:
                    self.label.setText("Error: Column 'EXPIRE_DATE' not found in the dataset.")
                    return

                # Populate dropdowns with column names
                column_list = self.data.columns.tolist()
                self.calls_delta_dropdown.addItems(column_list)
                self.calls_iv_dropdown.addItems(column_list)
                self.puts_delta_dropdown.addItems(column_list)
                self.puts_iv_dropdown.addItems(column_list)

                # Set default values
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


    def update_table(self, filtered_data):
        """Update the data table to display the filtered data."""
        try:
            # Ensure filtered_data is not empty
            if filtered_data.empty:
                self.data_table.clear()
                self.data_table.setRowCount(0)
                self.data_table.setColumnCount(0)
                self.data_table.setHorizontalHeaderLabels([])
                self.label.setText("No data to display in the table.")
                return

            # Debugging: Print the filtered data
            print("Filtered data for the table:")
            print(filtered_data)

            # Set the number of rows and columns
            self.data_table.setRowCount(len(filtered_data))
            self.data_table.setColumnCount(len(filtered_data.columns))

            # Set the column headers
            self.data_table.setHorizontalHeaderLabels(filtered_data.columns.tolist())

            # Populate the table with filtered data
            for row_idx, row in filtered_data.iterrows():
                for col_idx, value in enumerate(row):
                    self.data_table.setItem(row_idx, col_idx, QTableWidgetItem(str(value)))

            # Resize columns and rows to fit content
            self.data_table.resizeColumnsToContents()
            self.data_table.resizeRowsToContents()

            self.label.setText("Table updated successfully.")
        except Exception as e:
            self.label.setText(f"Error updating the table: {e}")
            print(f"Error updating table: {e}")


        # Clear existing table content
        self.data_table.clear()

        # Set the number of rows and columns
        self.data_table.setRowCount(len(filtered_data))
        self.data_table.setColumnCount(len(filtered_data.columns))

        # Set column headers
        self.data_table.setHorizontalHeaderLabels(filtered_data.columns)

        # Debugging: Print filtered data
        print("Filtered data for table:")
        print(filtered_data)

        # Populate the table with the filtered data
        for row_idx, row in filtered_data.iterrows():
            for col_idx, value in enumerate(row):
                self.data_table.setItem(row_idx, col_idx, QTableWidgetItem(str(value)))

        # Resize columns and rows to fit the content
        self.data_table.resizeColumnsToContents()
        self.data_table.resizeRowsToContents()


    def plot_data(self):
        if self.data is not None:
            try:
                # Get the selected expiry and filter data
                selected_expiry = self.expiry_dropdown.currentText()

                # Ensure that the EXPIRE_DATE column is treated as a string for matching
                self.data['EXPIRE_DATE'] = self.data['EXPIRE_DATE'].astype(str)

                # Filter the data for the selected expiry and create a copy
                filtered_data = self.data.loc[self.data['EXPIRE_DATE'] == selected_expiry].copy()

                # If no data exists for the selected expiry, clear the plot and display a message
                if filtered_data.empty:
                    self.plot_widget.clear()
                    self.label.setText(f"No data available for expiry '{selected_expiry}'.")
                    self.data_table.setRowCount(0)
                    return

                # Get selected columns for calls and puts
                calls_delta_col = self.calls_delta_dropdown.currentText()  # X-axis for Calls
                calls_iv_col = self.calls_iv_dropdown.currentText()        # Y-axis for Calls
                puts_delta_col = self.puts_delta_dropdown.currentText()    # X-axis for Puts
                puts_iv_col = self.puts_iv_dropdown.currentText()          # Y-axis for Puts

                # Ensure numeric data for plotting
                filtered_data.loc[:, calls_delta_col] = pd.to_numeric(filtered_data[calls_delta_col], errors="coerce")
                filtered_data.loc[:, calls_iv_col] = pd.to_numeric(filtered_data[calls_iv_col], errors="coerce")
                filtered_data.loc[:, puts_delta_col] = pd.to_numeric(filtered_data[puts_delta_col], errors="coerce")
                filtered_data.loc[:, puts_iv_col] = pd.to_numeric(filtered_data[puts_iv_col], errors="coerce")

                # Drop rows with NaN values for both calls and puts
                filtered_calls = filtered_data.dropna(subset=[calls_delta_col, calls_iv_col])
                filtered_puts = filtered_data.dropna(subset=[puts_delta_col, puts_iv_col])

                # Clear the plot
                self.plot_widget.clear()

                # Plot Calls IV
                self.plot_widget.plot(
                    filtered_calls[calls_delta_col].values,
                    filtered_calls[calls_iv_col].values,
                    pen=None,
                    symbol = 'o',
                    symbolBrush = 'b',
                    symbolSize = 8,
                    name="Calls IV",
                )

                # Plot Puts IV
                self.plot_widget.plot(
                    filtered_puts[puts_delta_col].values,
                    filtered_puts[puts_iv_col].values,
                    pen=None,
                    symbol = 'x',
                    symbolBrush = 'r',
                    symbolSize = 8,
                    name="Puts IV",
                )

                # Set X and Y labels
                self.plot_widget.getPlotItem().setLabel("bottom", text="Delta (Custom Selection)")
                self.plot_widget.getPlotItem().setLabel("left", text="Implied Volatility")

                # Automatically adjust the view to the data
                self.plot_widget.autoRange()

                # Update the data table to include relevant columns for both calls and puts
                self.update_table(filtered_data[[calls_delta_col, calls_iv_col, puts_delta_col, puts_iv_col]])

                self.label.setText(
                    f"Plotting IV: Calls ({calls_iv_col}, blue) vs {calls_delta_col} and Puts ({puts_iv_col}, red) vs {puts_delta_col}."
                )
            except Exception as e:
                self.label.setText(f"Error plotting data: {e}")
        else:
            self.label.setText("No data loaded. Please load a file first.")



app = QApplication([])
window = MainWindow()
window.show()
app.exec_()
