import os

def replace_commas(value):
    return float(value.replace(',', '.'))

def check_distinct_quotetime_file_presence(folder_path = "E://OutputParamsFiles", file_name = "outputDistinctQuotesTimes.csv"):
    file_path = os.path.join(folder_path, file_name)

    # Check if the file exists
    if os.path.isfile(file_path):
        print(f"The file '{file_name}' exists in the folder '{folder_path}'.")
        return True
    else:
        print(f"The file '{file_name}' does not exist in the folder '{folder_path}'.")
        return False

def get_file_names(folder_path):
    """
    Returns a list of file names in the specified folder.

    Parameters:
        folder_path (str): The path to the folder.

    Returns:
        list: A list of file names.
    """
    # Check if the folder exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder '{folder_path}' does not exist.")

    # Get all file names in the folder
    file_names = [f for f in os.listdir(folder_path) 
                  if os.path.isfile(os.path.join(folder_path, f))]

    return file_names