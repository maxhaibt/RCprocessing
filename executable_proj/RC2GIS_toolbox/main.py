import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import toolbox



def main():
    root = tk.Tk()
    root.geometry("300x200")  # Set the size of the window
    root.title("RC2GIS")  # Set the title of the window

    # Create a button for each tool
    btn_Orthobox2GIS = tk.Button(root, text="Orthobox2GIS", command=lambda: toolbox.open_file_browser_Orthobox2GIS(root))
    btn_Orthobox2GIS.pack(pady=20)  # Add some padding around the button



    btn_hugeTiffs2GIS = tk.Button(root, text="sideviewTiffs2GIS", command=lambda: toolbox.open_file_browser_sideviewTiffs2GIS(root))
    btn_hugeTiffs2GIS.pack(pady=20)  # Add some padding around the button

    btn_hugeTiffs2GIS_dem = tk.Button(root, text="hugeTiffs2GIS - DEM",
                                  command=lambda: toolbox.open_file_browser_hugeTiffs2GIS_dem(root))
    btn_hugeTiffs2GIS_dem.pack(pady=10)

    btn_hugeTiffs2GIS_true = tk.Button(root, text="hugeTiffs2GIS - Truecolor",
                                    command=lambda: toolbox.open_file_browser_hugeTiffs2GIS_truecolor(root))
    btn_hugeTiffs2GIS_true.pack(pady=10)

    root.mainloop()
print("Running main.py")
if __name__ == "__main__":
    print("Running main.py")
    main()