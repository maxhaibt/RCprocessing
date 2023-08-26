import os
import pyproj
import tkinter as tk
import tkinter.messagebox
from pathlib import Path
from tkinter import filedialog
from orthoboxreader import read_rcorthobox

# Set the data directory for pyproj
pyproj.datadir.set_data_dir('C:/Users/mhaibt/Anaconda3/envs/orthobox_env/Library/share/proj')

def main():
    # Create a simple GUI window to select input files
    root = tk.Tk()
    root.withdraw()

    def open_file_browser():
        file_paths = filedialog.askopenfilenames(title="Select orthobox files", filetypes=(("Orthobox files", "*.rcortho"), ("All files", "*.*")))

        # Read and process the orthobox files
        if file_paths:
            diclist = [{'orthoboxfile': file_path} for file_path in file_paths]
            orthobox_gpdf = read_rcorthobox(diclist)

            # Save the output to the parent folder of the first input file
            output_folder = Path(file_paths[0]).parent
            output_name = output_folder.name + ".gpkg"
            orthobox_gpdf.to_file(output_folder / output_name, driver='GPKG')

            # Show a message box to notify that the operation has finished successfully
            tkinter.messagebox.showinfo("Success", "The operation has finished successfully.")
            
            # Close the root window
            root.destroy()

    root.after(0, open_file_browser)
    root.mainloop()

if __name__ == "__main__":
    main()


