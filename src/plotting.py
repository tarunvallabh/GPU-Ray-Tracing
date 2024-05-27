import sys
import numpy as np
import matplotlib.pyplot as plt

def main(filename, rows, data_type):
    # Convert rows to an integer
    rows = int(rows)
    cols = rows  # Assuming cols is the same as rows

    # Determine the data type based on the command line argument
    if data_type == "single":
        dtype = np.float32
    elif data_type == "double":
        dtype = np.float64
    else:
        print("Data type must be 'single' or 'double'.")
        sys.exit(1)

    # Load data from the binary file
    data = np.fromfile(filename, dtype=dtype).reshape((rows, cols))

    # Plot the data
    plt.imshow(data, cmap="gray", interpolation="nearest")
    plt.colorbar()  # Add a color bar to show the scale
    plt.title("Ray Tracing Results")
    plt.axis("off")  # Hide axes
    
    # Construct output filename
    output_filename = f"{filename.split('.')[0]}.png"
    plt.savefig(output_filename, dpi=300)  # Save the figure to a file
    
    # Close the plot to free memory
    plt.close()
    print(f"Plot saved as {output_filename}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <filename> <rows> <single/double>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
