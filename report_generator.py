import matplotlib.pyplot as plt
import os
import fnmatch
from pylatex import Document, Section, Subsection, Itemize, Figure, Math, TikZ, Axis, Plot, Command, Package 
from pylatex.utils import NoEscape, bold

def generate_benchmark_plot(algorithm_name, test_case_sizes, test_case_times, output_filename='benchmark_plot.png'):
    """Generates a benchmark plot and saves it as a PNG file."""
    output_path = output_filename # Simplified for current context, adapt if needed for paths
    print(f"Attempting to generate plot: {output_path}") # DEBUG: Print plot output path
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(test_case_sizes, test_case_times, marker='o', linestyle='-')
        plt.title(f'{algorithm_name} Benchmark - Time vs Input Size')
        plt.xlabel('Input Size (Number of Elements)')
        plt.ylabel('Time (seconds)')
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()
        print(f"Plot successfully generated: {output_path}") # DEBUG: Confirmation message
    except Exception as e:
        print(f"Error generating plot: {output_path}") # DEBUG: Error message
        print(f"Exception details: {e}") # DEBUG: Print exception details


def generate_latex_report(algorithm_name, avg_time, description, input_details, test_case_times, output_filename="report", test_case_sizes=None):
    """Generates a LaTeX report for the sorting algorithm benchmark."""


    doc = Document('article')
    doc.documentclass = Command('documentclass', options=['12pt'], arguments=['article']) # Set font size

    doc.preamble.append(Command('title', f'{algorithm_name} Performance Report'))
    doc.preamble.append(Command('author', 'Sorting Algorithm Benchmark Script'))
    doc.preamble.append(Command('date', NoEscape(r'\today')))

    geometry_options = {"a4paper": True, "margin": "1in"} # Options as dictionary
    doc.packages.append(Package('geometry', options=geometry_options)) # Use Package class for geometry with options
    doc.packages.append(Package('graphicx')) # No options for graphicx
    doc.packages.append(Package('amsmath'))
    doc.packages.append(Package('amsfonts'))
    doc.packages.append(Package('amssymb'))
    doc.packages.append(Package('hyperref')) # For URLs and references
    doc.packages.append(Package('textcomp')) # For additional symbols
    doc.packages.append(Package('lastpage')) # To get the total number of pages
    #doc.packages.append(Package('lmodern')) # Latin Modern fonts

    doc.append(NoEscape(r'\maketitle'))

    with doc.create(Section('Algorithm Description')):
        doc.append(description)

    with doc.create(Section('Benchmark Results')):
        with doc.create(Subsection('General Performance')):
            doc.append(f"Average execution time over {len(test_case_times)} test cases: {avg_time:.6f} seconds.")
            if input_details:
                doc.append("\n\nInput Details:\n")
                with doc.create(Itemize()) as itemize:
                    for key, value in input_details.items():
                        itemize.add_item(f"{key}: {value}")

        if test_case_sizes and test_case_times:
            with doc.create(Subsection('Benchmark Plot')):
                plot_filename = f"{algorithm_name.replace(' ', '_').lower()}_{input_details.get('Data Type', 'unknown').lower()}_benchmark_plot.png" # Construct plot filename
                generate_benchmark_plot(algorithm_name, test_case_sizes, test_case_times, output_filename=plot_filename) # Generate plot image

                with doc.create(Figure(position='ht!')) as plot_figure: # Use 'ht!' for more flexible placement
                    plot_figure.add_image(plot_filename, width=NoEscape(r'0.8\textwidth'))
                    plot_figure.add_caption(f"Benchmark plot for {algorithm_name} showing execution time vs input size.")


    doc.append(NoEscape(r'\newpage')) # Start bibliography on a new page

    doc.append(NoEscape(r'\vspace*{\fill}')) # Push the following content to the bottom of the page
    doc.append(NoEscape(r'\footnotesize Generated on \today\ using a custom Python script.'))
    doc.append(NoEscape(r' \hfill Page \thepage\ of \pageref{LastPage}')) # Page number at the very bottom

    doc.generate_pdf(output_filename, clean_tex=True)

def generate_pdf_report(*args, **kwargs):
    """Wrapper function to generate PDF report using generate_latex_report and handle cleanup."""
    output_filename = kwargs.get('output_filename', "report")
    latex_filename = f"{output_filename}.tex"
    pdf_filename = f"{output_filename}.pdf"
    png_cleanup_pattern = "*_benchmark_plot.png"
    latex_cleanup_patterns = [
        f"{output_filename}.log",
        f"{output_filename}.aux",
        f"{output_filename}.out",
        f"{output_filename}.fls",
        f"{output_filename}.fdb_latexmk",
        f"{output_filename}.synctex.gz",
        "*_benchmark_plot.png", # Also cleanup PNG plots here for consistency
    ]


    generate_latex_report(*args, **kwargs)

    # Cleanup generated files - LaTeX temp files and PNG plots
    for pattern in latex_cleanup_patterns:
        files_to_clean = [f for f in os.listdir('.') if f == pattern or fnmatch.fnmatch(f, pattern)] # Match exact filename or pattern
        if files_to_clean:
            for file_to_clean in files_to_clean:
                try:
                    os.remove(file_to_clean)
                    print(f"Cleaned up: {file_to_clean}")
                except OSError as e:
                    print(f"Error cleaning up {file_to_clean}: {e}")
        else:
             print(f"No files to clean up matching pattern: {pattern}")


if __name__ == '__main__':
    # Example Usage (for testing report generation):
    algorithm_name = "SampleSort Algorithm"
    avg_time = 0.001234
    description = "This is a sample description of the SampleSort Algorithm. It's very fast and efficient... (etc.)"
    input_details = {
        "Data Type": "Numbers",
        "Input Size": "Varying",
        "Order": "Ascending",
        "Test Cases": 5
    }
    test_case_times = [0.001, 0.0012, 0.0011, 0.0013, 0.0015] # Example times in seconds
    test_case_sizes = [100, 200, 300, 400, 500]

    generate_pdf_report(algorithm_name, avg_time, description, input_details, test_case_times, output_filename="sample_report", test_case_sizes=test_case_sizes)
    print("\nSample PDF report generated: sample_report.pdf")