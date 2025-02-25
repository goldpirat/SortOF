from fpdf import FPDF

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Sorting Algorithm Report', ln=True, align='C')
        self.ln(10)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

def generate_pdf_report(algorithm_name, time_taken, description, output_filename="report.pdf"):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.cell(0, 10, f"Algorithm: {algorithm_name}", ln=True)
    pdf.cell(0, 10, f"Average Time Taken: {time_taken:.6f} seconds", ln=True)
    pdf.ln(10)
    pdf.multi_cell(0, 10, description)
    
    pdf.output(output_filename)
    print(f"PDF report generated: {output_filename}")
