from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

inp = 'GA1code_2779150_2849240_2732664_2861128_python.py'
out = 'GA1code_2779150_2849240_2732664_2861128_python.pdf'

# Read the source code
with open(inp, 'r', encoding='utf-8') as f:
    lines = f.readlines()

pdf = canvas.Canvas(out, pagesize=letter)
width, height = letter

x_margin = 40
y = height - 40
line_height = 12

pdf.setFont("Courier", 9)  # monospace font suits code

for line in lines:
    line = line.rstrip('\n')
    if y < 40:  # start a new page when space runs out
        pdf.showPage()
        pdf.setFont("Courier", 9)
        y = height - 40
    pdf.drawString(x_margin, y, line)
    y -= line_height

pdf.save()
print("PDF is Saved Successfully")