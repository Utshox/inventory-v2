from dotenv import load_dotenv
load_dotenv() 

import pandas as pd  # Add this import
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, send_file, session, redirect, url_for, flash
from config import Config
from modules.data_processor import DataProcessor
from modules.ai_agent import AIAgent
import os
import uuid

app = Flask(__name__)
app.config.from_object(Config)
app.secret_key = app.config['SECRET_KEY']

processor = DataProcessor(app.config['UPLOAD_FOLDER'])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(f"{uuid.uuid4()}.csv")
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            try:
                processor.load_data(filename)
                session['current_file'] = filename
                flash('File uploaded successfully', 'success')
            except Exception as e:
                flash(str(e), 'error')
            return redirect(url_for('index'))
    
    sample_data = processor.get_sample_data()
    columns = processor.get_columns()
    return render_template('index.html', 
                         columns=columns,
                         sample_data=sample_data,
                         current_file=session.get('current_file'))

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/query', methods=['POST'])
def handle_query():
    try:
        if 'current_file' not in session:
            flash('❌ Please upload a CSV file first', 'error')
            return redirect(url_for('index'))

        if processor.df is None:
            flash('❌ Data loading failed. Please re-upload your CSV.', 'error')
            return redirect(url_for('index'))

        query = request.form.get('query', '').strip()
        if len(query) < 3:
            flash('❌ Query must be at least 3 characters', 'error')
            return redirect(url_for('index'))

        try:
            ai_agent = AIAgent(processor.df)
            response, table = ai_agent.process_query(query)
            
            # Add validation to break loops
            if "python_repl_ast" in response:
                raise ValueError("Invalid tool reference in response")

            # Convert potential results to session storage
            session['last_response'] = {
                'text': response,
                'table': table.to_dict('records') if isinstance(table, pd.DataFrame) else None
            }

            return redirect(url_for('show_results'))

        except Exception as e:
            flash(f'❌ Query processing failed: {str(e)}', 'error')
            return redirect(url_for('index'))

    except Exception as e:
        if "name 'pd' is not defined" in str(e):
            flash('❌ System configuration error - please contact support', 'error')
        else:
            flash(f'❌ Query failed: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/results')
def show_results():
    response_data = session.get('last_response', {})
    table_data = response_data.get('table', [])  # Default to empty list
    
    return render_template('results.html',
                         response_text=response_data.get('text'),
                         table_data=table_data)

@app.route('/test_api')
def test_api():
    from google.generativeai import configure, list_models
    
    configure(api_key=os.getenv('GOOGLE_API_KEY'))
    try:
        models = list_models()
        return f"API Connection Successful! Available models: {[m.name for m in models]}"
    except Exception as e:
        return f"API Error: {str(e)}"

 # Add new route for invoice generation
# Add new route for invoice generation
@app.route('/generate_invoice', methods=['POST'])
def generate_invoice():
    try:
        product_id = request.form['product_id']
        product_name = request.form['product_name']
        unit_price = float(request.form['unit_price'])
        quantity = int(request.form['quantity'])

        item = {
            'product_id': product_id,
            'product_name': product_name,
            'unit_price': unit_price,
            'quantity': quantity,
            'total': unit_price * quantity
        }

        if 'invoice_items' not in session:
            session['invoice_items'] = []
        
        # Update quantity if product exists
        existing = next((i for i in session['invoice_items'] if i['product_id'] == product_id), None)
        if existing:
            existing['quantity'] += quantity
            existing['total'] += item['total']
        else:
            session['invoice_items'].append(item)
        
        session.modified = True
        flash(f'{quantity} x {product_name} added to order!', 'success')
        return redirect(url_for('show_results'))

    except Exception as e:
        flash(f'Error adding item: {str(e)}', 'error')
        return redirect(url_for('show_results'))

@app.route('/download/invoice')
def download_invoice():
    try:
        from fpdf import FPDF
        from datetime import datetime
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        
        # Header
        pdf.cell(0, 10, 'Commercial Door Solutions - Invoice', 0, 1, 'C')
        pdf.ln(10)
        
        # Invoice Details
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f'Invoice Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1)
        pdf.cell(0, 10, f'Invoice Number: INV-{datetime.now().strftime("%Y%m%d%H%M%S")}', 0, 1)
        pdf.ln(15)
        
        # Items Table
        pdf.set_font('Arial', 'B', 12)
        col_widths = [30, 80, 25, 25, 30]
        headers = ['ID', 'Product', 'Price', 'Qty', 'Total']
        
        # Table Header
        for width, header in zip(col_widths, headers):
            pdf.cell(width, 10, header, border=1)
        pdf.ln()
        
        # Table Rows
        pdf.set_font('Arial', '', 12)
        grand_total = 0
        for item in session.get('invoice_items', []):
            pdf.cell(col_widths[0], 10, item['product_id'], border=1)
            pdf.cell(col_widths[1], 10, item['product_name'], border=1)
            pdf.cell(col_widths[2], 10, f"${item['unit_price']:.2f}", border=1)
            pdf.cell(col_widths[3], 10, str(item['quantity']), border=1)
            pdf.cell(col_widths[4], 10, f"${item['total']:.2f}", border=1)
            pdf.ln()
            grand_total += item['total']
        
        # Total Row
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(sum(col_widths[:4]), 10, 'Grand Total:', border=1, align='R')
        pdf.cell(col_widths[4], 10, f"${grand_total:.2f}", border=1)
        
        # Save and send
        filename = f"invoice_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
        pdf.output(filename)
        
        return send_file(filename, as_attachment=True)
        
    except Exception as e:
        flash(f'Error generating invoice: {str(e)}', 'error')
        return redirect(url_for('show_results'))

# @app.route('/download/invoice')
# def download_invoice():
#     # Add your invoice generation logic here
#     return send_file('path/to/invoice.pdf', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)