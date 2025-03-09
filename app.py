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
    table_data = response_data.get('table')
    
    # Convert all values to strings for template safety
    if table_data:
        table_data = [
            {k: str(v) if not isinstance(v, (int, float)) else v 
             for k, v in item.items()}
            for item in table_data
        ]

    return render_template('results.html',
                         response_text=response_data.get('text'),
                         table_data=table_data)

@app.route('/download/invoice')
def download_invoice():
    # Add your invoice generation logic here
    return send_file('path/to/invoice.pdf', as_attachment=True)

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
        product_id = request.form.get('product_id')
        quantity = int(request.form.get('quantity'))
        
        # Get current data
        current_file = session.get('current_file')
        processor.load_data(current_file)
        
        # Find product in dataframe
        product = processor.df[processor.df['Product ID'] == product_id].iloc[0].to_dict()
        
        # Add to invoice session
        if 'invoice_items' not in session:
            session['invoice_items'] = []
            
        session['invoice_items'].append({
            'product_id': product_id,
            'name': product['Product Name'],
            'price': product['Unit Price'],
            'quantity': quantity
        })
        
        flash('Item added to invoice!', 'success')
        return redirect(url_for('show_results'))
    
    except Exception as e:
        flash(f'Error adding item: {str(e)}', 'error')
        return redirect(url_for('show_results'))

    # Update download invoice route
    def download_invoice():
        try:
            from fpdf import FPDF
            import datetime
            
            # Create PDF
            pdf = FPDF()
            pdf.add_page()
            
            # Add invoice content
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, 'Commercial Door Solutions - Invoice', 0, 1, 'C')
            
            # Add items
            pdf.set_font('Arial', '', 12)
            if 'invoice_items' in session:
                for item in session['invoice_items']:
                    pdf.cell(0, 10, f"{item['name']} (Qty: {item['quantity']}) - ${item['price'] * item['quantity']:.2f}", 0, 1)
            
            # Add total
            total = sum(item['price'] * item['quantity'] for item in session.get('invoice_items', []))
            pdf.cell(0, 10, f"Total: ${total:.2f}", 0, 1)
            
            # Save and send
            filename = f"invoice_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
            pdf.output(filename)
            
            return send_file(filename, as_attachment=True)
            
        except Exception as e:
            flash(f'Error generating invoice: {str(e)}', 'error')
            return redirect(url_for('show_results'))

if __name__ == '__main__':
    app.run(debug=True)