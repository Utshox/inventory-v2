from dotenv import load_dotenv
load_dotenv() 

import pandas as pd
import os
import uuid
import tempfile
import shutil
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, send_file, session, redirect, url_for, flash
from config import Config
from modules.data_processor import DataProcessor
from modules.ai_agent import AIAgent

app = Flask(__name__)
app.config.from_object(Config)
Config.init_app(app)  # Initialize app configuration
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
        # Check for required data
        if 'current_file' not in session:
            flash('❌ Please upload a CSV file first', 'error')
            return redirect(url_for('index'))

        if processor.df is None:
            flash('❌ Data loading failed. Please re-upload your CSV.', 'error')
            return redirect(url_for('index'))
            
        # Check for API key
        api_key = session.get('api_key') or app.config.get('GOOGLE_API_KEY')
        if not api_key:
            flash('❌ Please provide a Google API key first', 'error')
            return redirect(url_for('index'))

        query = request.form.get('query', '').strip()
        if len(query) < 3:
            flash('❌ Query must be at least 3 characters', 'error')
            return redirect(url_for('index'))

        try:
            # Pass the API key to the AIAgent
            ai_agent = AIAgent(processor.df, api_key=api_key)
            response, table = ai_agent.process_query(query)
            
            # Add validation to break loops
            if "python_repl_ast" in response:
                raise ValueError("Invalid tool reference in response")

            # Convert potential results to session storage
            session['last_response'] = {
                'text': response,
                'table': table if table else None
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

@app.route('/initialize', methods=['POST'])
def initialize():
    """Handle API key initialization"""
    try:
        api_key = request.form.get('api_key')
        if not api_key:
            flash('API key is required', 'error')
            return redirect(url_for('index'))
            
        # Store API key in session (not directly in environment)
        session['api_key'] = api_key
        
        # Test the API key
        from google.generativeai import configure, list_models
        configure(api_key=api_key)
        models = list_models()
        
        flash('API key successfully verified!', 'success')
        return redirect(url_for('index'))
    except Exception as e:
        flash(f'Invalid API key: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/generate_invoice', methods=['POST'])
def generate_invoice():
    try:
        # Validate required input fields
        if not all(key in request.form for key in ['product_id', 'product_name', 'unit_price', 'quantity']):
            flash('Missing required product information', 'error')
            return redirect(url_for('show_results'))
            
        # Extract and validate input values
        product_id = request.form['product_id']
        product_name = request.form['product_name']
        
        try:
            unit_price_str = request.form['unit_price']
            # Handle various formats including NaN and empty strings
            unit_price = 0.0
            if unit_price_str and unit_price_str.lower() != 'nan':
                unit_price = float(unit_price_str)
            if unit_price < 0:
                unit_price = 0.0
        except ValueError:
            unit_price = 0.0
            
        try:
            quantity = int(request.form['quantity'])
            if quantity <= 0:
                quantity = 1
        except ValueError:
            quantity = 1

        # Find additional product details from the table data
        additional_details = {}
        table_data = session.get('last_response', {}).get('table', [])
        
        for row in table_data:
            if str(row.get('Product ID')) == str(product_id):
                # Extract additional details if they exist
                if 'Size/Dimensions' in row:
                    additional_details['dimensions'] = row['Size/Dimensions']
                if 'Manufacturer' in row:
                    additional_details['manufacturer'] = row['Manufacturer']
                if 'Material' in row:
                    additional_details['material'] = row['Material']
                if 'Category' in row:
                    additional_details['category'] = row['Category']
                if 'Subcategory' in row:
                    additional_details['subcategory'] = row['Subcategory']
                if 'Warranty Information' in row:
                    additional_details['warranty'] = row['Warranty Information']
                break

        # Create invoice item with additional details
        item = {
            'product_id': product_id,
            'product_name': product_name,
            'unit_price': unit_price,
            'quantity': quantity,
            'total': unit_price * quantity,
            **additional_details  # Include all additional details
        }

        if 'invoice_items' not in session:
            session['invoice_items'] = []
        
        # Update quantity if product exists
        existing = next((i for i in session['invoice_items'] if i['product_id'] == product_id), None)
        if existing:
            existing['quantity'] += quantity
            existing['total'] = existing['unit_price'] * existing['quantity']  # Recalculate to prevent rounding errors
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
        import os
        
        # Create a temporary directory for generated files
        os.makedirs('generated', exist_ok=True)
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        
        # Header
        pdf.cell(0, 10, 'Commercial Door Solutions - Invoice', 0, 1, 'C')
        pdf.ln(10)
        
        # Invoice Details
        invoice_number = datetime.now().strftime("%Y%m%d%H%M%S")
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f'Invoice Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1)
        pdf.cell(0, 10, f'Invoice Number: INV-{invoice_number}', 0, 1)
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
        
        # Validate that we have invoice items
        invoice_items = session.get('invoice_items', [])
        if not invoice_items:
            flash('No items in invoice', 'error')
            return redirect(url_for('show_results'))
            
        for item in invoice_items:
            # Input validation
            if not all(k in item for k in ['product_id', 'product_name', 'unit_price', 'quantity', 'total']):
                continue
                
            pdf.cell(col_widths[0], 10, str(item['product_id']), border=1)
            pdf.cell(col_widths[1], 10, str(item['product_name'])[:30], border=1) # Limit text length
            pdf.cell(col_widths[2], 10, f"${float(item['unit_price']):.2f}", border=1)
            pdf.cell(col_widths[3], 10, str(int(item['quantity'])), border=1)
            pdf.cell(col_widths[4], 10, f"${float(item['total']):.2f}", border=1)
            pdf.ln()
            grand_total += float(item['total'])
        
        # Total Row
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(sum(col_widths[:4]), 10, 'Grand Total:', border=1, align='R')
        pdf.cell(col_widths[4], 10, f"${grand_total:.2f}", border=1)
        
        # Save and send
        filename = os.path.join('generated', f"invoice_{invoice_number}.pdf")
        pdf.output(filename)
        
        # Clear invoice items after generating
        session['invoice_items'] = []
        session.modified = True
        
        return send_file(filename, as_attachment=True)
        
    except Exception as e:
        flash(f'Error generating invoice: {str(e)}', 'error')
        return redirect(url_for('show_results'))

@app.route('/view_cart')
def view_cart():
    """View current invoice items"""
    invoice_items = session.get('invoice_items', [])
    
    # Ensure all numeric values are valid
    for item in invoice_items:
        if isinstance(item, dict):
            # Fix any NaN or invalid values
            if not isinstance(item.get('unit_price'), (int, float)) or pd.isna(item.get('unit_price')):
                item['unit_price'] = 0.0
            if not isinstance(item.get('quantity'), (int, float)) or pd.isna(item.get('quantity')):
                item['quantity'] = 1
            # Ensure total is recalculated correctly
            item['total'] = float(item['unit_price']) * float(item['quantity'])
    
    # Calculate grand total
    grand_total = sum(float(item['total']) for item in invoice_items if isinstance(item, dict))
    
    return render_template('invoice.html',
                          invoice_items=invoice_items,
                          grand_total=grand_total)

@app.route('/clear_cart')
def clear_cart():
    """Clear all invoice items"""
    session['invoice_items'] = []
    session.modified = True
    flash('Invoice items cleared', 'success')
    return redirect(url_for('show_results'))

@app.route('/remove_item/<product_id>')
def remove_item(product_id):
    """Remove a specific item from the invoice"""
    invoice_items = session.get('invoice_items', [])
    session['invoice_items'] = [item for item in invoice_items if item['product_id'] != product_id]
    session.modified = True
    flash('Item removed from cart', 'success')
    return redirect(url_for('view_cart'))

@app.route('/update_quantity', methods=['POST'])
def update_quantity():
    """Update the quantity of an item in the cart"""
    try:
        product_id = request.form.get('product_id')
        new_quantity = int(request.form.get('quantity', 1))
        
        if new_quantity <= 0:
            new_quantity = 1
            
        # Find and update the item
        invoice_items = session.get('invoice_items', [])
        item = next((i for i in invoice_items if i['product_id'] == product_id), None)
        
        if item:
            item['quantity'] = new_quantity
            # Ensure unit_price is a valid float
            if not isinstance(item['unit_price'], (int, float)) or pd.isna(item['unit_price']):
                item['unit_price'] = 0.0
            item['total'] = float(item['unit_price']) * new_quantity
            session.modified = True
            flash('Quantity updated successfully', 'success')
        else:
            flash('Item not found in cart', 'error')
            
        return redirect(url_for('view_cart'))
        
    except ValueError:
        flash('Invalid quantity value', 'error')
        return redirect(url_for('view_cart'))
    except Exception as e:
        flash(f'Error updating quantity: {str(e)}', 'error')
        return redirect(url_for('view_cart'))

@app.route('/customer_details')
def customer_details():
    """Show customer details form before generating invoice"""
    # Verify we have items in the cart
    invoice_items = session.get('invoice_items', [])
    if not invoice_items:
        flash('Your cart is empty. Please add items before checkout.', 'error')
        return redirect(url_for('show_results'))
    
    # Calculate grand total with valid numbers
    for item in invoice_items:
        # Ensure prices are valid
        if not isinstance(item.get('unit_price'), (int, float)) or pd.isna(item.get('unit_price')):
            item['unit_price'] = 0.0
        if not isinstance(item.get('quantity'), (int, float)) or pd.isna(item.get('quantity')):
            item['quantity'] = 1
        item['total'] = float(item['unit_price']) * int(item['quantity'])
    
    grand_total = sum(float(item['total']) for item in invoice_items)
    
    return render_template('customer_form.html',
                          invoice_items=invoice_items,
                          grand_total=grand_total)

@app.route('/generate_final_invoice', methods=['POST'])
def generate_final_invoice():
    """Generate invoice PDF with customer details"""
    try:
        # Get customer data from form
        customer_data = {
            'name': request.form.get('customer_name', ''),
            'email': request.form.get('customer_email', ''),
            'phone': request.form.get('customer_phone', ''),
            'address': request.form.get('customer_address', ''),
            'notes': request.form.get('notes', '')
        }
        
        # Verify we have items in the cart
        invoice_items = session.get('invoice_items', [])
        if not invoice_items:
            flash('Your cart is empty. Please add items before checkout.', 'error')
            return redirect(url_for('show_results'))
        
        from fpdf import FPDF
        from datetime import datetime
        import os
        
        # Create a temporary directory for generated files
        os.makedirs('generated', exist_ok=True)
        
        pdf = FPDF()
        pdf.add_page()
        
        # Header
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Commercial Door Solutions - Invoice', 0, 1, 'C')
        pdf.ln(5)
        
        # Company info
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 6, 'Door Solutions Inc.', 0, 1, 'L')
        pdf.cell(0, 6, '123 Security Lane', 0, 1, 'L')
        pdf.cell(0, 6, 'New York, NY 10001', 0, 1, 'L')
        pdf.cell(0, 6, 'Phone: (555) 123-4567', 0, 1, 'L')
        pdf.ln(5)
        
        # Customer Info
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 6, 'Bill To:', 0, 1, 'L')
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 6, customer_data['name'], 0, 1, 'L')
        pdf.cell(0, 6, customer_data['address'], 0, 1, 'L')
        pdf.cell(0, 6, f"Email: {customer_data['email']}", 0, 1, 'L')
        pdf.cell(0, 6, f"Phone: {customer_data['phone']}", 0, 1, 'L')
        pdf.ln(5)
        
        # Invoice Details
        invoice_number = datetime.now().strftime("%Y%m%d%H%M%S")
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 6, f'Invoice Date: {datetime.now().strftime("%Y-%m-%d")}', 0, 1, 'L')
        pdf.cell(0, 6, f'Invoice Number: INV-{invoice_number}', 0, 1, 'L')
        pdf.ln(10)
        
        # Items Table
        pdf.set_font('Arial', 'B', 10)
        col_widths = [20, 75, 30, 20, 30]
        headers = ['ID', 'Product', 'Price', 'Qty', 'Total']
        
        # Table Header
        for width, header in zip(col_widths, headers):
            pdf.cell(width, 10, header, border=1)
        pdf.ln()
        
        # Table Rows
        pdf.set_font('Arial', '', 9)
        subtotal = 0
        
        for item in invoice_items:
            # Ensure values are valid
            if not isinstance(item.get('unit_price'), (int, float)) or pd.isna(item.get('unit_price')):
                item['unit_price'] = 0.0
            if not isinstance(item.get('quantity'), (int, float)) or pd.isna(item.get('quantity')):
                item['quantity'] = 1
                
            # Calculate item total
            item_total = float(item['unit_price']) * int(item['quantity'])
                
            pdf.cell(col_widths[0], 10, str(item['product_id']), border=1)
            pdf.cell(col_widths[1], 10, str(item['product_name'])[:40], border=1) # Limit text length
            pdf.cell(col_widths[2], 10, f"${float(item['unit_price']):.2f}", border=1)
            pdf.cell(col_widths[3], 10, str(int(item['quantity'])), border=1)
            pdf.cell(col_widths[4], 10, f"${item_total:.2f}", border=1)
            pdf.ln()
            subtotal += item_total
        
        # Tax and Total
        tax = subtotal * 0.07  # 7% tax
        grand_total = subtotal + tax
        
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(sum(col_widths[:4]), 10, 'Subtotal:', border=1, align='R')
        pdf.cell(col_widths[4], 10, f"${subtotal:.2f}", border=1)
        pdf.ln()
        
        pdf.cell(sum(col_widths[:4]), 10, 'Tax (7%):', border=1, align='R')
        pdf.cell(col_widths[4], 10, f"${tax:.2f}", border=1)
        pdf.ln()
        
        pdf.cell(sum(col_widths[:4]), 10, 'Grand Total:', border=1, align='R')
        pdf.cell(col_widths[4], 10, f"${grand_total:.2f}", border=1)
        pdf.ln(15)
        
        # Notes
        if customer_data['notes']:
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(0, 6, 'Notes:', 0, 1)
            pdf.set_font('Arial', '', 10)
            pdf.multi_cell(0, 6, customer_data['notes'])
        
        # Save and send
        filename = os.path.join('generated', f"invoice_{invoice_number}.pdf")
        pdf.output(filename)
        
        # Clear invoice items after generating
        session['invoice_items'] = []
        session.modified = True
        
        return send_file(filename, as_attachment=True)
        
    except Exception as e:
        flash(f'Error generating invoice: {str(e)}', 'error')
        return redirect(url_for('view_cart'))

if __name__ == '__main__':
    app.run(debug=True)