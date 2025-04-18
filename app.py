from dotenv import load_dotenv
load_dotenv() 

import pandas as pd
import numpy as np  # Explicitly import numpy
import os
import uuid
import tempfile
import shutil
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, send_file, session, redirect, url_for, flash, jsonify
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from config import Config
from modules.data_processor import DataProcessor
from modules.ai_agent import AIAgent
from modules.models import db, User, Product, Customer, Invoice, InvoiceItem, ApiKey
from modules.forms import LoginForm, RegistrationForm, ProductForm, CustomerForm, SearchForm, ApiKeyForm
from modules.api import api as api_blueprint  # Import the API blueprint
from datetime import datetime
import secrets
from sqlalchemy import or_, func, desc, asc
import json

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)
Config.init_app(app)  # Initialize app configuration
app.secret_key = app.config['SECRET_KEY']

# Initialize database
db.init_app(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Register blueprints
app.register_blueprint(api_blueprint, url_prefix='/api/v1')  # Add API routes under /api/v1

# Initialize data processor for CSV import
processor = DataProcessor(app.config['UPLOAD_FOLDER'])

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Create database tables
def create_tables():
    db.create_all()
    # Create admin user if it doesn't exist
    if not User.query.filter_by(username='admin').first():
        admin = User(username='admin', email='admin@example.com', role='admin')
        admin.set_password('adminpassword')
        db.session.add(admin)
        db.session.commit()

# Call create_tables with app context
with app.app_context():
    create_tables()

# NOTE: The @app.before_first_request decorator is removed in Flask 2.0+
# If you need code to run before the first request, use the with app.app_context() pattern above
# or create an init_app function that's called when the app is created

# Authentication routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password', 'error')
            return redirect(url_for('login'))
        login_user(user, remember=form.remember_me.data)
        user.last_login = datetime.utcnow()
        db.session.commit()
        next_page = request.args.get('next')
        if not next_page or not next_page.startswith('/'):
            next_page = url_for('index')
        return redirect(next_page)
    return render_template('login.html', title='Sign In', form=form)

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Congratulations, you are now registered!', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)

# Home page route
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

# Import data to database from CSV
@app.route('/import_to_db')
@login_required
def import_to_db():
    if 'current_file' not in session or processor.df is None:
        flash('No CSV file loaded', 'error')
        return redirect(url_for('index'))
    
    try:
        counter = 0
        for _, row in processor.df.iterrows():
            # Skip if no product ID or price
            if 'Product ID' not in row or 'Unit Price' not in row or pd.isna(row['Product ID']) or pd.isna(row['Unit Price']):
                continue
                
            product_id = str(row['Product ID'])
            
            # Check if product already exists
            existing_product = Product.query.filter_by(product_id=product_id).first()
            if existing_product:
                continue
                
            # Create new product
            product = Product(
                product_id=product_id,
                name=row['Product Name'] if 'Product Name' in row else 'Unknown',
                price=float(row['Unit Price']) if pd.notna(row['Unit Price']) else 0.0,
                manufacturer=row['Manufacturer'] if 'Manufacturer' in row and pd.notna(row['Manufacturer']) else None,
                category=row['Category'] if 'Category' in row and pd.notna(row['Category']) else None,
                subcategory=row['Subcategory'] if 'Subcategory' in row and pd.notna(row['Subcategory']) else None,
                material=row['Material'] if 'Material' in row and pd.notna(row['Material']) else None,
                dimensions=row['Size/Dimensions'] if 'Size/Dimensions' in row and pd.notna(row['Size/Dimensions']) else None,
                warranty=int(row['Warranty Information']) if 'Warranty Information' in row and pd.notna(row['Warranty Information']) else None,
                stock_quantity=20  # Default stock
            )
            db.session.add(product)
            counter += 1
        
        db.session.commit()
        flash(f'Successfully imported {counter} products to database', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error importing to database: {str(e)}', 'error')
    
    return redirect(url_for('products'))

# Product management routes
@app.route('/products')
@login_required
def products():
    page = request.args.get('page', 1, type=int)
    products = Product.query.paginate(page=page, per_page=10)
    return render_template('products.html', title='Products', products=products)

@app.route('/product/new', methods=['GET', 'POST'])
@login_required
def new_product():
    form = ProductForm()
    if form.validate_on_submit():
        product = Product(
            product_id=form.product_id.data,
            name=form.name.data,
            price=form.price.data,
            manufacturer=form.manufacturer.data,
            category=form.category.data, 
            subcategory=form.subcategory.data,
            material=form.material.data,
            dimensions=form.dimensions.data,
            warranty=form.warranty.data,
            description=form.description.data,
            stock_quantity=form.stock_quantity.data or 0
        )
        db.session.add(product)
        db.session.commit()
        flash('Product added successfully!', 'success')
        return redirect(url_for('products'))
    return render_template('product_form.html', title='New Product', form=form)

@app.route('/product/<string:product_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_product(product_id):
    product = Product.query.filter_by(product_id=product_id).first_or_404()
    form = ProductForm(obj=product)
    if form.validate_on_submit():
        product.product_id = form.product_id.data
        product.name = form.name.data
        product.price = form.price.data
        product.manufacturer = form.manufacturer.data
        product.category = form.category.data
        product.subcategory = form.subcategory.data
        product.material = form.material.data
        product.dimensions = form.dimensions.data
        product.warranty = form.warranty.data
        product.description = form.description.data
        product.stock_quantity = form.stock_quantity.data
        db.session.commit()
        flash('Product updated successfully!', 'success')
        return redirect(url_for('products'))
    return render_template('product_form.html', title='Edit Product', form=form)

@app.route('/product/<string:product_id>/delete', methods=['POST'])
@login_required
def delete_product(product_id):
    product = Product.query.filter_by(product_id=product_id).first_or_404()
    db.session.delete(product)
    db.session.commit()
    flash('Product deleted successfully!', 'success')
    return redirect(url_for('products'))

# Customer management routes
@app.route('/customers')
@login_required
def customers():
    page = request.args.get('page', 1, type=int)
    customers = Customer.query.paginate(page=page, per_page=10)
    return render_template('customers.html', title='Customers', customers=customers)

@app.route('/customer/new', methods=['GET', 'POST'])
@login_required
def new_customer():
    form = CustomerForm()
    if form.validate_on_submit():
        customer = Customer(
            name=form.name.data,
            email=form.email.data,
            phone=form.phone.data,
            address=form.address.data
        )
        db.session.add(customer)
        db.session.commit()
        flash('Customer added successfully!', 'success')
        return redirect(url_for('customers'))
    return render_template('customer_form.html', title='New Customer', form=form)

@app.route('/customer/<int:id>/edit', methods=['GET', 'POST'])
@login_required
def edit_customer(id):
    customer = Customer.query.get_or_404(id)
    form = CustomerForm(obj=customer)
    if form.validate_on_submit():
        customer.name = form.name.data
        customer.email = form.email.data
        customer.phone = form.phone.data
        customer.address = form.address.data
        db.session.commit()
        flash('Customer updated successfully!', 'success')
        return redirect(url_for('customers'))
    return render_template('customer_form.html', title='Edit Customer', form=form)

@app.route('/customer/<int:id>/delete', methods=['POST'])
@login_required
def delete_customer(id):
    customer = Customer.query.get_or_404(id)
    db.session.delete(customer)
    db.session.commit()
    flash('Customer deleted successfully!', 'success')
    return redirect(url_for('customers'))

# API Key management
@app.route('/api_keys', methods=['GET', 'POST'])
@login_required
def api_keys():
    form = ApiKeyForm()
    if form.validate_on_submit():
        # Generate a secure API key
        key_value = secrets.token_hex(16)
        api_key = ApiKey(
            user_id=current_user.id,
            key_name=form.key_name.data,
            key_value=key_value
        )
        db.session.add(api_key)
        db.session.commit()
        flash(f'API Key generated: {key_value}', 'success')
        return redirect(url_for('api_keys'))
    
    keys = ApiKey.query.filter_by(user_id=current_user.id).all()
    return render_template('api_keys.html', title='API Keys', form=form, keys=keys)

@app.route('/api_key/<int:id>/delete', methods=['POST'])
@login_required
def delete_api_key(id):
    key = ApiKey.query.get_or_404(id)
    if key.user_id != current_user.id:
        flash('Permission denied', 'error')
        return redirect(url_for('api_keys'))
    
    db.session.delete(key)
    db.session.commit()
    flash('API Key deleted successfully', 'success')
    return redirect(url_for('api_keys'))

# Keep the existing query, cart, and invoice routes below
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
        
        # Handle unit price with proper validation
        try:
            unit_price_str = request.form['unit_price']
            # Handle various formats including NaN, nan, empty strings, and None values
            unit_price = 0.0
            if unit_price_str and unit_price_str.lower() not in ['nan', 'none', 'undefined', 'null', 'n/a']:
                try:
                    unit_price = float(unit_price_str)
                    # Check if it's a valid number (not NaN or infinite)
                    if pd.isna(unit_price) or not np.isfinite(unit_price):
                        unit_price = 0.0
                except (ValueError, TypeError):
                    unit_price = 0.0
            if unit_price < 0:
                unit_price = 0.0
        except (ValueError, TypeError):
            unit_price = 0.0
            
        # Handle quantity with validation
        try:
            quantity = int(request.form['quantity'])
            if quantity <= 0:
                quantity = 1
        except (ValueError, TypeError):
            quantity = 1

        # Look up product in database to get additional details
        product_details = {}
        try:
            product = Product.query.filter_by(product_id=str(product_id)).first()
            if product:
                product_details = {
                    'manufacturer': product.manufacturer,
                    'category': product.category,
                    'subcategory': product.subcategory,
                    'material': product.material,
                    'dimensions': product.dimensions,
                    'warranty': product.warranty,
                    'stock_quantity': product.stock_quantity,
                    'description': product.description
                }
        except Exception as e:
            app.logger.warning(f"Could not load product details: {str(e)}")
            
        # Add the item to the cart using Cart class with additional details
        from modules.utils import Cart
        success = Cart.add_item(product_id, product_name, unit_price, quantity, **product_details)
        
        if success:
            flash(f'{quantity} x {product_name} added to order!', 'success')
        else:
            flash('Failed to add item to cart', 'error')
            
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
        from modules.utils import Cart
        
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
        
        # Get cart items using Cart class
        cart_items = Cart.get_items()
        if not cart_items:
            flash('No items in invoice', 'error')
            return redirect(url_for('show_results'))
            
        # Calculate grand total using Cart class
        grand_total = Cart.get_total()
            
        for item in cart_items:
            # Input validation
            if not all(k in item for k in ['product_id', 'product_name', 'unit_price', 'quantity', 'total']):
                continue
                
            pdf.cell(col_widths[0], 10, str(item['product_id']), border=1)
            pdf.cell(col_widths[1], 10, str(item['product_name'])[:30], border=1) # Limit text length
            pdf.cell(col_widths[2], 10, f"${float(item['unit_price']):.2f}", border=1)
            pdf.cell(col_widths[3], 10, str(int(item['quantity'])), border=1)
            pdf.cell(col_widths[4], 10, f"${float(item['total']):.2f}", border=1)
            pdf.ln()
        
        # Total Row
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(sum(col_widths[:4]), 10, 'Grand Total:', border=1, align='R')
        pdf.cell(col_widths[4], 10, f"${grand_total:.2f}", border=1)
        
        # Save and send
        filename = os.path.join('generated', f"invoice_{invoice_number}.pdf")
        pdf.output(filename)
        
        # Clear cart after generating invoice
        Cart.clear()
        
        return send_file(filename, as_attachment=True)
        
    except Exception as e:
        flash(f'Error generating invoice: {str(e)}', 'error')
        return redirect(url_for('show_results'))

@app.route('/view_cart')
def view_cart():
    """View current invoice items"""
    from modules.utils import Cart
    
    # Get all items in the cart
    cart_items = Cart.get_items()
    
    # Calculate grand total
    grand_total = Cart.get_total()
    
    return render_template('invoice.html',
                          invoice_items=cart_items,
                          grand_total=grand_total)

@app.route('/clear_cart')
def clear_cart():
    """Clear all invoice items"""
    from modules.utils import Cart
    Cart.clear()
    flash('Invoice items cleared', 'success')
    return redirect(url_for('show_results'))

@app.route('/remove_item/<product_id>')
def remove_item(product_id):
    """Remove a specific item from the invoice"""
    from modules.utils import Cart
    Cart.remove_item(product_id)
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
            
        # Update quantity using Cart class
        from modules.utils import Cart
        Cart.update_quantity(product_id, new_quantity)
        flash('Quantity updated successfully', 'success')
            
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
    # Verify we have items in the cart using Cart class
    from modules.utils import Cart
    
    # Get all items in the cart
    invoice_items = Cart.get_items()
    
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
    
    return render_template('checkout.html',
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
        
        # Verify we have items in the cart using Cart class
        from modules.utils import Cart
        invoice_items = Cart.get_items()
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
        
        # Clear cart after generating invoice
        Cart.clear()
        
        return send_file(filename, as_attachment=True)
        
    except Exception as e:
        flash(f'Error generating invoice: {str(e)}', 'error')
        return redirect(url_for('view_cart'))

@app.route('/dashboard')
@login_required
def dashboard():
    """Display analytical dashboard with inventory statistics"""
    # General statistics
    stats = {
        'total_products': Product.query.count(),
        'total_customers': Customer.query.count(),
        'total_sales': Invoice.query.with_entities(db.func.sum(Invoice.total)).scalar() or 0.0,
        'low_stock': Product.query.filter(Product.stock_quantity < 5).count()
    }
    
    # Recent products
    recent_products = Product.query.order_by(Product.date_added.desc()).limit(5).all()
    
    # Low stock items
    low_stock_items = Product.query.filter(Product.stock_quantity < 5).order_by(Product.stock_quantity).limit(10).all()
    
    # Category distribution for chart
    category_counts = db.session.query(
        Product.category, 
        db.func.count(Product.id)
    ).group_by(Product.category).all()
    
    # Handle None category
    category_counts = [(cat or 'Uncategorized', count) for cat, count in category_counts]
    
    # Prepare data for chart
    category_labels = [cat for cat, _ in category_counts]
    category_data = [count for _, count in category_counts]
    
    # Mocked recent activities (in real app, this would come from Activity model)
    recent_activities = [
        {
            'action': 'Product Added',
            'description': 'New product "Steel Security Door" was added to inventory',
            'user': 'admin',
            'timestamp': '2 hours ago'
        },
        {
            'action': 'Invoice Generated',
            'description': 'Invoice #INV-20250417001 was generated for customer "Smith Construction"',
            'user': 'admin',
            'timestamp': '5 hours ago'
        },
        {
            'action': 'Stock Updated',
            'description': 'Stock level for "Aluminum Glass Door" was updated from 15 to 8',
            'user': 'admin',
            'timestamp': '1 day ago'
        },
        {
            'action': 'Customer Added',
            'description': 'New customer "ABC Builders" was added to the system',
            'user': 'admin',
            'timestamp': '2 days ago'
        }
    ]
    
    return render_template(
        'dashboard.html',
        stats=stats,
        recent_products=recent_products,
        low_stock_items=low_stock_items,
        category_labels=category_labels,
        category_data=category_data,
        recent_activities=recent_activities
    )

@app.route('/api_docs')
def api_docs():
    """Display API documentation"""
    return render_template('api_docs.html', title='API Documentation')

@app.route('/search', methods=['GET', 'POST'])
@login_required
def search():
    """Search for products with multiple criteria"""
    search_term = request.args.get('query', '')
    category = request.args.get('category', '')
    manufacturer = request.args.get('manufacturer', '')
    min_price = request.args.get('min_price', '')
    max_price = request.args.get('max_price', '')
    find_matching = request.args.get('find_matching', '')
    match_threshold = request.args.get('match_threshold', '2')  # Default to 2 instead of 3
    
    try:
        match_threshold = int(match_threshold)
    except ValueError:
        match_threshold = 2  # Default to 2 if invalid value
    
    # Start with base query
    query = Product.query
    
    # Apply filters
    if search_term:
        query = query.filter(or_(
            Product.name.ilike(f'%{search_term}%'),
            Product.product_id.ilike(f'%{search_term}%'),
            Product.description.ilike(f'%{search_term}%')
        ))
    if category:
        query = query.filter(Product.category.ilike(f'%{category}%'))
    if manufacturer:
        query = query.filter(Product.manufacturer.ilike(f'%{manufacturer}%'))
    if min_price:
        try:
            query = query.filter(Product.price >= float(min_price))
        except ValueError:
            flash('Invalid minimum price', 'error')
    if max_price:
        try:
            query = query.filter(Product.price <= float(max_price))
        except ValueError:
            flash('Invalid maximum price', 'error')
    
    # Get all categories and manufacturers for filter dropdowns
    categories = db.session.query(Product.category).distinct().all()
    manufacturers = db.session.query(Product.manufacturer).distinct().all()
    
    # Execute query
    products = query.all()
    
    # For finding products with matching attributes
    matching_groups = None
    if find_matching == 'on':
        # Group products by combination of attributes
        groups = {}
        for product in products:
            # Normalize values to handle None values and case sensitivity
            manufacturer = (str(product.manufacturer).strip().lower() if product.manufacturer else 'none')
            dimensions = (str(product.dimensions).strip().lower() if product.dimensions else 'none')
            
            # Round price to 2 decimal places to avoid floating point comparison issues
            price = round(float(product.price), 2) if product.price is not None else 0.0
            
            # Create a key based on the attributes we want to match
            key = (manufacturer, dimensions, price)
            
            if key not in groups:
                groups[key] = []
            groups[key].append(product)
        
        # Filter groups with count >= threshold
        matching_groups = []
        for key, group in groups.items():
            if len(group) >= match_threshold:
                matching_groups.append({
                    'attributes': {
                        'manufacturer': group[0].manufacturer if group[0].manufacturer else 'N/A',
                        'dimensions': group[0].dimensions if group[0].dimensions else 'N/A',
                        'price': f"${group[0].price:.2f}" if group[0].price else '$0.00',
                    },
                    'count': len(group),
                    'products': group
                })
        
        # Sort by group size (descending)
        matching_groups.sort(key=lambda x: x['count'], reverse=True)
        
        # Check if there are any groups with 5 or more products
        has_groups_with_five_or_more = any(group['count'] >= 5 for group in matching_groups)
        if matching_groups and not has_groups_with_five_or_more:
            flash('No groups with 5 or more matching products found. Showing all groups that match the threshold.', 'info')
    
    return render_template('search.html', 
                          products=products,
                          categories=[c[0] for c in categories if c[0]],
                          manufacturers=[m[0] for m in manufacturers if m[0]],
                          search_term=search_term,
                          category=category,
                          manufacturer=manufacturer,
                          min_price=min_price,
                          max_price=max_price,
                          find_matching=find_matching,
                          match_threshold=match_threshold,
                          matching_groups=matching_groups)

# New API endpoints for enhanced cart functionality
@app.route('/get_alternative_products/<product_id>')
def get_alternative_products(product_id):
    """Get alternative products based on similar attributes"""
    try:
        # Find the original product
        original_product = Product.query.filter_by(product_id=str(product_id)).first()
        
        if not original_product:
            return jsonify({'alternatives': [], 'message': 'Product not found'})
        
        # Find similar products based on category, material, manufacturer, etc.
        # Use a query builder approach to construct the query with filters
        query = Product.query.filter(Product.product_id != str(product_id))  # Exclude the current product
        
        # Filter by category if available
        if original_product.category:
            query = query.filter(Product.category == original_product.category)
        
        # Filter by subcategory if available
        if original_product.subcategory:
            query = query.filter(Product.subcategory == original_product.subcategory)
            
        # Get alternative products ordered by similarity in price
        alternatives = query.all()
        
        # If we don't have enough alternatives, try a broader search
        if len(alternatives) < 3:
            query = Product.query.filter(Product.product_id != str(product_id))
            if original_product.category:
                query = query.filter(Product.category == original_product.category)
            alternatives = query.all()
            
        # If we still don't have enough, try an even broader search
        if len(alternatives) < 3:
            original_price = original_product.price or 0
            # Find products with similar price range (±30%)
            min_price = original_price * 0.7
            max_price = original_price * 1.3
            alternatives = Product.query.filter(
                Product.product_id != str(product_id),
                Product.price.between(min_price, max_price)
            ).order_by(func.abs(Product.price - original_price)).limit(10).all()
        
        # Format the response
        result = []
        for product in alternatives:
            result.append({
                'product_id': product.product_id,
                'name': product.name,
                'price': float(product.price) if product.price else 0.0,
                'manufacturer': product.manufacturer,
                'category': product.category,
                'subcategory': product.subcategory,
                'material': product.material,
                'dimensions': product.dimensions,
                'warranty': product.warranty,
                'stock': product.stock_quantity
            })
            
        return jsonify({'alternatives': result})
        
    except Exception as e:
        app.logger.error(f"Error in get_alternative_products: {str(e)}")
        return jsonify({'alternatives': [], 'message': f'Error: {str(e)}'})

@app.route('/swap_product/<old_id>/<new_id>', methods=['POST'])
def swap_product(old_id, new_id):
    """Swap a product in the cart with an alternative"""
    try:
        from modules.utils import Cart
        
        # Get cart items
        cart_items = Cart.get_items()
        
        # Check if old product exists in cart
        old_item = None
        for item in cart_items:
            if str(item['product_id']) == str(old_id):
                old_item = item
                break
                
        if not old_item:
            return jsonify({'success': False, 'message': 'Original product not found in cart'})
            
        # Get the new product
        new_product = Product.query.filter_by(product_id=str(new_id)).first()
        if not new_product:
            return jsonify({'success': False, 'message': 'Alternative product not found'})
            
        # Remember the quantity
        quantity = old_item['quantity']
            
        # Remove the old product
        Cart.remove_item(old_id)
        
        # Add the new product
        Cart.add_item(
            new_product.product_id,
            new_product.name,
            float(new_product.price) if new_product.price else 0.0,
            quantity
        )
        
        return jsonify({
            'success': True, 
            'message': 'Product successfully swapped',
            'new_product_name': new_product.name,
            'new_product_price': float(new_product.price) if new_product.price else 0.0
        })
        
    except Exception as e:
        app.logger.error(f"Error in swap_product: {str(e)}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/get_product_stock/<product_id>')
def get_product_stock(product_id):
    """Get current stock level for a product"""
    try:
        product = Product.query.filter_by(product_id=str(product_id)).first()
        
        if not product:
            return jsonify({'stock': 0, 'message': 'Product not found'})
            
        return jsonify({
            'stock': product.stock_quantity or 0,
            'status': 'in_stock' if (product.stock_quantity or 0) > 0 else 'out_of_stock'
        })
        
    except Exception as e:
        app.logger.error(f"Error in get_product_stock: {str(e)}")
        return jsonify({'stock': 0, 'message': f'Error: {str(e)}'})

@app.route('/get_product_location/<product_id>')
def get_product_location(product_id):
    """Get warehouse location for a product"""
    try:
        product = Product.query.filter_by(product_id=str(product_id)).first()
        
        if not product:
            return jsonify({'location': 'Unknown', 'message': 'Product not found'})
            
        # In a real system, this would come from a warehouse management system
        # For demo purposes, generate a warehouse location based on product ID
        # This is just a mock implementation
        product_id_hash = sum(ord(c) for c in str(product_id))
        warehouse = ['A', 'B', 'C', 'D'][product_id_hash % 4]
        aisle = (product_id_hash % 20) + 1
        bin_number = (product_id_hash % 50) + 1
            
        return jsonify({
            'location': f'Warehouse {warehouse}, Aisle {aisle}, Bin {bin_number}',
            'warehouse': warehouse,
            'aisle': aisle,
            'bin': bin_number
        })
        
    except Exception as e:
        app.logger.error(f"Error in get_product_location: {str(e)}")
        return jsonify({'location': 'Unknown', 'message': f'Error: {str(e)}'})

@app.route('/get_product_details/<product_id>')
def get_product_details(product_id):
    """Get detailed information about a product for comparison"""
    try:
        product = Product.query.filter_by(product_id=str(product_id)).first()
        
        if not product:
            return jsonify({'error': 'Product not found'})
            
        # Get cart items to check if this product is in the cart
        from modules.utils import Cart
        cart_items = Cart.get_items()
        cart_item = next((item for item in cart_items if str(item['product_id']) == str(product_id)), None)
        
        # Check if this is a cart item or a database product
        if cart_item:
            # For items already in the cart
            return jsonify({
                'product_id': cart_item['product_id'],
                'product_name': cart_item['product_name'],
                'price': float(cart_item['unit_price']),
                'quantity': cart_item['quantity'],
                'manufacturer': cart_item.get('manufacturer', 'N/A'),
                'material': cart_item.get('material', 'N/A'),
                'dimensions': cart_item.get('dimensions', 'N/A'),
                'warranty': cart_item.get('warranty', 'N/A'),
                'category': cart_item.get('category', 'N/A'),
                'subcategory': cart_item.get('subcategory', 'N/A'),
                'stock': cart_item.get('stock_quantity', 'Unknown')
            })
        else:
            # For database products
            return jsonify({
                'product_id': product.product_id,
                'product_name': product.name,
                'price': float(product.price) if product.price else 0.0,
                'manufacturer': product.manufacturer or 'N/A',
                'material': product.material or 'N/A',
                'dimensions': product.dimensions or 'N/A',
                'warranty': product.warranty or 'N/A',
                'category': product.category or 'N/A',
                'subcategory': product.subcategory or 'N/A',
                'stock': product.stock_quantity or 0,
                'description': product.description or 'N/A'
            })
        
    except Exception as e:
        app.logger.error(f"Error in get_product_details: {str(e)}")
        return jsonify({'error': f'Error: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)