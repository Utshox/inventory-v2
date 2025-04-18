from flask import Blueprint, jsonify, request, current_app
from flask_login import login_required, current_user
from modules.models import db, Product, Customer, Invoice, ApiKey
from functools import wraps
import datetime

# Create API blueprint
api = Blueprint('api', __name__)

# API Key authentication decorator
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get API key from header or query parameter
        api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
        
        if not api_key:
            return jsonify({'error': 'API key is required'}), 401
            
        # Check if API key exists and is active
        key = ApiKey.query.filter_by(key_value=api_key, is_active=True).first()
        if not key:
            return jsonify({'error': 'Invalid or inactive API key'}), 401
            
        # Update last used timestamp
        key.last_used = datetime.datetime.utcnow()
        db.session.commit()
        
        return f(*args, **kwargs)
    return decorated_function

# Root API endpoint - API information
@api.route('/')
def api_info():
    """Return API information and available endpoints"""
    return jsonify({
        'name': 'Door Inventory Management API',
        'version': 'v1',
        'description': 'RESTful API for Door Inventory Management System',
        'endpoints': {
            'products': '/api/v1/products',
            'customers': '/api/v1/customers',
            'stats': '/api/v1/stats'
        },
        'documentation': '/api_docs'
    })

# Products endpoints
@api.route('/products')
@require_api_key
def get_products():
    """Get all products or filter by query parameters"""
    # Get query parameters
    category = request.args.get('category')
    manufacturer = request.args.get('manufacturer')
    material = request.args.get('material')
    min_price = request.args.get('min_price')
    max_price = request.args.get('max_price')
    
    # Start with base query
    query = Product.query
    
    # Apply filters
    if category:
        query = query.filter(Product.category.ilike(f'%{category}%'))
    if manufacturer:
        query = query.filter(Product.manufacturer.ilike(f'%{manufacturer}%'))
    if material:
        query = query.filter(Product.material.ilike(f'%{material}%'))
    if min_price:
        query = query.filter(Product.price >= float(min_price))
    if max_price:
        query = query.filter(Product.price <= float(max_price))
    
    # Get pagination parameters
    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 50, type=int), 100)  # Max 100 items per page
    
    # Execute paginated query
    products_page = query.paginate(page=page, per_page=per_page)
    
    # Build response
    response = {
        'products': [product.to_dict() for product in products_page.items],
        'pagination': {
            'page': page,
            'per_page': per_page,
            'total_pages': products_page.pages,
            'total_items': products_page.total
        }
    }
    
    return jsonify(response)

@api.route('/products/<string:product_id>')
@require_api_key
def get_product(product_id):
    """Get a specific product by ID"""
    product = Product.query.filter_by(product_id=product_id).first()
    
    if not product:
        return jsonify({'error': 'Product not found'}), 404
        
    return jsonify(product.to_dict())

@api.route('/products', methods=['POST'])
@require_api_key
def create_product():
    """Create a new product"""
    data = request.get_json()
    
    # Validate required fields
    required_fields = ['product_id', 'name', 'price']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    
    # Check if product already exists
    if Product.query.filter_by(product_id=data['product_id']).first():
        return jsonify({'error': 'Product with this ID already exists'}), 409
    
    # Create new product
    product = Product(
        product_id=data['product_id'],
        name=data['name'],
        price=float(data['price']),
        manufacturer=data.get('manufacturer'),
        category=data.get('category'),
        subcategory=data.get('subcategory'),
        material=data.get('material'),
        dimensions=data.get('dimensions'),
        warranty=data.get('warranty'),
        description=data.get('description'),
        stock_quantity=data.get('stock_quantity', 0)
    )
    
    db.session.add(product)
    db.session.commit()
    
    return jsonify({
        'message': 'Product created successfully',
        'product': product.to_dict()
    }), 201

@api.route('/products/<string:product_id>', methods=['PUT', 'PATCH'])
@require_api_key
def update_product(product_id):
    """Update an existing product"""
    product = Product.query.filter_by(product_id=product_id).first()
    
    if not product:
        return jsonify({'error': 'Product not found'}), 404
        
    data = request.get_json()
    
    # Update product fields
    if 'name' in data:
        product.name = data['name']
    if 'price' in data:
        product.price = float(data['price'])
    if 'manufacturer' in data:
        product.manufacturer = data['manufacturer']
    if 'category' in data:
        product.category = data['category']
    if 'subcategory' in data:
        product.subcategory = data['subcategory']
    if 'material' in data:
        product.material = data['material']
    if 'dimensions' in data:
        product.dimensions = data['dimensions']
    if 'warranty' in data:
        product.warranty = data['warranty']
    if 'description' in data:
        product.description = data['description']
    if 'stock_quantity' in data:
        product.stock_quantity = data['stock_quantity']
    
    # Update the last_updated timestamp
    product.last_updated = datetime.datetime.utcnow()
    
    db.session.commit()
    
    return jsonify({
        'message': 'Product updated successfully',
        'product': product.to_dict()
    })

@api.route('/products/<string:product_id>', methods=['DELETE'])
@require_api_key
def delete_product(product_id):
    """Delete a product"""
    product = Product.query.filter_by(product_id=product_id).first()
    
    if not product:
        return jsonify({'error': 'Product not found'}), 404
        
    db.session.delete(product)
    db.session.commit()
    
    return jsonify({'message': 'Product deleted successfully'})

# Customer endpoints
@api.route('/customers')
@require_api_key
def get_customers():
    """Get all customers with pagination"""
    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 50, type=int), 100)  # Max 100 items per page
    
    customers_page = Customer.query.paginate(page=page, per_page=per_page)
    
    response = {
        'customers': [customer.to_dict() for customer in customers_page.items],
        'pagination': {
            'page': page,
            'per_page': per_page,
            'total_pages': customers_page.pages,
            'total_items': customers_page.total
        }
    }
    
    return jsonify(response)

@api.route('/customers/<int:customer_id>')
@require_api_key
def get_customer(customer_id):
    """Get a specific customer by ID"""
    customer = Customer.query.get(customer_id)
    
    if not customer:
        return jsonify({'error': 'Customer not found'}), 404
        
    return jsonify(customer.to_dict())

@api.route('/customers', methods=['POST'])
@require_api_key
def create_customer():
    """Create a new customer"""
    data = request.get_json()
    
    # Validate required fields
    required_fields = ['name', 'email']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    
    # Check if email already exists
    if Customer.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'Customer with this email already exists'}), 409
    
    # Create new customer
    customer = Customer(
        name=data['name'],
        email=data['email'],
        phone=data.get('phone'),
        address=data.get('address'),
        notes=data.get('notes')
    )
    
    db.session.add(customer)
    db.session.commit()
    
    return jsonify({
        'message': 'Customer created successfully',
        'customer': customer.to_dict()
    }), 201

@api.route('/customers/<int:customer_id>', methods=['PUT', 'PATCH'])
@require_api_key
def update_customer(customer_id):
    """Update an existing customer"""
    customer = Customer.query.get(customer_id)
    
    if not customer:
        return jsonify({'error': 'Customer not found'}), 404
        
    data = request.get_json()
    
    # Update customer fields
    if 'name' in data:
        customer.name = data['name']
    if 'email' in data:
        # Check if email already exists
        existing = Customer.query.filter_by(email=data['email']).first()
        if existing and existing.id != customer.id:
            return jsonify({'error': 'Email already in use by another customer'}), 409
        customer.email = data['email']
    if 'phone' in data:
        customer.phone = data['phone']
    if 'address' in data:
        customer.address = data['address']
    if 'notes' in data:
        customer.notes = data['notes']
    
    # Update the last_updated timestamp
    customer.last_updated = datetime.datetime.utcnow()
    
    db.session.commit()
    
    return jsonify({
        'message': 'Customer updated successfully',
        'customer': customer.to_dict()
    })

@api.route('/customers/<int:customer_id>', methods=['DELETE'])
@require_api_key
def delete_customer(customer_id):
    """Delete a customer"""
    customer = Customer.query.get(customer_id)
    
    if not customer:
        return jsonify({'error': 'Customer not found'}), 404
        
    db.session.delete(customer)
    db.session.commit()
    
    return jsonify({'message': 'Customer deleted successfully'})

# Statistics endpoints
@api.route('/stats')
@require_api_key
def get_stats():
    """Get system statistics"""
    # Product stats
    product_count = Product.query.count()
    low_stock_count = Product.query.filter(Product.stock_quantity < 5).count()
    
    # Category distribution
    category_stats = db.session.query(
        Product.category, 
        db.func.count(Product.id)
    ).group_by(Product.category).all()
    
    # Format category stats, handling None
    categories = [{'name': cat or 'Uncategorized', 'count': count} 
                  for cat, count in category_stats]
    
    # Customer stats
    customer_count = Customer.query.count()
    
    # Invoice stats (if implemented)
    invoice_count = Invoice.query.count() if hasattr(db.Model, 'Invoice') else 0
    total_sales = db.session.query(db.func.sum(Invoice.total)).scalar() or 0 if hasattr(db.Model, 'Invoice') else 0
    
    return jsonify({
        'products': {
            'total': product_count,
            'low_stock': low_stock_count,
            'categories': categories
        },
        'customers': {
            'total': customer_count
        },
        'sales': {
            'invoice_count': invoice_count,
            'total_sales': float(total_sales)
        }
    })