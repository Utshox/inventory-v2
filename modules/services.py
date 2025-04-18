import pandas as pd
from modules.models import db, Product, Invoice, InvoiceItem, Customer
from typing import List, Optional, Dict, Any
from sqlalchemy.orm.exc import NoResultFound
from datetime import datetime
from modules.utils import Cart
from flask import session

class ProductService:
    """
    Service class for product-related operations.
    Implements proper OOP design for product management.
    """
    
    @staticmethod
    def get_all_products(page: int = 1, per_page: int = 10) -> Dict[str, Any]:
        """
        Get all products with pagination
        
        Args:
            page: Page number (default: 1)
            per_page: Number of items per page (default: 10)
            
        Returns:
            Dict containing paginated products
        """
        return Product.query.paginate(page=page, per_page=per_page)
    
    @staticmethod
    def get_product_by_id(product_id: str) -> Optional[Product]:
        """
        Get a product by its ID
        
        Args:
            product_id: The product ID
            
        Returns:
            Product object if found, None otherwise
        """
        return Product.query.filter_by(product_id=product_id).first()
    
    @staticmethod
    def create_product(product_data: Dict[str, Any]) -> Product:
        """
        Create a new product
        
        Args:
            product_data: Dictionary containing product data
            
        Returns:
            The created Product object
        """
        product = Product(
            product_id=product_data.get('product_id'),
            name=product_data.get('name'),
            price=product_data.get('price', 0.0),
            manufacturer=product_data.get('manufacturer'),
            category=product_data.get('category'),
            subcategory=product_data.get('subcategory'),
            material=product_data.get('material'),
            dimensions=product_data.get('dimensions'),
            warranty=product_data.get('warranty'),
            description=product_data.get('description'),
            stock_quantity=product_data.get('stock_quantity', 0)
        )
        db.session.add(product)
        db.session.commit()
        return product
    
    @staticmethod
    def update_product(product_id: str, product_data: Dict[str, Any]) -> Optional[Product]:
        """
        Update an existing product
        
        Args:
            product_id: The product ID
            product_data: Dictionary containing updated product data
            
        Returns:
            The updated Product object if found, None otherwise
        """
        product = ProductService.get_product_by_id(product_id)
        if not product:
            return None
            
        # Update product fields
        for key, value in product_data.items():
            if hasattr(product, key):
                setattr(product, key, value)
                
        db.session.commit()
        return product
    
    @staticmethod
    def delete_product(product_id: str) -> bool:
        """
        Delete a product
        
        Args:
            product_id: The product ID
            
        Returns:
            True if deleted successfully, False otherwise
        """
        product = ProductService.get_product_by_id(product_id)
        if not product:
            return False
            
        db.session.delete(product)
        db.session.commit()
        return True
    
    @staticmethod
    def search_products(query: str) -> List[Product]:
        """
        Search for products by name, ID, or category
        
        Args:
            query: The search query
            
        Returns:
            List of matching products
        """
        return Product.query.filter(
            (Product.name.ilike(f'%{query}%')) |
            (Product.product_id.ilike(f'%{query}%')) |
            (Product.category.ilike(f'%{query}%'))
        ).all()
    
    @staticmethod
    def import_products_from_dataframe(df) -> int:
        """
        Import products from a pandas DataFrame
        
        Args:
            df: pandas DataFrame containing product data
            
        Returns:
            Number of imported products
        """
        counter = 0
        for _, row in df.iterrows():
            # Skip if no product ID or price
            if 'Product ID' not in row or 'Unit Price' not in row or pd.isna(row['Product ID']) or pd.isna(row['Unit Price']):
                continue
                
            product_id = str(row['Product ID'])
            
            # Check if product already exists
            existing_product = ProductService.get_product_by_id(product_id)
            if existing_product:
                continue
                
            # Create new product
            product_data = {
                'product_id': product_id,
                'name': row['Product Name'] if 'Product Name' in row else 'Unknown',
                'price': float(row['Unit Price']) if pd.notna(row['Unit Price']) else 0.0,
                'manufacturer': row['Manufacturer'] if 'Manufacturer' in row and pd.notna(row['Manufacturer']) else None,
                'category': row['Category'] if 'Category' in row and pd.notna(row['Category']) else None,
                'subcategory': row['Subcategory'] if 'Subcategory' in row and pd.notna(row['Subcategory']) else None,
                'material': row['Material'] if 'Material' in row and pd.notna(row['Material']) else None,
                'dimensions': row['Size/Dimensions'] if 'Size/Dimensions' in row and pd.notna(row['Size/Dimensions']) else None,
                'warranty': int(row['Warranty Information']) if 'Warranty Information' in row and pd.notna(row['Warranty Information']) else None,
                'stock_quantity': 20  # Default stock
            }
            
            try:
                ProductService.create_product(product_data)
                counter += 1
            except Exception as e:
                print(f"Error creating product {product_id}: {str(e)}")
                db.session.rollback()
                
        return counter

class CustomerService:
    """
    Service class for customer-related operations.
    Implements proper OOP design for customer management.
    """
    
    @staticmethod
    def get_all_customers(page: int = 1, per_page: int = 10):
        """Get all customers with pagination"""
        return Customer.query.paginate(page=page, per_page=per_page)
    
    @staticmethod
    def get_customer_by_id(customer_id: int) -> Optional[Customer]:
        """Get a customer by ID"""
        return Customer.query.get(customer_id)
    
    @staticmethod
    def create_customer(customer_data: Dict[str, Any]) -> Customer:
        """Create a new customer"""
        customer = Customer(
            name=customer_data.get('name'),
            email=customer_data.get('email'),
            phone=customer_data.get('phone'),
            address=customer_data.get('address'),
            city=customer_data.get('city'),
            state=customer_data.get('state'),
            country=customer_data.get('country'),
            postal_code=customer_data.get('postal_code')
        )
        db.session.add(customer)
        db.session.commit()
        return customer
    
    @staticmethod
    def update_customer(customer_id: int, customer_data: Dict[str, Any]) -> Optional[Customer]:
        """Update an existing customer"""
        customer = CustomerService.get_customer_by_id(customer_id)
        if not customer:
            return None
            
        # Update customer fields
        for key, value in customer_data.items():
            if hasattr(customer, key):
                setattr(customer, key, value)
                
        db.session.commit()
        return customer
    
    @staticmethod
    def delete_customer(customer_id: int) -> bool:
        """Delete a customer"""
        customer = CustomerService.get_customer_by_id(customer_id)
        if not customer:
            return False
            
        db.session.delete(customer)
        db.session.commit()
        return True
    
    @staticmethod
    def search_customers(query: str) -> List[Customer]:
        """Search for customers"""
        return Customer.query.filter(
            (Customer.name.ilike(f'%{query}%')) |
            (Customer.email.ilike(f'%{query}%')) |
            (Customer.phone.ilike(f'%{query}%'))
        ).all()

class InvoiceService:
    """
    Service class for invoice-related operations.
    Implements proper OOP design for invoice management.
    """
    
    @staticmethod
    def get_all_invoices(page: int = 1, per_page: int = 10):
        """Get all invoices with pagination"""
        return Invoice.query.order_by(Invoice.created_at.desc()).paginate(page=page, per_page=per_page)
    
    @staticmethod
    def get_invoice_by_id(invoice_id: int) -> Optional[Invoice]:
        """Get an invoice by ID"""
        return Invoice.query.get(invoice_id)
    
    @staticmethod
    def create_invoice(invoice_data: Dict[str, Any], cart_items: List[Dict[str, Any]]) -> Optional[Invoice]:
        """
        Create a new invoice with items from the cart
        
        Args:
            invoice_data: Dictionary containing invoice data
            cart_items: List of items in the cart
            
        Returns:
            The created Invoice object or None if failed
        """
        try:
            # Validate customer
            customer_id = invoice_data.get('customer_id')
            if not customer_id or not CustomerService.get_customer_by_id(customer_id):
                return None
                
            # Create invoice
            invoice = Invoice(
                customer_id=customer_id,
                total_amount=Cart.get_total(),
                status=invoice_data.get('status', 'pending'),
                payment_method=invoice_data.get('payment_method', 'cash'),
                shipping_address=invoice_data.get('shipping_address', ''),
                notes=invoice_data.get('notes', '')
            )
            db.session.add(invoice)
            db.session.flush()  # Get invoice ID without committing
            
            # Add invoice items
            for item in cart_items:
                invoice_item = InvoiceItem(
                    invoice_id=invoice.id,
                    product_id=item['product_id'],
                    quantity=item['quantity'],
                    unit_price=item['unit_price'],
                    total_price=item['total']
                )
                db.session.add(invoice_item)
                
                # Update product stock (optional)
                product = Product.query.filter_by(product_id=item['product_id']).first()
                if product and product.stock_quantity is not None:
                    product.stock_quantity = max(0, product.stock_quantity - item['quantity'])
            
            db.session.commit()
            return invoice
            
        except Exception as e:
            print(f"Error creating invoice: {str(e)}")
            db.session.rollback()
            return None
    
    @staticmethod
    def update_invoice_status(invoice_id: int, status: str) -> bool:
        """Update an invoice's status"""
        invoice = InvoiceService.get_invoice_by_id(invoice_id)
        if not invoice:
            return False
            
        invoice.status = status
        db.session.commit()
        return True
    
    @staticmethod
    def delete_invoice(invoice_id: int) -> bool:
        """Delete an invoice"""
        invoice = InvoiceService.get_invoice_by_id(invoice_id)
        if not invoice:
            return False
            
        # Delete related invoice items first
        InvoiceItem.query.filter_by(invoice_id=invoice_id).delete()
        db.session.delete(invoice)
        db.session.commit()
        return True
    
    @staticmethod
    def get_invoice_items(invoice_id: int) -> List[InvoiceItem]:
        """Get all items for an invoice"""
        return InvoiceItem.query.filter_by(invoice_id=invoice_id).all()
    
    @staticmethod
    def get_invoice_total(invoice_id: int) -> float:
        """Calculate the total amount for an invoice"""
        items = InvoiceService.get_invoice_items(invoice_id)
        return sum(item.total_price for item in items)
    
    @staticmethod
    def get_customer_invoices(customer_id: int) -> List[Invoice]:
        """Get all invoices for a customer"""
        return Invoice.query.filter_by(customer_id=customer_id).order_by(Invoice.created_at.desc()).all()

class CartService:
    """
    Service class to handle shopping cart operations.
    Implements proper OOP design by encapsulating cart functionality.
    Uses the Cart class from utils.py to manage cart operations.
    """
    
    @staticmethod
    def get_cart_items():
        """Get all items currently in the cart"""
        return Cart.get_items()
    
    @staticmethod
    def add_item(product_id, product_name, unit_price, quantity, additional_details=None):
        """
        Add an item to the cart
        
        Args:
            product_id (str): Product identifier
            product_name (str): Name of the product
            unit_price (float): Price per unit
            quantity (int): Number of units
            additional_details (dict, optional): Additional product information
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Input validation
            if not product_id or not product_name:
                return False
                
            # Validate and sanitize unit price
            try:
                unit_price = float(unit_price) if unit_price else 0.0
                if pd.isna(unit_price) or not pd.np.isfinite(unit_price) or unit_price < 0:
                    unit_price = 0.0
            except (ValueError, TypeError):
                unit_price = 0.0
                
            # Validate and sanitize quantity
            try:
                quantity = int(quantity) if quantity else 1
                if quantity <= 0:
                    quantity = 1
            except (ValueError, TypeError):
                quantity = 1
            
            # Use the Cart class to add the item
            return Cart.add_item(product_id, product_name, unit_price, quantity)
            
        except Exception as e:
            print(f"Error adding item to cart: {str(e)}")
            return False
    
    @staticmethod
    def update_quantity(product_id, quantity):
        """
        Update the quantity of an item in the cart
        
        Args:
            product_id (str): Product identifier
            quantity (int): New quantity
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Validate inputs
            if not product_id:
                return False
                
            # Sanitize quantity
            try:
                quantity = int(quantity) if quantity else 1
                if quantity <= 0:
                    quantity = 1
            except (ValueError, TypeError):
                quantity = 1
            
            # Use the Cart class to update the quantity
            return Cart.update_quantity(product_id, quantity)
            
        except Exception as e:
            print(f"Error updating cart quantity: {str(e)}")
            return False
    
    @staticmethod
    def remove_item(product_id):
        """
        Remove an item from the cart
        
        Args:
            product_id (str): Product identifier
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not product_id:
                return False
            
            # Use the Cart class to remove the item
            return Cart.remove_item(product_id)
            
        except Exception as e:
            print(f"Error removing item from cart: {str(e)}")
            return False
    
    @staticmethod
    def clear_cart():
        """Empty the cart"""
        Cart.clear()
        return True
    
    @staticmethod
    def get_cart_total():
        """Calculate the total price of all items in the cart"""
        return Cart.get_total()
    
    @staticmethod
    def save_to_database(customer_id, user_id=None):
        """
        Save cart as invoice in the database
        
        Args:
            customer_id (int): Customer ID
            user_id (int, optional): User ID who created the invoice
            
        Returns:
            tuple: (success, invoice_id or error message)
        """
        try:
            # Validate customer exists
            customer = Customer.query.get(customer_id)
            if not customer:
                return False, "Customer not found"
                
            # Get cart items
            cart_items = Cart.get_items()
            if not cart_items:
                return False, "Cart is empty"
                
            # Create new invoice
            invoice_number = f"INV-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            total_amount = Cart.get_total()
            
            invoice = Invoice(
                invoice_number=invoice_number,
                customer_id=customer_id,
                date_created=datetime.utcnow(),
                due_date=datetime.utcnow(),  # Can be adjusted based on payment terms
                status='pending',
                total=total_amount,
                created_by=user_id
            )
            
            db.session.add(invoice)
            db.session.flush()  # Get invoice ID without committing
            
            # Add invoice items
            for item in cart_items:
                # Find product in database
                product = Product.query.filter_by(product_id=item['product_id']).first()
                if not product:
                    continue
                    
                # Create invoice item
                invoice_item = InvoiceItem(
                    invoice_id=invoice.id,
                    product_id=product.id,
                    quantity=item['quantity'],
                    unit_price=item['unit_price'],
                    discount=0.0,  # Default discount
                    line_total=item['total']
                )
                
                # Update product stock
                product.stock_quantity = max(0, product.stock_quantity - item['quantity'])
                
                db.session.add(invoice_item)
                
            # Commit transaction
            db.session.commit()
            
            # Clear cart after saving
            Cart.clear()
            
            return True, invoice.id
            
        except Exception as e:
            db.session.rollback()
            print(f"Error saving cart to database: {str(e)}")
            return False, str(e)